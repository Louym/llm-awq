import torch
import torch.nn as nn
from typing import Optional, Tuple
from typing_extensions import Unpack
from awq.quantize import W8A8OF16LinearDynamicInputScale
from tinychat.utils.input_metadata import ActivationBuffer
from transformers.modeling_outputs import BaseModelOutputWithPast
import awq_inference_engine
import math


def precompute_freqs(
    dim: int, end: int, theta: float = 10000.0, scale: float = 1.0, device=None
):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float().to(device) / dim))
    seq = torch.arange(end, dtype=inv_freq.dtype, device=device)
    freqs = torch.einsum("i , j -> i j", seq, inv_freq)
    freqs = freqs.reshape(freqs.shape[0], 1, 1, -1)
    return torch.cat((freqs, freqs), dim=-1)


class QuantLlamaModel(nn.Module):
    def __init__(self, module: nn.Module, bsz=64, seqlen=1024):
        super().__init__()

        config = module.config
        self.config = module.config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx).cuda()

        self.layers = [QuantLlamaDecoderLayer(layer) for layer in module.layers]
        self.norm = module.norm
        self.rotary_emb = module.rotary_emb

        self.input_dim=module.layers[0].self_attn.q_proj.in_features
        self.hidden_size = module.layers[0].self_attn.q_proj.out_features
        self.intermediate_size = module.layers[0].mlp.up_proj.out_features
        self.attn_group=module.layers[0].self_attn.q_proj.out_features//module.layers[0].self_attn.k_proj.out_features
        self.buffer = ActivationBuffer(module, intermediate_size=self.intermediate_size, hidden_size=self.hidden_size, input_dim=self.input_dim, attn_group=self.attn_group)
        self.bsz = bsz
        self.seqlen = seqlen
        self.buffer.allocate_activation_buffer(self.bsz * self.seqlen)

        self.theta=config.rope_theta
        dim=module.layers[0].self_attn.head_dim
        self.freqs=precompute_freqs(dim, seqlen+1, self.theta, device="cuda")

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs,
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            raise ValueError("inputs_embeds is None")

        # TODO do this conversion somewhere else
        inputs_embeds = inputs_embeds.to(torch.float16)

        bsz, seqlen, _ = inputs_embeds.shape
        
        if self.bsz != bsz or self.seqlen != seqlen:
            self.buffer.allocate_activation_buffer(bsz * seqlen)
            self.bsz = bsz
            self.seqlen = seqlen

        # if use_cache and past_key_values is None:
        #     past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # causal_mask = self._update_causal_mask(
        #     attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        # )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states, self.buffer, self.freqs, attention_mask, past_key_values, bsz, seqlen
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )



class QuantLlamaDecoderLayer(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.embed_dim = module.self_attn.q_proj.in_features
        self.hidden_size = module.hidden_size

        self.self_attn = QuantLlamaAttention(module.self_attn)

        self.mlp = QuantLlamaMLP(module.mlp)
        self.input_layernorm = RMSNormGeneral(module.input_layernorm.weight.data, module.input_layernorm.variance_epsilon, True).cuda()
        self.post_attention_layernorm = RMSNormGeneral(module.post_attention_layernorm.weight.data, module.post_attention_layernorm.variance_epsilon, True).cuda()

        # TODO where is this used?
        self.quant = self.invoke_quant_norm
    
    def invoke_quant_norm(self, buffer, normfn_output):
        awq_inference_engine.invoke_quant(
            buffer.quantized_hidden_states_buffer,
            normfn_output,
            buffer.quantized_scale_buffer,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        buffer: ActivationBuffer,
        freqs: torch.Tensor,
        attention_mask,
        kv_cache,
        bsz,
        seqlen,
    ) -> Tuple[torch.FloatTensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(
            hidden_states.reshape(-1, self.embed_dim),
            buffer.quantized_input_buffer,
            buffer.quantized_scale_buffer,
        )

        # TODO verify in_out_fc2_act_buffer is the correct buffer

        # INT8 -> FP16
        self.self_attn(buffer, bsz, seqlen, freqs, attention_mask, kv_cache)
        hidden_states = (
            residual.reshape(-1, self.embed_dim) + buffer.in_out_fc2_act_buffer
        )
        # Fully Connected
        residual = hidden_states

        hidden_states = self.post_attention_layernorm(
            hidden_states.reshape(-1, self.embed_dim),
            buffer.quantized_input_buffer,
            buffer.quantized_scale_buffer,
        )
        # INT8 -> FP16
        self.mlp(buffer)
        hidden_states = (
            residual.reshape(-1, self.embed_dim) + buffer.in_out_fc2_act_buffer
        )
        return hidden_states


class QuantLlamaMLP(nn.Module):
    def __init__(self, module, init_only=False):
        super().__init__()
        config = module.config
        self.config = module.config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gateup_proj = W8A8OF16LinearDynamicInputScale.from_linear(
            [module.gate_proj, module.up_proj], init_only=init_only
        )
        self.down_proj = W8A8OF16LinearDynamicInputScale.from_linear(module.down_proj, init_only=init_only, fc1=False)

    def forward(self, buffer: ActivationBuffer) -> torch.Tensor:

        self.gateup_proj(
            buffer.quantized_input_buffer,
            buffer.quantized_scale_buffer,
            buffer.gateup_buffer,
        )
        #FP16 in, INT8 out
        # out=nn.functional.gelu(buffer.gateup_buffer[:,:self.intermediate_size], approximate="tanh") * buffer.gateup_buffer[:,self.intermediate_size:]
        # buffer.in_out_fc2_act_buffer=self.down_proj(out)
        awq_inference_engine.silu_and_mul_quant(buffer.quantized_mlp_act_buffer, buffer.gateup_buffer, buffer.quantized_scale_buffer, buffer.tmp)
        # INT8 in, FP16 out
        self.down_proj(
            buffer.quantized_mlp_act_buffer,
            buffer.quantized_scale_buffer,
            buffer.in_out_fc2_act_buffer,
        )

        # down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        # return down_proj


class QuantLlamaAttention(nn.Module):

    def __init__(
        self,
        module,
        init_only=False,
    ):
        super().__init__()
        self.config = module.config
        self.num_heads = module.num_heads
        self.head_dim = module.head_dim
        self.num_key_value_heads = module.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.qkv_proj = W8A8OF16LinearDynamicInputScale.from_linear(
            [module.q_proj, module.k_proj, module.v_proj], init_only=init_only
        )
        self.out_proj = W8A8OF16LinearDynamicInputScale.from_linear(
            module.o_proj, init_only=init_only
        )
        
        self.invoke_quant = self.invoke_quant_wo
        
        self.rotary_emb = module.rotary_emb
        self.attn_softclamp=50 
        self.q_dim=self.head_dim*self.num_heads
        self.kv_dim=self.q_dim//self.num_key_value_groups
    def invoke_quant_wo(self, buffer, attn_output):
        awq_inference_engine.invoke_quant(
            buffer.quantized_hidden_states_buffer,
            attn_output,
            buffer.quantized_scale_buffer,
        )

    # Adapted from transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward
    def forward(
        self, buffer: ActivationBuffer, bsz=64, seqlen=1024, freqs=None, mask=None, kv_cache=None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # qkv
        self.qkv_proj(
            buffer.quantized_input_buffer,
            buffer.quantized_scale_buffer,
            buffer.qkv_proj_act_buffer,
        )
        q, k, v = buffer.qkv_proj_act_buffer.split(
            [self.q_dim, self.kv_dim, self.kv_dim], dim=-1
        )
        q = q.reshape(bsz, seqlen, self.num_heads, self.head_dim).contiguous()
        k = k.reshape(bsz, seqlen, self.num_key_value_heads, self.head_dim).contiguous()
        v = v.reshape(bsz, seqlen, self.num_key_value_heads, self.head_dim).transpose(1,2).contiguous()
        if freqs is not None:
            q = awq_inference_engine.fused_rope_with_pos_forward_func(q, freqs, True).transpose(1,2)
            k = awq_inference_engine.fused_rope_with_pos_forward_func(k, freqs, True).transpose(1,2)
        if kv_cache is not None:
            k=torch.cat([kv_cache[0],k],dim=-2).contiguous()
            v=torch.cat([kv_cache[1],v],dim=-2).contiguous()
        
        attn_output=attention(q,k,v,mask,self.attn_softclamp)
        attn_output = attn_output.reshape(bsz * seqlen, -1)
        # FP16 -> int8
        self.invoke_quant(buffer, attn_output)
        # INT8 in, FP16 out
        self.out_proj(
            buffer.quantized_hidden_states_buffer,
            buffer.quantized_scale_buffer,
            buffer.in_out_fc2_act_buffer,
        )


class RMSNormGeneral(nn.Module):
    """RMS normalization (w/ per-token or per-tensor quant).
    """

    def __init__(
        self,
        weight: torch.tensor,
        eps: float = 1e-6,
        use_per_token_quant: bool = True,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.variance_epsilon = eps
        self.use_per_token_quant = use_per_token_quant

    def forward(
        self,
        x: torch.Tensor,
        quantized_hidden_states_buffer: torch.Tensor,
        quantized_scale_buffer: torch.Tensor,
        quantized_sum_buffer: torch.Tensor = None,
    ) -> torch.Tensor:
        # quantized_sum_buffer is not used, only to keep the consistency of the interface
        
        awq_inference_engine.rms_norm_general(
            quantized_hidden_states_buffer,
            x,
            self.weight.data,
            quantized_scale_buffer,
            self.variance_epsilon,
            self.use_per_token_quant,
        )


def attention(q, k, v, mask=None, softcap=0):
    num_key_value_groups=q.shape[1]//k.shape[1]
    if num_key_value_groups>1:
        k=torch.repeat_interleave(
                k, dim=1, repeats=num_key_value_groups
            )
        v=torch.repeat_interleave(
                v, dim=1, repeats=num_key_value_groups
            )
    attn_weights = q@k.transpose(2, 3) / math.sqrt(
        q.shape[-1]
    )
    
    # Soft capping
    if softcap!=0:
        attn_weights = attn_weights / softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap

    # Apply the softmax / dropout
    if mask is not None:
        attn_weights = attn_weights + mask
    # [Batch_Size, Num_Heads_Q, Full_Seq_Len, Full_Seq_Len]
    attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(
        q.dtype
    )
    # Multiply by the values. [Batch_Size, Num_Heads_Q, Full_Seq_Len, Full_Seq_Len] x [Batch_Size, Num_Heads_KV, Full_Seq_Len, Head_Dim] -> [Batch_Size, Num_Heads_Q, Full_Seq_Len, Head_Dim]
    attn_output = torch.matmul(attn_weights, v)

    # Make sure the sequence length is the second dimension. # [Batch_Size, Num_Heads_Q, Full_Seq_Len, Head_Dim] -> [Batch_Size, Full_Seq_Len, Num_Heads_Q, Head_Dim]
    attn_output = attn_output.transpose(1, 2).flatten(-2,-1)
    return attn_output.contiguous()