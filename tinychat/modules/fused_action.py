import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from awq.quantize import W8A8OF16LinearDynamicInputScale
from tinychat.modules import QuantSiglipMLP
from tinychat.utils.input_metadata import ActivationBuffer
from transformers.modeling_outputs import BaseModelOutput
from typing import Optional, Tuple, Union
from flash_attn import flash_attn_func
import math
CLIP_RANGE = 5
import awq_inference_engine


def precompute_freqs(
    dim: int, end: int, theta: float = 10000.0, scale: float = 1.0, device=None
):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float().to(device) / dim))
    seq = torch.arange(end, dtype=inv_freq.dtype, device=device)
    freqs = torch.einsum("i , j -> i j", seq, inv_freq)
    freqs = freqs.reshape(freqs.shape[0], 1, 1, -1)
    return torch.cat((freqs, freqs), dim=-1)


class QuantMixture(nn.Module):
    def __init__(self, module:nn.Module, bsz=1, seqlen=4):
        super().__init__()
        self.layers = [QuantMixtureEncoderLayer(layer) for layer in module.layers]
        self.input_dim=module.layers[0].self_attn.q_proj.in_features
        self.hidden_size = module.layers[0].self_attn.q_proj.out_features
        self.intermediate_size = module.layers[0].mlp.up_proj.out_features
        self.attn_group=module.layers[0].self_attn.q_proj.out_features//module.layers[0].self_attn.k_proj.out_features
        self.buffer = ActivationBuffer(module, intermediate_size=self.intermediate_size, hidden_size=self.hidden_size, input_dim=self.input_dim, attn_group=self.attn_group)
        self.bsz = bsz
        self.seqlen = seqlen
        self.buffer.allocate_activation_buffer(self.bsz * self.seqlen)
        self.theta=module.layers[0].self_attn.rotary_emb.base
        dim=module.layers[0].self_attn.head_dim
        self.freqs=precompute_freqs(dim, seqlen+1, self.theta, device="cuda")
        self.norm=module.norm
    # Ignore copy
    def forward(
        self,
        attention_mask: torch.Tensor,
        position_ids_all: dict[torch.LongTensor],
        embeds_all: dict[torch.FloatTensor],
        time_cond: Optional[torch.FloatTensor] = None,#Temporiarily Dummy
        final_layer_post_attn_skip_names: Tuple[str, ...] = ("vlm", "proprio"),#Dummy
        kv_caches=None,
        cache_mode: str = "append_non_active",#Dummy
        return_caches: bool = False,
    ) -> Union[Tuple, BaseModelOutput]:
        inputs_embeds=embeds_all["action"].contiguous()
        bsz, seqlen, _ = inputs_embeds.shape
        if self.bsz != bsz or self.seqlen != seqlen:
            self.buffer.allocate_activation_buffer(bsz * seqlen)
            self.bsz = bsz
            self.seqlen = seqlen
        hidden_states = inputs_embeds
        start=position_ids_all["action"][0,0]
        end=position_ids_all["action"][0,-1]
        freqs=self.freqs[start:end]
        for i, encoder_layer in enumerate(self.layers):
            print(i)
            hidden_states = encoder_layer(
                hidden_states, self.buffer, freqs, attention_mask, kv_caches["vlm"].get(i), kv_caches["proprio"].get(i), bsz, seqlen
            )
        return {"action":self.norm(hidden_states.reshape(self.bsz, self.seqlen, -1))}


class QuantMixtureAttention2(nn.Module):
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
        self, buffer: ActivationBuffer, bsz=64, seqlen=1024, freqs=None, mask=None, kv_cache_vlm=None, kv_cache_proprio=None
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
        if kv_cache_vlm is not None:
            k=torch.cat([kv_cache_vlm[0],kv_cache_proprio[0],k],dim=-2).contiguous()
            v=torch.cat([kv_cache_vlm[1],kv_cache_proprio[1],v],dim=-2).contiguous()
        
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


class QuantMixtureMLP(nn.Module):
    def __init__(self, siglipmlp, init_only=False):
        super().__init__()
        self.config = siglipmlp.config
        self.activation_fn = getattr(siglipmlp, "activation_fn", None)
        self.gateup_proj = W8A8OF16LinearDynamicInputScale.from_linear(
            [siglipmlp.gate_proj, siglipmlp.up_proj], init_only=init_only
        )
        self.down_proj = W8A8OF16LinearDynamicInputScale.from_linear(
            siglipmlp.down_proj, init_only=init_only
        )

    def forward(self, buffer: ActivationBuffer) -> torch.Tensor:
        # INT8 in, FP16 out
        torch.sum(buffer.quantized_input_buffer)
        torch.sum(buffer.quantized_scale_buffer)
        self.gateup_proj(
            buffer.quantized_input_buffer,
            buffer.quantized_scale_buffer,
            buffer.gateup_buffer,
        )
        torch.sum(buffer.gateup_buffer)################
        #FP16 in, INT8 out
        awq_inference_engine.gelu_and_mul_quant(buffer.quantized_mlp_act_buffer, buffer.gateup_buffer, buffer.quantized_scale_buffer, buffer.tmp)
        # INT8 in, FP16 out
        torch.sum(buffer.quantized_mlp_act_buffer)
        torch.sum(buffer.quantized_scale_buffer)
        torch.sum(buffer.in_out_fc2_act_buffer)
        self.down_proj(
            buffer.quantized_mlp_act_buffer,
            buffer.quantized_scale_buffer,
            buffer.in_out_fc2_act_buffer,
        )

class QuantMixtureEncoderLayer(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.embed_dim = module.self_attn.q_proj.in_features
        self.self_attn = QuantMixtureAttention2(module.self_attn)
        self.adaptive_mode = module.adaptive_mode
        if "layernorm" in str(module.input_layernorm.__class__).lower() and self.adaptive_mode is None: 
            self.input_layernorm = LayerNormGeneral(
                module.input_layernorm.weight.data,
                module.input_layernorm.bias.data,
                module.input_layernorm.eps,
                True,
            ).cuda()
            self.post_attention_layernorm = LayerNormGeneral(
                module.post_attention_layernorm.weight.data,
                module.post_attention_layernorm.bias.data,
                module.post_attention_layernorm.eps,
                True,
            ).cuda()
        elif "rmsnorm" in str(module.input_layernorm.__class__).lower() and self.adaptive_mode is None: 
            if "gemma" in str(module.input_layernorm.__class__).lower():
                self.input_layernorm = RMSNormGeneral(
                    module.input_layernorm.weight.data+1,
                    module.input_layernorm.eps,
                    True,
                ).cuda()
                self.post_attention_layernorm = RMSNormGeneral(
                    module.post_attention_layernorm.weight.data+1,
                    module.post_attention_layernorm.eps,
                    True,
                ).cuda()
            else:
                self.input_layernorm = RMSNormGeneral(
                    module.input_layernorm.weight.data,
                    module.input_layernorm.eps,
                    True,
                ).cuda()
                self.post_attention_layernorm = RMSNormGeneral(
                    module.post_attention_layernorm.weight.data,
                    module.post_attention_layernorm.eps,
                    True,
                ).cuda()
        else:
            raise NotImplementedError(type(module.input_layernorm))  
        self.mlp = QuantMixtureMLP(module.mlp)
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
        freqs,
        attention_mask,
        kv_cache_vlm,
        kv_cache_proprio,
        bsz,
        seqlen,
    ) -> Tuple[torch.FloatTensor]:
        # Attention block
        # FP16 in int8 out, layernorm & quantization
        residual = hidden_states
        self.input_layernorm(
            hidden_states.reshape(-1, self.embed_dim).contiguous(),
            buffer.quantized_input_buffer,
            buffer.quantized_scale_buffer,
        )
        # INT8 -> FP16
        self.self_attn(buffer, bsz, seqlen, freqs, attention_mask, kv_cache_vlm, kv_cache_proprio)
        hidden_states = (
            residual.reshape(-1, self.embed_dim) + buffer.in_out_fc2_act_buffer
        )
        # Fully Connected
        residual = hidden_states
        torch.sum(hidden_states)
        # FP16 in int8 out, layernorm & quantization
        self.post_attention_layernorm(
            hidden_states.reshape(-1, self.embed_dim),
            buffer.quantized_input_buffer,
            buffer.quantized_scale_buffer,
        )

        # INT8 -> FP16
        self.mlp(buffer)
        hidden_states = (
            residual.reshape(-1, self.embed_dim) + buffer.in_out_fc2_act_buffer
        ).contiguous()
        return hidden_states

class RMSNormGeneral(nn.Module):
    """RMS normalization (w/ per-token or per-tensor quant).

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
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
