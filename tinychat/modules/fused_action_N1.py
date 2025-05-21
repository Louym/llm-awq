import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from awq.quantize import W8A8OF16LinearDynamicInputScale
from tinychat.utils.input_metadata import ActivationBuffer
from transformers.modeling_outputs import BaseModelOutput
from typing import Optional, Tuple, Union
from flash_attn import flash_attn_func
import math
CLIP_RANGE = 5
import awq_inference_engine
from tinychat.modules.fused_siglipdecoder import LayerNormGeneral


class QuantQKVSplitFlashAttention2(nn.Module):
    def __init__(
        self,
        module,
        init_only=False,
        cross_attn=False,
    ):
        super().__init__()
        self.embed_dim = module.inner_dim
        self.attn_group=module.query_dim//module.inner_kv_dim
        self.num_heads = module.heads
        self.head_dim = self.embed_dim // self.num_heads

        self.q_proj = W8A8OF16LinearDynamicInputScale.from_linear(
            module.to_q, init_only=init_only
        )
        self.k_proj = W8A8OF16LinearDynamicInputScale.from_linear(
            module.to_k, init_only=init_only
        )
        self.v_proj = W8A8OF16LinearDynamicInputScale.from_linear(
            module.to_v, init_only=init_only
        )
        self.out_proj = W8A8OF16LinearDynamicInputScale.from_linear(
            module.to_out[0], init_only=init_only
        )
        self.invoke_quant = self.invoke_quant_wo
        if cross_attn:
            self.forward=self.forward_cross_attn
        else:
            self.forward=self.forward_self_attn

    def invoke_quant_wo(self, buffer, attn_output):
        awq_inference_engine.invoke_quant(
            buffer.quantized_hidden_states_buffer,
            attn_output,
            buffer.quantized_scale_buffer,
        )
        
    def forward_self_attn(
        self, buffer: ActivationBuffer,  bsz=1, qlen=17, kvlen=99
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # qkv
        self.q_proj(
            buffer.quantized_input_buffer,
            buffer.quantized_scale_buffer,
            buffer.q_proj_act_buffer,
        )
        self.k_proj(
            buffer.quantized_input_buffer,
            buffer.quantized_scale_buffer,
            buffer.k_proj_act_buffer,
        )
        self.v_proj(
            buffer.quantized_input_buffer,
            buffer.quantized_scale_buffer,
            buffer.v_proj_act_buffer,
        )
        q = buffer.q_proj_act_buffer.reshape(bsz, qlen, self.num_heads, self.head_dim)
        k = buffer.k_proj_act_buffer.reshape(bsz, qlen, self.num_heads, self.head_dim)
        v = buffer.v_proj_act_buffer.reshape(bsz, qlen, self.num_heads, self.head_dim)
        attn_output = flash_attn_func(q, k, v, softmax_scale=None, causal=False)
        attn_output = attn_output.reshape(bsz * qlen, -1)
        # FP16 -> int8
        self.invoke_quant(buffer, attn_output)
        # INT8 in, FP16 out
        self.out_proj(
            buffer.quantized_hidden_states_buffer,
            buffer.quantized_scale_buffer,
            buffer.in_out_fc2_act_buffer,
        )
    def forward_cross_attn(
        self, buffer: ActivationBuffer, bsz=1, qlen=17, kvlen=99
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # qkv
        self.q_proj(
            buffer.quantized_input_buffer,
            buffer.quantized_scale_buffer,
            buffer.q_proj_act_buffer,
        )
        self.k_proj(
            buffer.corss_quantized_act_buffer,
            buffer.cross_quantized_scale_buffer,
            buffer.cross_k_proj_act_buffer,
        )
        self.v_proj(
            buffer.corss_quantized_act_buffer,
            buffer.cross_quantized_scale_buffer,
            buffer.cross_v_proj_act_buffer,
        )
        q = buffer.q_proj_act_buffer.reshape(bsz, qlen, self.num_heads, self.head_dim)
        k = buffer.cross_k_proj_act_buffer.reshape(bsz, kvlen, self.num_heads, self.head_dim)
        v = buffer.cross_v_proj_act_buffer.reshape(bsz, kvlen, self.num_heads, self.head_dim)
        attn_output = flash_attn_func(q, k, v, softmax_scale=None, causal=False)
        attn_output = attn_output.reshape(bsz * qlen, -1)
        # FP16 -> int8
        self.invoke_quant(buffer, attn_output)
        # INT8 in, FP16 out
        self.out_proj(
            buffer.quantized_hidden_states_buffer,
            buffer.quantized_scale_buffer,
            buffer.in_out_fc2_act_buffer,
        )

class QuantMLP(nn.Module):
    def __init__(self, mlp, init_only=False):
        super().__init__()
        #Only for N1 currently
        self.fc1 = W8A8OF16LinearDynamicInputScale.from_linear(
            mlp.net[0].proj, init_only=init_only
        )
        self.fc2 = W8A8OF16LinearDynamicInputScale.from_linear(
            mlp.net[2], init_only=init_only
        )

    def forward(self, buffer: ActivationBuffer) -> torch.Tensor:
        # INT8 in, FP16 out
        self.fc1(
            buffer.quantized_input_buffer,
            buffer.quantized_scale_buffer,
            buffer.fc1_buffer,
        )
        # Act & quantization
        awq_inference_engine.gelu_and_quant(
            buffer.quantized_mlp_act_buffer,
            buffer.fc1_buffer,
            buffer.quantized_scale_buffer,
            buffer.tmp,
        )
        # INT8 in, FP16 out
        self.fc2(
            buffer.quantized_mlp_act_buffer,
            buffer.quantized_scale_buffer,
            buffer.in_out_fc2_act_buffer,
        )

class QuantDiT(nn.Module):
    def __init__(
        self,
        module,
        bsz=1,
        qlen = 17,
        kvlen = 99,        
    ):
        super().__init__()
        self.config = module.config
        self.attention_head_dim = module.attention_head_dim
        self.hidden_size = module.inner_dim
        # Timestep encoder
        self.timestep_encoder = module.timestep_encoder
        self.intermediate_size=module.transformer_blocks[0].ff.net[2].in_features
        self.transformer_blocks = [QuantBasicTransformerBlock(layer, index%2==0) for index, layer in enumerate(module.transformer_blocks)]

        # Output blocks
        self.norm_out = module.norm_out.cuda()
        self.proj_out_1 = module.proj_out_1.cuda()
        self.proj_out_2 = module.proj_out_2.cuda()
        self.bsz = bsz
        self.qlen = qlen
        self.kvlen = kvlen
        self.buffer = ActivationBuffer(module, intermediate_size=self.intermediate_size, hidden_size=self.hidden_size, input_dim=self.hidden_size, attn_group=1)
        self.buffer.allocate_activation_buffer(self.bsz * self.qlen, self.bsz * self.kvlen)

    def forward(
        self,
        hidden_states: torch.Tensor,  # Shape: (B, T, D)
        encoder_hidden_states: torch.Tensor,  # Shape: (B, S, D)
        timestep: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_all_hidden_states: bool = False,
    ):
        # Encode timesteps
        temb = self.timestep_encoder(timestep)

        # Process through transformer blocks - single pass through the blocks
        hidden_states = hidden_states.half().contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()

        all_hidden_states = [hidden_states]
        
        # buffer allocation:
        bsz, qlen, _ =hidden_states.shape
        bsz, kvlen, _ =encoder_hidden_states.shape
        if self.bsz != bsz or self.qlen != qlen or self.kvlen != kvlen:
            self.buffer.allocate_activation_buffer(bsz * qlen, bsz * kvlen)
            self.bsz = bsz
            self.qlen = qlen
            self.kvlen = kvlen
        # Quantize encoder_hidden_states
        awq_inference_engine.invoke_quant(
            self.buffer.corss_quantized_act_buffer,
            encoder_hidden_states.half(),
            self.buffer.cross_quantized_scale_buffer,
        )
        # Only replace forwarding of transformer blocks
        for idx, block in enumerate(self.transformer_blocks):
            hidden_states=block(hidden_states,
                temb,
                self.buffer,
                self.bsz,
                self.qlen,
                self.kvlen,
            )
            all_hidden_states.append(hidden_states.reshape(self.bsz, self.qlen, -1))

        # Output processing
        conditioning = temb
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states.reshape(self.bsz, self.qlen, -1)) * (1 + scale[:, None]) + shift[:, None]
        if return_all_hidden_states:
            return self.proj_out_2(hidden_states), all_hidden_states
        else:
            return self.proj_out_2(hidden_states)

class QuantBasicTransformerBlock(nn.Module):
    def __init__(self,module, cross_attn):
        super().__init__()
        self.embed_dim = module.dim
        self.num_attention_heads = module.num_attention_heads
        self.attention_head_dim = module.attention_head_dim
        self.dropout = module.dropout
        self.cross_attention_dim = module.cross_attention_dim #None
        self.activation_fn = module.activation_fn
        self.attention_bias = module.attention_bias
        self.norm_elementwise_affine = module.norm_elementwise_affine
        self.positional_embeddings = module.positional_embeddings
        self.num_positional_embeddings = module.num_positional_embeddings
        self.norm_type = module.norm_type
        assert module.pos_embed is None, "We do not support pos_embed yet"
        self.pos_embed = None


        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        if self.norm_type == "ada_norm":
            self.norm1=module.norm1.half().cuda()
            # self.norm1 = QuantAdaLayerNorm(module.norm1)
        else:
            raise NotImplementedError(f"Norm type {self.norm_type} not implemented")
        
        self.attn1 = QuantQKVSplitFlashAttention2(module.attn1, cross_attn=cross_attn)

        # 3. Feed-forward
        self.layer_norm3 = LayerNormGeneral(
            torch.ones(module.norm3.normalized_shape[0], dtype=torch.half).cuda().contiguous(),
            torch.zeros(module.norm3.normalized_shape[0], dtype=torch.half).cuda().contiguous(),
            module.norm3.eps,
            True,
        ).cuda()

        self.ff = QuantMLP(module.ff)
        
    def invoke_input_quant(self, buffer, input):
        awq_inference_engine.invoke_quant(
            buffer.quantized_input_buffer,
            input,
            buffer.quantized_scale_buffer,
        )
    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.LongTensor] = None,
        buffer=None,
        bsz=1,
        qlen=17,
        kvlen=99,
    ) -> torch.Tensor:
        residual = hidden_states
        # 0. Self-Attention
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, temb)
        else:
            norm_hidden_states = self.norm1(hidden_states)
        self.invoke_input_quant(buffer, norm_hidden_states)

        self.attn1(buffer, bsz, qlen, kvlen)       
        
        hidden_states = (
            residual.reshape(-1, self.embed_dim) + buffer.in_out_fc2_act_buffer
        )
        residual = hidden_states
        # FP16 in int8 out, layernorm & quantization
        self.layer_norm3(
            hidden_states.reshape(-1, self.embed_dim),
            buffer.quantized_input_buffer,
            buffer.quantized_scale_buffer,
        )
        # INT8 -> FP16
        self.ff(buffer)
        hidden_states = (
            residual.reshape(-1, self.embed_dim) + buffer.in_out_fc2_act_buffer
        )
        return hidden_states

class QuantAdaLayerNorm(nn.Module):
    def __init__(
        self,
        norm
    ) -> None:
        super().__init__()     
        self.linear=W8A8OF16LinearDynamicInputScale.from_linear(norm.linear, init_only=False)
        self.norm_weight=torch.ones(norm.normalized_shape[0], dtype=torch.half).cuda().contiguous()
        self.norm_bias=torch.zeros(norm.normalized_shape[0], dtype=torch.half).cuda().contiguous()
        self.norm_variance_epsilon=norm.eps
    def forward(
        self,
        buffer,
        input
    ) -> torch.Tensor:
        # quantized_sum_buffer is not used, only to keep the consistency of the interface
        # SiLU & Quant
        awq_inference_engine.silu_and_quant(
            buffer.quantized_hidden_states_buffer,
            input,
            buffer.quantized_scale_buffer,
            buffer.tmp_input,
        )
        # INT8 in, FP16 out
        self.fc2(
            buffer.quantized_hidden_states_buffer,
            buffer.quantized_scale_buffer,
            self.in_out_fc2_act_buffer,
        )
        awq_inference_engine.layer_norm_general(
            buffer.quantized_hidden_states_buffer,
            self.in_out_fc2_act_buffer,
            self.norm_weight,
            self.norm_bias,
            buffer.quantized_scale_buffer,
            self.norm_variance_epsilon,
            True,
        )
        # TODO Rewrite this class and kernels
'''
class AdaLayerNorm(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        chunk_dim: int = 0,
    ):
        super().__init__()
        self.chunk_dim = chunk_dim
        output_dim = embedding_dim * 2
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim // 2, norm_eps, norm_elementwise_affine)

    def forward(
        self,
        x: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        temb = self.linear(self.silu(temb))
        scale, shift = temb.chunk(2, dim=1)
        x = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return x
'''
        
        