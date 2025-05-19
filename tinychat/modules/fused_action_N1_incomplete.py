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

class QuantDiT(nn.Module):
    def __init__(
        self,
        module,
        bsz=1,
        qlen=4,
        kvlen = 256,        
    ):
        super().__init__()

        self.attention_head_dim = module.attention_head_dim
        self.inner_dim = module.inner_dim
        # Timestep encoder
        self.timestep_encoder = module.timestep_encoder

        self.transformer_blocks = [QuantBasicTransformerBlock(layer) for layer in module.transformer_blocks]

        # Output blocks
        self.norm_out = module.norm_out
        self.proj_out_1 = module.norm_out.proj_out_1
        self.proj_out_2 = module.norm_out.proj_out_2
        self.bsz = bsz
        self.qlen = qlen
        self.kvlen = kvlen
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
        hidden_states = hidden_states.contiguous()
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
            encoder_hidden_states,
            self.buffer.cross_quantized_scale_buffer,
        )
''' 
        rewrite this part
        # Only replace forwarding of transformer blocks
        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1 and self.config.interleave_self_attention:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            all_hidden_states.append(hidden_states)
'''
        # Output processing
        conditioning = temb
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        if return_all_hidden_states:
            return self.proj_out_2(hidden_states), all_hidden_states
        else:
            return self.proj_out_2(hidden_states)

class QuantBasicTransformerBlock(nn.Module):
    def __init__(self,module):
        super().__init__()
        self.dim = module.dim
        self.num_attention_heads = module.num_attention_heads
        self.attention_head_dim = module.attention_head_dim
        self.dropout = module.dropout
        self.cross_attention_dim = module.cross_attention_dim
        self.activation_fn = module.activation_fn
        self.attention_bias = module.attention_bias
        self.norm_elementwise_affine = module.norm_elementwise_affine
        self.positional_embeddings = module.positional_embeddings
        self.num_positional_embeddings = module.num_positional_embeddings
        self.norm_type = module.norm_type
        assert module.pos_embed is None, "We do not support pos_embed yet"
        self.pos_embed = None
        self.buffer = ActivationBuffer(module, intermediate_size=self.intermediate_size, hidden_size=self.hidden_size, input_dim=self.input_dim, attn_group=self.attn_group)
        self.bsz = bsz
        self.seqlen = seqlen
        self.buffer.allocate_activation_buffer(self.bsz * self.seqlen)

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        if norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        ''' TODO: Write the attention block 
        (Note: no rope, no mask, no ak norm, no softcap, you may just use attention from siglip)
        The only thing special is k,v is derived from encoder_hidden_states and q is from hidden_states, therefore, not fuse qkv, just use three W8A8 linears
        '''
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=None,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        ''' TODO: Write the MLP block 
        same as siglip
        '''
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )
 '''
 TODO: rewrite this forward function
 The most difficult thing may be the buffer. You need to use the buffer with 'cross' in its name for encoder_hidden_states
 For the cross attention, we quant encoder_hidden_states just once in the QuantDiT fwd, since they are same for all layers, you may just quant it for one time in the QuantDiT fwd
 '''
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        # 0. Self-Attention
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, temb)
        else:
            norm_hidden_states = self.norm1(hidden_states)

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            # encoder_attention_mask=encoder_attention_mask,
        )
        if self.final_dropout:
            attn_output = self.final_dropout(attn_output)

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 4. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        return hidden_states
class QuantAdaLayerNorm(nn.Module):
    def __init__(
        self,
        norm
    ) -> None:
        super().__init__()     
        self.linear=W8A8OF16LinearDynamicInputScale.from_linear(norm.linear, init_only=False)
        self.norm_weight=torch.ones(in_features, dtype=torch.half).cuda().contiguous()
        self.norm_bias=torch.zeros(in_features, dtype=torch.half).cuda().contiguous()
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
            hidden_states,
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

        
        