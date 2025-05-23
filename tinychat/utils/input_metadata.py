# File authors: Haotian Tang, Shang Yang, Yujun Lin, Song Han
# @article{lin2024awq,
#   title={AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration},
#   author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and Chen, Wei-Ming and Wang, Wei-Chen and Xiao, Guangxuan and Dang, Xingyu and Gan, Chuang and Han, Song},
#   journal={Proceedings of Machine Learning and Systems},
#   volume={6},
#   pages={87--100},
#   year={2024}
# }

import torch


class ActivationBuffer:
    """
    Pre-allocated Buffer for activation in the siglip model.

    Args:
        model: The input model
        batched_seq_len: The batched sequence length. Sum of all the sequence lengths in the batch.
    """

    def __init__(self, model, intermediate_size=None, hidden_size=None, input_dim=None, attn_group=1, llama_like=False):
        self.model_class = model.__class__.__name__
        try:
            self.model_dtype = model.layers[0].self_attn.k_proj.weight.dtype
        except:
            self.model_dtype=torch.half
        self.device = "cuda"
        assert self.model_class in [
            "SiglipEncoder", "Mixture", "LlamaModel", "DiT"
        ], f"model_class: {self.model_class} is currently not supported."
        assert (
            self.model_dtype == torch.float16
        ), f"model_dtype is expected to be fp16. Current: {self.model_dtype}."
        if intermediate_size is None:
            self.intermediate_size = model.config.intermediate_size
            self.hidden_size = model.config.hidden_size
            self.input_dim=self.hidden_size
        else:
            self.intermediate_size = intermediate_size
            self.hidden_size = hidden_size
            self.input_dim=input_dim
        self.attn_group=attn_group
        self.llama_like=llama_like
    def allocate_activation_buffer(self, batched_seq_len, batched_seq_len_KV=None):
        if self.model_class == "SiglipEncoder":
            self.__allocate_activation_buffer_siglip(batched_seq_len)
        elif "mixture" in str(self.model_class).lower() or "llama" in str(self.model_class).lower():
            self.__allocate_activation_buffer_llama(batched_seq_len)
        elif "DiT" in str(self.model_class):
            if batched_seq_len_KV is None:
                batched_seq_len_KV=batched_seq_len
            self.__allocate_activation_buffer_N1(batched_seq_len, batched_seq_len_KV)
        else:
            raise NotImplementedError(
                f"model_class: {self.model_class} is currently not supported."
            )

    def __allocate_activation_buffer_siglip(self, batched_seq_len):
        # Allocate fp16 activation buffer.
        qkv_dim=int(self.hidden_size*(1+2/self.attn_group))
        self.act_buffer = torch.empty(
            (batched_seq_len * max(qkv_dim, self.intermediate_size)),
            device=self.device,
            dtype=torch.float16,
        )
        self.qkv_proj_act_buffer = self.act_buffer[
            : batched_seq_len * qkv_dim
        ].view(
            batched_seq_len, qkv_dim
        )  # qkv

        self.in_out_fc2_act_buffer = self.act_buffer[
            : batched_seq_len * self.input_dim
        ].view(
            batched_seq_len, self.input_dim
        )  # LN1, Wo_out, LN2, all_out

        self.fc1_buffer = self.act_buffer[
            : batched_seq_len * self.intermediate_size
        ].view(batched_seq_len, self.intermediate_size)

        # Allocate quantized activation buffer.
        self.quantized_act_buffer = torch.empty(
            (batched_seq_len * max(self.input_dim, self.hidden_size, self.intermediate_size)),
            device=self.device,
            dtype=torch.int8,
        )
        self.quantized_hidden_states_buffer = self.quantized_act_buffer[
            : batched_seq_len * self.hidden_size
        ].view(
            batched_seq_len, self.hidden_size
        )  # Wo_in,
        self.quantized_input_buffer = self.quantized_act_buffer[
            : batched_seq_len * self.input_dim
        ].view(
            batched_seq_len, self.input_dim
        )  # qkv_in, up_gate_in,
        self.quantized_mlp_act_buffer = self.quantized_act_buffer[
            : batched_seq_len * self.intermediate_size
        ].view(batched_seq_len, self.intermediate_size)

        # per token
        self.quantized_scale_buffer = torch.empty(
            (batched_seq_len), device=self.device, dtype=torch.float16
        )

        # For faster act-quant implementation
        self.tmp = torch.empty(
            (batched_seq_len * self.intermediate_size),
            device=self.device,
            dtype=torch.float16,
        )
    def __allocate_activation_buffer_llama(self, batched_seq_len):
        # Allocate fp16 activation buffer.
        qkv_dim=int(self.hidden_size*(1+2/self.attn_group))
        self.act_buffer = torch.empty(
            (batched_seq_len * max(qkv_dim, 2 * self.intermediate_size)),
            device=self.device,
            dtype=torch.float16,
        )
        self.qkv_proj_act_buffer = self.act_buffer[
            : batched_seq_len * qkv_dim
        ].view(
            batched_seq_len, qkv_dim
        )  # qkv

        self.in_out_fc2_act_buffer = self.act_buffer[
            : batched_seq_len * self.input_dim
        ].view(
            batched_seq_len, self.input_dim
        )  # LN1, Wo_out, LN2, all_out

        self.gateup_buffer = self.act_buffer[
            : batched_seq_len * 2*self.intermediate_size
        ].view(batched_seq_len, 2*self.intermediate_size)

        # Allocate quantized activation buffer.
        self.quantized_act_buffer = torch.empty(
            (batched_seq_len * max(self.input_dim, self.hidden_size, self.intermediate_size)),
            device=self.device,
            dtype=torch.int8,
        )
        self.quantized_hidden_states_buffer = self.quantized_act_buffer[
            : batched_seq_len * self.hidden_size
        ].view(
            batched_seq_len, self.hidden_size
        )  # Wo_in,
        self.quantized_input_buffer = self.quantized_act_buffer[
            : batched_seq_len * self.input_dim
        ].view(
            batched_seq_len, self.input_dim
        )  # qkv_in, up_gate_in,
        self.quantized_mlp_act_buffer = self.quantized_act_buffer[
            : batched_seq_len * self.intermediate_size
        ].view(batched_seq_len, self.intermediate_size)

        # per token
        self.quantized_scale_buffer = torch.empty(
            (batched_seq_len), device=self.device, dtype=torch.float16
        )

        # For faster act-quant implementation
        self.tmp = torch.empty(
            (batched_seq_len * self.intermediate_size),
            device=self.device,
            dtype=torch.float16,
        )
    def __allocate_activation_buffer_N1(self, batched_seq_len_Q, batched_seq_len_KV):
        # Allocate fp16 activation buffer.
        qkv_dim=int(self.hidden_size*(1+2/self.attn_group))
        self.act_buffer = torch.empty(
            (batched_seq_len_Q * max(qkv_dim, self.intermediate_size)),
            device=self.device,
            dtype=torch.float16,
        )
        # self-attn
        # q
        self.q_proj_act_buffer = self.act_buffer[
            : batched_seq_len_Q * self.hidden_size
        ].view(
            batched_seq_len_Q, self.hidden_size
        )  
        # k
        self.k_proj_act_buffer = self.act_buffer[
            : batched_seq_len_Q * self.hidden_size//self.attn_group
        ].view(
            batched_seq_len_Q, self.hidden_size//self.attn_group
        )  
        # v
        self.v_proj_act_buffer = self.act_buffer[
            : batched_seq_len_Q * self.hidden_size//self.attn_group
        ].view(
            batched_seq_len_Q, self.hidden_size//self.attn_group
        )

        self.in_out_fc2_act_buffer = self.act_buffer[
            : batched_seq_len_Q * self.input_dim
        ].view(
            batched_seq_len_Q, self.input_dim
        )  # LN1, Wo_out, LN2, all_out

        self.fc1_buffer = self.act_buffer[
            : batched_seq_len_Q * self.intermediate_size
        ].view(batched_seq_len_Q, self.intermediate_size)

        # cross-attn
        self.cross_act_buffer = torch.empty(
            (batched_seq_len_KV * self.hidden_size//self.attn_group *2),
            device=self.device,
            dtype=torch.float16,
        )
        # k
        self.cross_k_proj_act_buffer = self.cross_act_buffer[
            : batched_seq_len_KV * self.hidden_size//self.attn_group
        ].view(
            batched_seq_len_KV, self.hidden_size//self.attn_group
        )  
        # v
        self.cross_v_proj_act_buffer = self.cross_act_buffer[
            batched_seq_len_KV * self.hidden_size//self.attn_group: 
        ].view(
            batched_seq_len_KV, self.hidden_size//self.attn_group
        )
        
        # Allocate quantized activation buffer.
        self.quantized_act_buffer = torch.empty(
            (batched_seq_len_Q * max(self.input_dim, self.hidden_size, self.intermediate_size)),
            device=self.device,
            dtype=torch.int8,
        )
        self.quantized_hidden_states_buffer = self.quantized_act_buffer[
            : batched_seq_len_Q * self.hidden_size
        ].view(
            batched_seq_len_Q, self.hidden_size
        )  # Wo_in,
        self.quantized_input_buffer = self.quantized_act_buffer[
            : batched_seq_len_Q * self.input_dim
        ].view(
            batched_seq_len_Q, self.input_dim
        )  # q, k, v share this buffer for self-attn; up_gate_in,
        self.quantized_mlp_act_buffer = self.quantized_act_buffer[
            : batched_seq_len_Q * self.intermediate_size
        ].view(batched_seq_len_Q, self.intermediate_size)
        
        
        
        
        # For quantized input of cross attn
        self.corss_quantized_act_buffer = torch.empty(
            (batched_seq_len_KV * self.hidden_size),
            device=self.device,
            dtype=torch.int8,
        ).view(
            batched_seq_len_KV, self.input_dim
        )  # k,v share this buffer for cross attn


        # per token
        self.quantized_scale_buffer = torch.empty(
            (batched_seq_len_Q), device=self.device, dtype=torch.float16
        )
        
        self.cross_quantized_scale_buffer = torch.empty(
            (batched_seq_len_KV), device=self.device, dtype=torch.float16
        )

        # For faster act-quant implementation
        self.tmp = torch.empty(
            (batched_seq_len_Q * self.intermediate_size),
            device=self.device,
            dtype=torch.float16,
        )
        
        # For faster act-quant implementation
        self.tmp_input = torch.empty(
            (batched_seq_len_Q * self.hidden_size),
            device=self.device,
            dtype=torch.float16,
        )