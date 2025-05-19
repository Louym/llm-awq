import torch
from transformers import LlamaForCausalLM, AutoTokenizer
from tinychat.modules.fused_llama import QuantLlamaModel

# Load model and tokenizer
model_id = "andrijdavid/Llama3-1B-Base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = LlamaForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).cuda()

# # Create quantized model
model.model = QuantLlamaModel(model.model, bsz=1, seqlen=512)

# Create dummy input
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt").to("cuda")

# Create attention mask
attention_mask = torch.ones_like(inputs.input_ids)

# Create position ids
position_ids = torch.arange(len(inputs.input_ids[0])).unsqueeze(0).to("cuda")

# Convert input ids to embeddings
inputs_embeds = model.model.embed_tokens(inputs.input_ids)

# Run inference
with torch.no_grad():
    # Warmup
    for _ in range(3):
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids
        )

    # Benchmark
    import time
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(10):
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
        
    torch.cuda.synchronize()
    end = time.time()
    
    print(f"Average inference time: {(end-start)*100/10:.2f}ms")
