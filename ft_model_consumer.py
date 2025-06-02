from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Path to your local model directory (unzipped)
model_dir = "../outputs/phi-bankingqa-out5/merged"  # Change this

# from safetensors.torch import load_file

# model_path = "../outputs/phi-bankingqa-out5/merged/model.safetensors"
# state_dict = load_file(model_path)
# print(f"Loaded {len(state_dict)} tensors.")


# Load tokenizer and model (safetensors auto-detected)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float16,  # or torch.float32 if on CPU
    device_map="auto"           # maps to GPU if available
)

# Input prompt
prompt = "I want to open an account. Can you please help me?"

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate response
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7
    )

# Decode and print
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
