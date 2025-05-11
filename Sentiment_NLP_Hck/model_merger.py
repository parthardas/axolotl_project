from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Base model (same as you used for fine-tuning)
base_model_name = "microsoft/phi-1_5"

# Path to your trained LoRA adapter
lora_model_path = "./outputs1/phi-sentiment-out"

# Where to save the merged full model
merged_output_dir = "./phi1_5-sentiment-merged"

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA adapter on top of base
model = PeftModel.from_pretrained(base_model, lora_model_path)

# Merge LoRA weights into base model
#model = model.merge_and_unload()
model = model.merge_and_unload()

# Save the merged model
model.save_pretrained(merged_output_dir, safe_serialization=True)

# Also save tokenizer (required)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.save_pretrained(merged_output_dir)
