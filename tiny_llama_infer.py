from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
# 加载 LoRA 适配器
model = PeftModel.from_pretrained(base_model, "./peft-tinyllama-final")
model.eval()

# 构造输入
messages = [{"role": "user", "content": "你好！"}]
input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt").to("mps")

# 生成
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))