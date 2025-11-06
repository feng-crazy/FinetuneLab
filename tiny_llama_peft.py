import torch

# 加载数据集
from datasets import load_dataset

dataset = load_dataset("HuggingFaceH4/no_robots", split="train[:100]")  # 只取100条用于测试

# 加载模型
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # 使用 float16 节省内存
    device_map="auto"           # 自动使用 mps（Apple GPU）
)

from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,                    # LoRA 秩
    lora_alpha=16,          # 缩放因子
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",  # 因为是语言模型
    target_modules=["q_proj", "v_proj"]  # TinyLlama 的注意力层模块名
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 查看可训练参数数量（应远小于总参数）

def format_chat(sample):
    messages = sample["messages"]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

dataset = dataset.map(format_chat)
dataset = dataset.train_test_split(test_size=0.1)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)


from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./peft-tinyllama",
    per_device_train_batch_size=1,      # Mac 内存小，batch_size=1
    gradient_accumulation_steps=4,      # 模拟更大 batch
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    fp16=True,                          # 启用 float16（MPS 支持）
    report_to="none",                   # 不连 wandb
    optim="adamw_torch",
    lr_scheduler_type="cosine",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

trainer.train()


