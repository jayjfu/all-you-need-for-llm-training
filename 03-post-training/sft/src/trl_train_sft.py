import os

from transformers import AutoProcessor, AutoModelForImageTextToText
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from peft import LoraConfig

BASE_MODEL = "Qwen/Qwen3.5-0.8B-Base"
INSTRUCT_MODEL = "Qwen/Qwen3.5-0.8B"
DATASET = "trl-lib/Capybara"
OUTPUT_ROOT = "../ckpts"
EPOCHS = 1

os.makedirs(OUTPUT_ROOT, exist_ok=True)

dataset = load_dataset(DATASET, split="train")

model = AutoModelForImageTextToText.from_pretrained(BASE_MODEL)
processor = AutoProcessor.from_pretrained(BASE_MODEL)
instruct_processor = AutoProcessor.from_pretrained(INSTRUCT_MODEL)
processor.chat_template = instruct_processor.chat_template

def train_variant(name, lr, batch, accum, packing, peft, lora_r=None, lora_alpha=None):
    out_dir = os.path.join(OUTPUT_ROOT, name)

    model_copy = AutoModelForImageTextToText.from_pretrained(BASE_MODEL)

    peft_config = None
    if peft:
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )

    training_args = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=EPOCHS,
        learning_rate=lr,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=accum,
        bf16=True,
        gradient_checkpointing=True,
        packing=packing,
    )

    trainer = SFTTrainer(
        model=model_copy,
        processing_class=processor,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(os.path.join(out_dir, "final"))

# Full fine-tune
train_variant("sft_lr2e5_bs32_acc1_pack",   2e-5, 32, 1, True,  False)
train_variant("sft_lr1e5_bs32_acc1_pack",   1e-5, 32, 1, True,  False)
train_variant("sft_lr5e5_bs32_acc1_pack",   5e-5, 32, 1, True,  False)
train_variant("sft_lr2e5_bs32_acc2_pack",   2e-5, 32, 2, True,  False)
train_variant("sft_lr5e5_bs32_acc1_nopack", 5e-5, 32, 1, False, False)

# LoRA
train_variant("sft_lora_r16_lr2e4_pack",  2e-4, 32, 1, True,  True,  16, 32)
train_variant("sft_lora_r32_lr2e4_pack",  2e-4, 32, 1, True,  True,  32, 64)
train_variant("sft_lora_r32_lr1e4_pack",  1e-4, 32, 1, True,  True,  32, 64)
train_variant("sft_lora_r8_lr2e4_nopack", 2e-4, 32, 1, False, True,  8,  16)