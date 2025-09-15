import logging
import os
import sys
from dotenv import load_dotenv

import torch
from datasets import load_dataset
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint

from trl import DPOTrainer, DPOConfig, TrlParser
from peft import LoraConfig
from dataclass import *

# envファイルから環境変数の読み込み
load_dotenv()

def main():
    parser = TrlParser((DPOConfig, ModelArgs, TokenizerArgs, DatasetArgs, PeftArgs, BnbArgs, DsArgs, MiscArgs))
    dpo_config, model_args, tok_args, data_args, peft_args, bnb_args, ds_args, misc = parser.parse_args_and_config()

    set_seed(42)
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=tok_args.use_fast,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = tok_args.padding_side
    tokenizer._pref_chat_template = tok_args.chat_template

    # Model
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(model_args.dtype, torch.bfloat16)
    kwargs = dict(
        trust_remote_code=model_args.trust_remote_code,
        dtype=dtype,
        #device_map="auto",
        use_cache=False if dpo_config.gradient_checkpointing else True,
        low_cpu_mem_usage=True,
        attn_implementation=model_args.attn_implementation
        )
    if bnb_args.bnb_enable:
        if bnb_args.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, bnb_args.bnb_4bit_compute_dtype),
                bnb_4bit_use_double_quant=bnb_args.bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=bnb_args.bnb_4bit_quant_type,
            )
            kwargs.update(quantization_config=bnb_config)
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
            kwargs.update(quantization_config=bnb_config)
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **kwargs)

    # Dataset
    # データセットのキーを変換
    def preprocess_function(example):
        return {
            "prompt": example["prompt"],
            "chosen": example["chosen"],
            "rejected": example["rejected"],
        }
    # split="train[:100]"でデータの数を変更可能
    train_data = load_dataset(data_args.train_data, split="train", streaming=data_args.streaming)
    if data_args.shuffle and not data_args.streaming:
        train_data = train_data.shuffle(seed=42)
        train_data = train_data.map(preprocess_function)
    if data_args.eval_data is not None:
        eval_data = load_dataset(data_args.eval_data, split="train", streaming=data_args.streaming)
        eval_data = eval_data.map(preprocess_function)
    else:
        eval_data = None

    # LoRA
    peft_config = None
    if peft_args.peft_enable:
        peft_config = LoraConfig(
            r=peft_args.lora_r,
            lora_alpha=peft_args.lora_alpha,
            lora_dropout=peft_args.lora_dropout,
            bias=peft_args.lora_bias,
            target_modules=peft_args.lora_target_modules,
            task_type=peft_args.lora_task_type
        )
    
    # DeepSpeed
    if ds_args.ds_enable and ds_args.ds_config_file:
        setattr(dpo_config, "deepspeed", ds_args.ds_config_file)
    
    # SFT Trainer
    trainer = DPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_data,
        eval_dataset=eval_data,
        peft_config=peft_config,
        args=dpo_config,
    )

    # Train
    trainer.train()

    # Save
    os.makedirs(dpo_config.output_dir, exist_ok=True)
    if misc.merge_lora_after_training and peft_config is not None:
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(dpo_config.output_dir, safe_serialization=True)
    else:
        trainer.model.save_pretrained(dpo_config.output_dir)
    tokenizer.save_pretrained(dpo_config.output_dir)

    if misc.push_to_hub:
        trainer.model.push_to_hub(dpo_config.output_dir)
        tokenizer.push_to_hub(dpo_config.output_dir)

if __name__ == "__main__":
    main()