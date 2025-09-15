# 自作のdataclass
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ModelArgs:
    model_name_or_path: str
    dtype: str = "bfloat16"
    trust_remote_code: bool = False
    attn_implementation: str = "eager"

@dataclass
class TokenizerArgs:
    use_fast: bool = True
    padding_side: str = "right"
    chat_template: str = "auto"

@dataclass
class DatasetArgs:
    train_data: str = None
    eval_data: Optional[str] = None
    text_field: str = "instruction"
    num_proc: int = 4
    streaming: bool = False
    shuffle: bool = True

@dataclass
class PeftArgs:
    peft_enable: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    lora_target_modules: Optional[List[str]] = field(default_factory=lambda: None)
    lora_task_type: str = "CAUSAL_LM"

@dataclass
class BnbArgs:
    bnb_enable: bool = True
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    use_bnb_nested_quant: bool = True

@dataclass
class DsArgs:
    ds_enable: bool = False
    ds_config_file: Optional[str] = None

@dataclass
class MiscArgs:
    #push_to_hub: bool = False
    merge_lora_after_training: bool = False