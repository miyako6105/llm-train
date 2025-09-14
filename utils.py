import os
import sys
import yaml
import datetime
import argparse
from datasets import Dataset
from transformers import AutoTokenizer

# コマンドライン引数を処理する関数
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, type=str, help="config file")
    return parser.parse_args()

# yamlファイルを読み込む関数
def load_yaml(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        configs = yaml.safe_load(f)
    return configs
