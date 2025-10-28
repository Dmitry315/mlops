import argparse
import os
import yaml
import numpy
import torch
import random
from nip import load, wrap_module
from dotenv import load_dotenv
from pprint import pprint

from models.hf_bpe_tokenizer import HFBPETokenizerTrainer

load_dotenv()

def train_from_config(config):
    tokenizer_trainer = config["trainer"]
    tokenizer_trainer.train()
    tokenizer.save_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Llama-Guard for toxicity classification")
    parser.add_argument("--config", type=str, default="configs/train.yaml",
                        help="Path to config file (default: configs/train.nip")
    
    args = parser.parse_args()

    

    for obj in load('config.nip'):
        print("================CONFIG================")
        pprint(obj)
        print("======================================")
        train_from_config(config=obj)