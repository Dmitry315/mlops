import argparse
import torch.distributed._shard.checkpoint as dist_cp
from accelerate import load_checkpoint_and_dispatch
from transformers import Qwen2ForCausalLM, PreTrainedTokenizerFast, Qwen2Config

parser = argparse.ArgumentParser(description="Convert FSDP checkpoint to safetensors")
parser.add_argument("--fsdp-checkpoint", type=str)
parser.add_argument("--output-model", type=str)
parser.add_argument("--tokenizer", type=str)

args = parser.parse_args()

tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)
qwen_config = Qwen2Config(
    vocab_size=150000,
    hidden_size= 512,
    head_dim= 128,
    intermediate_size= 1536,
    num_hidden_layers= 14,
    max_window_layers= 14,
    num_attention_heads= 16,
    num_key_value_heads= 8,
    max_position_embeddings= 2048,
    attention_dropout= 0.0,
    hidden_act= "silu",
    attention_bias= False
)

model = Qwen2ForCausalLM(config=qwen_config)


state_dict = {
        "model": model.state_dict()
    }

dist_cp.load_state_dict(
    state_dict=state_dict,
    storage_reader= dist_cp.FileSystemReader(args.fsdp_checkpoint),
    no_dist=True,
)

model.save_pretrained(args.output_model)
tokenizer.save_pretrained(args.output_model)