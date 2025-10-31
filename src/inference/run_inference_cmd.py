import os
import torch
import argparse
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast


def load_safetensors_model_simple(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Llama-Guard for toxicity classification")
    parser.add_argument("--model-path", type=str, default="src/experiments/models/qwen_pretrain0.7B", help="Model to run locally")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=10)
    args = parser.parse_args()
    print(args)
    model_path = args.model_path
    model, tokenizer = load_safetensors_model_simple(model_path)
    model.eval()

    # if torch.cuda.is_available():
    #     device = "cuda"
    # elif 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    text = input("User:")
    
    with torch.no_grad():
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        ).to(device)
        print("==[TOKENIZED]==")
        outputs = model.generate(
            inputs.input_ids,
            max_length=128,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )
        print("==[GENERATED]==")
        generated_text = tokenizer.decode(
            outputs[0], 
            skip_special_tokens=True
        )
    
    print("Model: ", generated_text)