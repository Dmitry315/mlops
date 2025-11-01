import argparse
from datasets import load_dataset
import pandas as pd

def read_fine_web(save_path="../experiments/data/fineweb2/train.jsonl"):
    dataset_name = "HuggingFaceFW/fineweb-2"
    ds = load_dataset(dataset_name, "ell_Grek", split="train[:10%]")
    ds.to_json(save_path, lines=True, orient="records", force_ascii=False)

def save_txt(read_path="../experiments/data/fineweb2/train.jsonl", save_path="../experiments/data/fineweb2/train.txt"):
    df = pd.read_json(read_path, orient="records", lines=True)
    df["text"] = df["text"] + "\n\n"
    with open(save_path, "w") as f:
        f.writelines(df["text"].to_list())
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Fineweb2 Greek")
    parser.add_argument("--save-path", type=str, default="../experiments/data/fineweb2/train.jsonl", help="Data save path")
    parser.add_argument("--to-txt",action="store_true",help="Convert to txt")
    parser.add_argument("--save-path-txt", type=str, default="../experiments/data/fineweb2/train.txt", help="Data save path")
    args = parser.parse_args()
    
    read_fine_web(save_path=args.save_path)

    if args.to_txt:
        save_txt(read_path=args.save_path, save_path=args.save_path_txt)