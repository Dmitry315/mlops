from nip import nip
import json

from datasets import Dataset

def load_text_jsonl_data(file_path):
    """Load data from a JSONL file"""
    texts = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line.strip())
                texts.append(data['text'])
    
    return Dataset.from_dict({
        'text': texts
    })

@nip
def get_pretrain_data(data_path, *args, **kwargs):
    train_dataset = load_text_jsonl_data(data_path)
    
    # Format datasets
    return train_dataset