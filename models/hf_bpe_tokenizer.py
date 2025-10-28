from nip import nip

from transformers import PreTrainedTokenizerFast

from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, Digits
from tokenizers.normalizers import NFD, StripAccents

@nip
class HFBPETokenizerTrainer:
    def __init__(self, bpe_init_params, bpe_trainer_params, train_corpus_files, save_path, tokenizer_config):
        self.tokenizer = BPE(**bpe_init_params)
        self.trainer = BpeTrainer(**bpe_trainer_params)
        self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Digits(individual_digits=True)])
        self.tokenizer.normalizer = normalizers.Sequence([NFD(), StripAccents()])
        self.train_corpus_files = train_corpus_files
        self.save_path = save_path
        self.tokenizer_config = tokenizer_config

    def train(self):
        self.tokenizer.train(files=self.train_corpus_files, trainer=self.trainer)

    def save(self):
        fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=self.tokenizer)
        hf_path = self.save_path.get("hub", None)
        local_path = self.save_path.get("local_path", None)
        if local_path is not None:
            fast_tokenizer.save_pretrained(local_path)
        if hf_path is not None:
            fast_tokenizer.push_to_hub(hf_path)
