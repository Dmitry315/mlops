from nip import nip

from trl import SFTTrainer, SFTConfig
from transformers import PreTrainedTokenizerFast, Qwen2ForCausalLM, Qwen2Config

@nip
class HFQwenTrainer:
    def __init__(self, 
                 qwen_params, 
                 tokenizer_path, 
                 trainer_config_params, 
                 train_dataset, 
                 save_path, 
                 resume_from_checkpoint=False,
                 add_size_to_name=True,
                 tokenizer_truncation_side="right",
                 tokenizer_padding_side="left",
                 *args, **kwargs
                 ):
        self.train_dataset = train_dataset
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        self.tokenizer.padding_side = tokenizer_padding_side
        self.tokenizer.truncation_side = tokenizer_truncation_side

        print(f"pad_token: {self.tokenizer.pad_token}")
        print(f"pad_token_id: {self.tokenizer.pad_token_id}")
        print(f"eos_token: {self.tokenizer.eos_token}")
        print(f"eos_token_id: {self.tokenizer.eos_token_id}")

        self.qwen_config = Qwen2Config(
            vocab_size= self.tokenizer.vocab_size,
            **qwen_params
        )
        self.model = Qwen2ForCausalLM(config=self.qwen_config)
        self.train_config = SFTConfig(**trainer_config_params)
        print(self.tokenizer)
        print(self.train_dataset)
        self.resume_from_checkpoint = resume_from_checkpoint
        self.save_path = save_path
        total_params = sum(p.numel() for p in self.model.parameters())
        end = ""
        if total_params >= 100_000_000:
            total_params /= 1_000_000_000
            end = "B"
        elif total_params >= 100_000:
            total_params //= 1_000_000
            end = "M"
        print("==================================")
        print(f"Total parameters: {total_params:.1f}"+end)
        print("==================================")
        if add_size_to_name:
            self.save_path += f"{total_params:.1f}" + end

        self.save_path = self.save_path.replace(" ", "_")

        self.trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=self.train_dataset,
            # peft_config=peft_config,
            args=self.train_config
        )

    def train(self):
        self.trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)

    def save(self):
        self.model.save_pretrained(self.save_path)
        self.tokenizer.save_pretrained(self.save_path)