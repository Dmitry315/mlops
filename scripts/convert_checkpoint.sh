python src/convert/fsdp_checkpoint_to_safetensors.py \
    --fsdp-checkpoint "src/experiments/fsdp_chkpt/qwen_medium_pretrain0.2B/pytorch_model_fsdp_0" \
    --output-model "src/experiments/models/qwen_medium_pretrain0.2B" \
    --tokenizer "src/experiments/bpe_greek_tokenizer_medium"