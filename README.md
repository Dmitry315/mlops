# ElladaGPT

Цель проекта - создать чат бот с нуля... на греческом языке.

# К чему стремимся

Точность на открытом бенчмарке (перевод): > 75%

Скорость отклика модели: < 1000мс

# Набор данных

## Pretrain

Для предобучения будут использоваться данные fineweb2 на греческом: 
- Современный (~70GB parquet, ~200MB utf-8 байтов, ~22B слов)[fineweb-2/viewer/ell_Grek](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/viewer/ell_Grek)

## SFT

Для SFT стадии будет использоваться:
- GPT-4 синта на греческом: [CausalLM/GPT-4-Self-Instruct-Greek](https://huggingface.co/datasets/CausalLM/GPT-4-Self-Instruct-Greek)
- instruct датсет (перевод почищенной альпаки от Стенфорда): [iamshnoo/alpaca-cleaned-greek](https://huggingface.co/datasets/iamshnoo/alpaca-cleaned-greek)

## Bench

Перевод mt-бенча на греческий: [ilsp/mt-bench-greek](https://huggingface.co/datasets/ilsp/mt-bench-greek)

# План технической реализации

Для этого проекта потребуется обучить токенизатор и модель, а также научиться бытро их инферить и считать качество на незнакомом языке.

Обучение BPE токенизатора происходит через библиотеку tokenizers. Для эффективного инференса лучше всего использовать tiktoken, так как он гораздо быстрее чем HF токенизаторы.

Планируется обучить 1-7b модель с использованием библиотек TRL (для SFT) и Accelerate (для FSDP).

Нужно разработать бенчмарк, где оценка считается через перевод запроса и оценки перевода ответа (LLM as a judge).

Быстрый инференс может осуществлятся через библиотеку vllm.