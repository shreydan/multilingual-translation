# Multilingual Machine Translation with Transformers

> This project was for learning purposes only. Hence, focused on getting decent results rather than an building alternative to existing multilingual models.

- Implemented a [7M parameter model](./model.py).
- Trained a BERT style tokenizer.
- Trained on [Opus100](https://huggingface.co/datasets/opus100) Dataset with `en-hi` & `en-te` subsets.
- Go through the entirety on [Kaggle](https://www.kaggle.com/code/shreydan/en-hi-te-translation).

```
ENGLISH ----> HINDI
          |
          --> TELUGU
```

## Working

- The model understands which language to translate to based on the preceding beginning-of-sentence `bos` token:
  - english sentences start with `<s-en>` token
  - hindi sentences start with `<s-hi>` token
  - telugu sentences start with `<s-te>` token
  - all sentences end with `</s>` token
- trained as a Sequence-to-Sequence transformer model with an encoder-decoder style architecture. Encoder handles english and decoder handles both hindi & telugu.


## Model Config
```py
config = {
    'dim': 128,
    'n_heads': 4,
    'attn_dropout': 0.1,
    'mlp_dropout': 0.1,
    'depth': 8,
    'vocab_size': 30000,
    'max_len': 128
 }
```

## Inference Results

```
python inference.py --text 'how are you?' -l hi -s
>>> आप कैसे हैं?

python inference.py --text 'please call me' -l hi   
>>> कृपया मुझे पुकारो

python inference.py --text 'what are you doing?' -l te -s -t 0.5
>>> మీరు ఏం చేస్తున్నారు?

python inference.py --text "what's wrong?" -l te -s
>>> ఏమి తప్పు?
```

> The results are kinda hilarious but atleast it works


```
I have refrained my feet from every evil way,
That I might keep thy word.
                                Psalm 119:101
```