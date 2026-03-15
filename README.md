# Hindi Tokenizer

Two tokenizers trained on Hindi Wikipedia for use in Hindi and Hindi-English (Hinglish) language models.



- [ag_hindi_bpe_tokenizer_32k](https://huggingface.co/adityaghai07/ag_hindi_bpe_tokenizer_32k)   BPE with Metaspace, HuggingFace-compatible
- [ag_hindi_uni_tokenizer_32k](https://huggingface.co/adityaghai07/ag_hindi_uni_tokenizer_32k)   Unigram via SentencePiece

---

## Benchmarks

<img width="2050" height="1012" alt="tokenizer_comparison" src="https://github.com/user-attachments/assets/1164c132-781d-499a-a838-1ffa68394f0a" />

___

[NOTE]: There is nothing architecturally novel here. This outcome was expected since the tokenizer was trained only on the Devanagari script. The takeaway is that you do not always need a generalized solution small, domain-specific approaches can sometimes outperform state-of-the-art systems.

***




## Project Structure

```
├── src/
│   ├── pretokenize.py      # Hindi-aware pre-tokenization regex
│   ├── preprocess.py       # Corpus cleaning pipeline
│   ├── train_bpe.py        # BPE training (HuggingFace tokenizers)
│   ├── train_unigram.py    # Unigram training (SentencePiece)
│   ├── evaluate.py         # Fertility + qualitative benchmarks
│   └── data_download.py    # Download Hindi Wikipedia / OSCAR
├── tests/
│   ├── test_pretokenize.py
│   ├── test_preprocess.py
│   └── test_tokenizer_integration.py
├── data/
│   ├── raw/
│   └── clean/
├── models/
│   ├── ag_hindi_bpe_tokenizer_32k/   # Trained BPE tokenizer.json
│   └── ag_hindi_uni_tokenizer_32k/   # Trained SentencePiece model
└── requirements.txt
```

## Models

### BPE (`ag_hindi_bpe_tokenizer_32k`)

Character-level BPE trained with the HuggingFace `tokenizers` library. Uses Metaspace pre-tokenization, which attaches the space marker `▁` to the beginning of each word rather than emitting spaces as separate tokens. This reduces token count by roughly 30-40% compared to treating spaces independently.

Fertility on Hindi text: ~1.5-1.8 tokens/word vs ~6-8 for GPT-2.

```python
from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast.from_pretrained("adityaghai07/ag_hindi_bpe_tokenizer_32k")
ids = tokenizer.encode("भारत एक महान देश है।")
print(tokenizer.decode(ids))
```

Special tokens: `<|pad|>` `<|bos|>` `<|eos|>` `<|unk|>` `<|sep|>` `<|mask|>` `<|user|>` `<|assistant|>`

### Unigram (`ag_hindi_uni_tokenizer_32k`)

Unigram language model trained with SentencePiece. Unlike BPE which greedily merges the most frequent pairs, Unigram starts from a large vocabulary and prunes it by removing tokens that least affect the overall likelihood of the corpus. This produces a probabilistic model where multiple segmentations are possible   useful for subword regularization during training.

Fertility on Hindi text: ~1.4-2.0 tokens/word.

```python
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.Load("models/ag_hindi_uni_tokenizer_32k/hindi_unigram.model")
print(sp.EncodeAsPieces("भारत एक महान देश है।"))
```

From HuggingFace Hub:

```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="adityaghai07/ag_hindi_uni_tokenizer_32k",
    filename="hindi_unigram.model"
)
sp.Load(model_path)
```

## Design Decisions

### Why 32K vocabulary

Hindi has a rich morphology   verbs conjugate heavily and nouns take postpositional suffixes, so a purely word-level vocabulary would be enormous and would still fail on unseen inflections. Character-level is the other extreme and produces very long sequences. 32K sits in a practical middle ground:

- Large enough to represent common Hindi words as single tokens (भारत, सरकार, etc.)
- Small enough that the embedding table stays manageable for a mid-size LM
- Leaves room for ~8 special tokens and a reasonable English subword vocabulary for code-mixed text
- Matches the scale used by multilingual models (mBERT uses 30K, XLM-R uses 250K but covers 100 languages)

### How words are broken

The tokenizer operates at the character level, not the byte level. This is a deliberate choice   byte-level BPE (used by GPT-2, LLaMA) encodes each UTF-8 byte as an initial token, so a single Devanagari character (which is 3 bytes in UTF-8) starts as 3 tokens before any merges happen. That creates a heavy tokenization tax on Devanagari text.

Instead, the initial alphabet here is the set of Unicode characters that appear in the corpus. Each Devanagari character   consonant, vowel, matra, anusvara, virama   is a single starting unit. BPE merges then build up from there.

Word boundary handling differs between the two models:

- BPE uses Metaspace: spaces are replaced with `▁` and prepended to the following word. `"भारत है"` becomes `["▁भारत", "▁है"]`. No separate space token is ever emitted.
- Unigram uses SentencePiece's `add_dummy_prefix=True`, which adds a leading space before the first word so that tokenization is position-independent (a word at the start of a sentence gets the same tokens as the same word mid-sentence).

Devanagari-specific rules enforced during pre-tokenization:

- Virama sequences (halant + following consonant) are never split, keeping conjuncts like क्ष, त्र, ज्ञ intact
- Combining marks (matras ा ि ी ु ू, anusvara ं, chandrabindu ँ, nukta ़) always stay attached to their base character
- Danda (।) and double danda (॥) are treated as punctuation and split off as separate tokens

## Performance

| Metric | BPE | Unigram | GPT-2 (baseline) |
|---|---|---|---|
| Fertility (tokens/word) | ~1.5-1.8 | ~1.8-2.0 | ~6-8 |
| Compression (chars/token) | ~3.5 | ~3.2 | ~1.2 |

Lower fertility and higher compression are better.

## Usage

### 1. Download corpus

```bash
python src/data_download.py --source wikipedia --output-dir data/raw
```

### 2. Clean corpus

```bash
python src/preprocess.py data/raw/hindi_wikipedia.txt data/clean/hindi_clean.txt
```

### 3. Train BPE

```bash
python src/train_bpe.py \
  --corpus data/clean/hindi_clean.txt \
  --output models/bpe_32k \
  --vocab-size 32000 \
  --hf-output models/ag_hindi_bpe_tokenizer_32k
```

### 4. Train Unigram

```bash
python src/train_unigram.py \
  --corpus data/clean/hindi_clean.txt \
  --output models/ag_hindi_uni_tokenizer_32k \
  --vocab-size 32000
```

### 5. Evaluate

```bash
python src/evaluate.py --tokenizer models/ag_hindi_bpe_tokenizer_32k --compare
```

## Running Tests

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

## License

MIT
