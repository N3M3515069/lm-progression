# Bigram Language Model

First stage of the `lm-progression` series — a from-scratch bigram language model
built following Karpathy's "Let's build GPT" lecture.

## What this is

A minimal character-level language model that predicts the next character based
purely on the current character — no attention, no context beyond a single token.
Implemented as a single `nn.Embedding` layer that maps each token directly to logits
over the vocabulary.

## Why start here

Before attention and transformers, this establishes the core training loop,
tokenization, and generation mechanics that every later stage (toy GPT, GPT-2 124M)
builds on:
- Character-level tokenization (encode/decode)
- Train/val split and batching
- Loss estimation via `estimate_loss()`
- Autoregressive generation via multinomial sampling

## Architecture

- Vocab size: 81 (character-level, derived from dataset)
- Block size: 8
- Batch size: 32
- Single embedding table (vocab_size × vocab_size) — no hidden layers, no attention

## Training

- Optimizer: AdamW, lr = 1e-3
- Steps: 10,000
- Loss: cross-entropy over next-character prediction

## Results

- Starting train loss: 4.791 (near the random-baseline ceiling of ln(81) ≈ 4.394)
- Final train loss: 2.29
- Final val loss: ~2.49
- Loss dropped steadily over training and converged close to the theoretical ceiling
  for a bigram model on this text — since a bigram model has zero context beyond the
  immediately preceding character, this is close to as good as it can get.

## Sample generation

**Before training** (random weights):

Z:l 99[Y8.gycT&zlAXc5-6zEDkY8D5t;YsKlFvyZ!m8&iGCnSpF]pbLdOC2cxk3,z8F1yDs-]wn 9p7-1PYN"Wx3pTm !7PRvbyb)xc0EaXV*FWx7.dVqQMMH?4.h'OpBm:ogRF]MabE4bQnqRx

**After training:**

The itatr woruived bithed Fode pithed bo fod f m'toom!"Hiftousiz5Yo ggngnd.

f Zedito abutingouthe e. je toould ablet aigam
Ar

"
athinsteds akan hat!"Wxcemee granternge wste, aintaunshare wo sowhethi; w pl

Mgatio foulocat; Zere cen.

bomf f hid "I
BLOf eim tin ie, adine hesm clarewan FLard aty lld te t parinved
CIsir tshed wan wewey upped vere be chey te m g ck!"
" pe*
aney drorinaI Silise
"Soor thasthin t bo, ansoo, wsosehas ig quneres on t aner bedund crkld atot DEBut bshfNoferledm thes

Even though the output is far from coherent English — expected, since bigram has no
context beyond the immediately preceding character — the model has clearly learned
plausible character-level structure: word-length groupings, sensible space
placement, and sentence-like capitalization/punctuation it didn't have pre-training.

## Limitations (expected at this stage)

- No context window beyond the immediately preceding character
- Generated text lacks coherence — this is by design, not a bug, since bigram
  models can't model long-range dependencies
- Next stage (toy GPT) introduces self-attention to address this directly

## Next

→ `toy-gpt/` — introduces multi-head self-attention, positional embeddings,
and causal masking.
