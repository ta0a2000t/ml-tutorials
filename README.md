# 🧠 NLP & Transformers Learning Path (with PyTorch Projects)

This checklist walks you through foundational NLP concepts and papers — with hands-on PyTorch projects — designed to:

* [ ] Teach deep learning fundamentals.
* [ ] Avoid high-level wrappers (no HuggingFace).
* [ ] Include some focus on Arabic and speech data.
* [ ] Help you eventually build a full audio/language model from scratch.

---

## ✅ Step 0: NLP Basics & Tokenization

### 📄 [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/)

* [ ] Read core chapters on tokenization and preprocessing.
* [ ] Tokenize a small Arabic dataset (news titles, short stories).
* [ ] Implement:

  * [ ] Character-level tokenizer
  * [ ] Word-level tokenizer
  * [ ] Subword tokenizer using BPE
* [ ] Handle Arabic-specific text processing:

  * [ ] Diacritics removal
  * [ ] Alif/Hamza normalization
  * [ ] Ligatures (e.g. Lam-Alif)
* [ ] Build token ↔ ID mapping manually

🧩 **Tokenizer Note:** Prioritize understanding. Use `tokenizers` (not `transformers`) only if speed becomes a bottleneck.

---

## 🔁 Step 1: RNNs, LSTMs & Attention

### 📄 [LSTM (1997)](https://www.bioinf.jku.at/publications/older/2604.pdf)

* [ ] Implement from scratch:

  * [ ] Vanilla RNN
  * [ ] LSTM
* [ ] Wrap into `nn.Module`
* [ ] Train on next-character prediction in Arabic strings
* [ ] Manually manage hidden states
* [ ] Handle batching with `Dataset` and `DataLoader`

### 📄 [Bahdanau Attention (2014)](https://arxiv.org/abs/1409.0473)

* [ ] Build RNN-based encoder-decoder model with attention
* [ ] Task: Reverse Arabic sentence (input: "مرحبا كيف حالك" → output: "كلاح فيك ابحرم")
* [ ] Implement:

  * [ ] Encoder & decoder with LSTMs
  * [ ] Additive attention
* [ ] Use `nn.Embedding` for token embeddings
* [ ] Try random vs. pre-trained embeddings (AraVec/FastText)
* [ ] Evaluate with token accuracy or character error rate

---

## ⚡️ Step 2: Transformers

### 📄 [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762)

* [ ] Implement a Transformer encoder from scratch:

  * [ ] Multi-head self-attention
  * [ ] Feed-forward network
  * [ ] LayerNorm and residuals
  * [ ] Positional encodings (sinusoidal or learned)
* [ ] Train on sentence classification (Arabic or English)
* [ ] Build a tokenizer (e.g., BPE) for input IDs
* [ ] Use `nn.Embedding` for token embeddings

🧩 **Embedding Note:** Understand how token embeddings, position embeddings, and attention layers interact.

---

## 🧠 Step 3: Pretraining & Fine-tuning

### 📄 [BERT (2018)](https://arxiv.org/abs/1810.04805)

* [ ] Understand MLM + NSP objectives
* [ ] Use your Transformer encoder from Step 2
* [ ] Implement:

  * [ ] MLM masking logic
  * [ ] NSP classifier head
  * [ ] Segment + position + token embeddings
* [ ] Train on Arabic Wikipedia or similar dataset
* [ ] Write custom `Dataset` for MLM/NSP pairs

### 📄 [GPT-1 (2018)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

* [ ] Implement decoder-only Transformer with causal masking
* [ ] Train on Arabic quotes, jokes, or poetry
* [ ] Build:

  * [ ] GPT-style tokenizer (char or BPE)
  * [ ] Causal mask in attention
  * [ ] Sampling method (greedy, top-k, nucleus)

🧩 **Tokenizer Note:** Handle RTL ordering and proper sampling.

### 📄 [T5 (2020)](https://arxiv.org/abs/1910.10683)

* [ ] Frame tasks as text-to-text (e.g., translation)
* [ ] Combine encoder and decoder with cross-attention
* [ ] Build tokenizer for both Arabic and English
* [ ] Implement translation task: "translate Arabic to English: مرحبا كيف حالك" → "Hello how are you"

---

## 🔊 Step 4: Speech + Multimodal

### 📄 [SpeechT5 (2021)](https://arxiv.org/abs/2110.07205)

* [ ] Extract MFCC features using `torchaudio`
* [ ] Train Transformer encoder on speech (MFCC input)
* [ ] Predict phonemes or characters (ASR)
* [ ] Use CTC loss or simple seq2seq objective
* [ ] Toy TTS: map text tokens to simple sinewave segments and stitch together for synthetic audio

🧩 **Speech Note:** Treat audio frames as input tokens. Learn how text ↔ audio mapping works in simplified TTS.

---

## 🧪 Optional Deepening (Pick Any)

* [ ] [A Primer in BERTology (2020)](https://arxiv.org/abs/2002.12327) – BERT internals & interpretability
* [ ] [Reformer (2020)](https://arxiv.org/abs/2001.04451) – Memory-efficient Transformers
* [ ] [State of Sparsity (2022)](https://arxiv.org/abs/2102.00554) – Sparse attention trends

---

## 📝 PyTorch Practice Tips

* [ ] Use `torch.nn.Module` over high-level abstractions
* [ ] Implement custom `Dataset` and `DataLoader`
* [ ] Manually define loss functions and optimizers
* [ ] Build core attention/math blocks from scratch
* [ ] Test tokenizers on Arabic-specific quirks
* [ ] Use GPU (Colab/Kaggle/local) for heavier models
* [ ] Start small → scale up only after success

---

## 🧭 Final Project Goals (Optional)

* [ ] 🔡 Mini Arabic GPT: generate poetry or short stories
* [ ] 🧏 Tiny ASR: recognize digits/words from your own recordings
* [ ] 🎤 Basic TTS: generate sinewave audio from characters or phonemes
