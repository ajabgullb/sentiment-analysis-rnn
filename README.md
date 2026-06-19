# IMDB Movie Review Sentiment Analysis — RNN with Word2Vec Embeddings
 
A from-scratch sentiment classification pipeline trained on the Stanford Large Movie Review Dataset (IMDB), built to study how recurrent architectures handle long-range sequence dependencies — and where they break down.
 
This is the first stage of a planned comparison: **vanilla RNN → LSTM/GRU → Transformer fine-tuning**, using the same dataset and evaluation pipeline throughout, to make the progression of NLP architecture design tangible rather than theoretical.
 
---
 
## Overview
 
| | |
|---|---|
| **Task** | Binary sentiment classification (positive / negative) |
| **Dataset** | [Stanford Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) (25k train / 25k test, balanced) |
| **Embeddings** | Word2Vec (Skip-Gram, negative sampling), trained from scratch on the training corpus via Gensim |
| **Model** | Vanilla RNN (`nn.RNN`), single layer, 128 hidden units |
| **Framework** | PyTorch |
 
---
 
## Pipeline
 
### 1. Preprocessing
- HTML stripping, lowercasing, non-alphabetic character removal
- NLTK tokenization, stopword removal, WordNet lemmatization
- Word2Vec (Skip-Gram, `vector_size=128`, `window=5`, `negative=5`) trained on the cleaned training corpus
- Sequence length capped at `MAX_SEQ_LEN=309` (95th percentile of cleaned training sequence lengths)
- Post-padding, integer encoding via custom `word2idx` vocabulary (`<PAD>=0`, `<UNK>=1`)
### 2. Model
- `nn.Embedding` initialized with pretrained Word2Vec weights (fine-tuned during training)
- Single-layer `nn.RNN` (`tanh` nonlinearity, hidden_dim=128)
- `pack_padded_sequence` to skip computation over padding
- Dropout + linear output layer → sigmoid (via `BCEWithLogitsLoss`)
### 3. Training regime
- Adam optimizer with `weight_decay` for L2 regularization
- Gradient clipping (`max_norm=5.0`) to control exploding gradients
- Embedding layer frozen for the first several epochs, then unfrozen for fine-tuning
- Checkpointing on best validation loss
- Early stopping based on validation loss plateau
---
 
## Results
 
| Metric | Score |
|---|---|
| Test Accuracy | 85.26% |
| Precision | 0.868 |
| Recall | 0.832 |
| F1 Score | 0.849 |
| ROC-AUC | 0.916 |
 
```
              precision    recall  f1-score   support
neg               0.84      0.87      0.86     12500
pos               0.87      0.83      0.85     12500
```
 
---
 
## Known limitations
 
This model's failure cases are not random — they're a direct, observable consequence of the vanilla RNN's vanishing-gradient problem over long sequences (BPTT gradient decay ≈ ρᵗ, where ρ < 1 for this architecture).
 
Empirically, the model struggles specifically with:
- **Long-range sentiment reversal** — e.g. *"The first hour was boring but the ending completely redeemed it"* — where the deciding word sits far from the cue it overrides.
- **Negation handling** — e.g. *"Not bad at all, actually quite good"* — compounded by the fact that common negation words can be inadvertently dropped during stopword removal.
- **Sentiment expressed late in long reviews**, where early signal has decayed out of the hidden state by the final timestep.
These cases are deliberately preserved as a baseline for comparison against the LSTM/GRU implementation, which uses gated memory specifically to address this.
 
---
 
## Project structure
 
```
.
├── notebooks/
│   └── sentiment_rnn_imdb.ipynb     # full pipeline: preprocessing → training → evaluation
├── README.md
├── requirements.txt
└── .gitignore
```
 
---
 
## Setup & usage
 
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```
 
Download and extract the dataset:
```bash
wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xzf aclImdb_v1.tar.gz
```
 
Then open `notebooks/sentiment_rnn_imdb.ipynb` and run top to bottom.
 
> **Note:** the raw dataset (~80MB compressed, ~230MB extracted), trained Word2Vec model, and model checkpoints are excluded from version control via `.gitignore` — see below.
 
---
 
## Roadmap
 
- [x] Vanilla RNN baseline with Word2Vec embeddings
- [ ] LSTM implementation — same pipeline, gated memory, direct comparison
- [ ] GRU implementation — lighter-weight gating alternative
- [ ] Transformer fine-tuning baseline (DistilBERT) for a modern-architecture comparison point
- [ ] Gradient-flow visualization across architectures (empirical vanishing-gradient comparison)
---
 
## References
 
- Maas, A. L. et al. (2011). [Learning Word Vectors for Sentiment Analysis](https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf). ACL 2011.
- Mikolov, T. et al. (2013). [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781).
- Mikolov, T. et al. (2013). [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546).
- Pascanu, R., Mikolov, T., & Bengio, Y. (2013). [On the Difficulty of Training Recurrent Neural Networks](https://arxiv.org/abs/1211.5063).
- Hochreiter, S., & Schmidhuber, J. (1997). [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf).
---
 
## License
 
This project is released under the MIT License. The IMDB dataset is provided by Stanford under its own terms — see the [dataset page](https://ai.stanford.edu/~amaas/data/sentiment/) for citation requirements.

