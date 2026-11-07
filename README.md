# NeuroText AI – Intelligent Predictive Writing & Language Intelligence Platform

[![CI/CD Pipeline](https://github.com/UMARMUKTHAR03/predictive-text-generator/actions/workflows/ci.yml/badge.svg)](https://github.com/UMARMUKTHAR03/predictive-text-generator/actions/workflows/ci.yml)
[![FastAPI Tests](https://img.shields.io/badge/pytest-13%20passed-brightgreen?logo=pytest)](https://github.com/UMARMUKTHAR03/predictive-text-generator)
[![Frontend Tests](https://img.shields.io/badge/vitest-2%20passed-brightgreen?logo=vitest)](https://github.com/UMARMUKTHAR03/predictive-text-generator)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.13-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.15%20CPU-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![React](https://img.shields.io/badge/React-18.2-61DAFB?logo=react)](https://react.dev/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**NeuroText AI** is a production-grade full-stack language intelligence platform that performs real-time next-token prediction and sequential headline generation using an authentic Bidirectional Long Short-Term Memory (BiLSTM) neural network.

- **Live Application:** [https://umarmukthar03.github.io/predictive-text-generator/](https://umarmukthar03.github.io/predictive-text-generator/)
- **Backend API Docs (Swagger UI):** `https://neurotext-ai-api.onrender.com/docs`
- **Original Notebook & Research Preserved:** [`Final.ipynb`](Final.ipynb), [`RNN and LSTM.ipynb`](RNN%20and%20LSTM.ipynb), [`legacy_archive/`](legacy_archive/)

---

## Architecture Overview

```
                      +---------------------------------------+
                      |       React 18 + TypeScript UI        |
                      |  (Vite + Tailwind CSS + Lucide Icons) |
                      +-------------------+-------------------+
                                          | HTTP / JSON
                                          v
                      +---------------------------------------+
                      |         FastAPI Inference Engine      |
                      |   - Lifespan Model & Vocab Loader     |
                      |   - End-to-End Latency Tracker        |
                      |   - Top-K & Temperature Decoding      |
                      +-------------------+-------------------+
                                          |
                        +-----------------+-----------------+
                        |                                   |
                        v                                   v
          +---------------------------+       +---------------------------+
          |      Text Tokenizer       |       |       PyTorch BiLSTM      |
          |  (7,299 Vocab JSON Map)   |       |  Embedding (100d)         |
          |  Pre-padding maxlen=39    |       |  BiLSTM (150 hidden units)|
          +---------------------------+       |  Linear Head (7,299 logits)|
                                              +---------------------------+
```

---

## Authentic Model Performance & Evaluation

The language model was trained on 6,508 technical publications from the Medium articles dataset (Dorian Lazar collection). The titles were partitioned **before** sequence creation (80% Train, 10% Validation, 10% Holdout Test) to completely eliminate train-test sequence overlap and data leakage.

### Holdout Test Set Benchmark (4,862 Unseen Sequences)

| Evaluation Metric | Random Guessing | Unigram Frequency Baseline | **NeuroText AI BiLSTM (Current)** | Historical Notebook (Train Only) |
| :--- | :---: | :---: | :---: | :---: |
| **Top-1 Accuracy** | 0.014% | 3.56% | **15.18%** | 54.60% (No split, evaluated on train) |
| **Top-3 Accuracy** | 0.041% | 8.54% | **22.79%** | — |
| **Top-5 Accuracy** | 0.068% | 12.92% | **27.44%** | — |
| **Top-10 Accuracy** | 0.137% | 21.43% | **34.45%** | — |
| **Cross-Entropy Loss** | 8.89 | 7.82 | **6.38** | 2.01 |
| **Perplexity** | 7,299 | 2,489 | **587.82** | 7.45 |
| **Inference Latency** | — | — | **0.331 ms / token** | — |
| **End-to-End API Latency**| — | — | **1.836 ms (mean)** | — |

> [!NOTE]
> **Performance Gap & Generalization Disclosure:**
> In the historical student notebook `Final.ipynb`, an accuracy of 54.60% was reported after 50 epochs because the model was evaluated directly on the training set without a holdout validation or test split. When evaluated on our leak-free holdout test partition of unseen titles, the model achieves **15.18% Top-1** and **27.44% Top-5** accuracy. This vastly outperforms random guessing (0.014%) and unigram frequency baselines (3.56%), demonstrating genuine semantic and syntactic next-token prediction across technical headlines.

---

## Key Features

1. **Predictive Writing Studio:**
   - Real-time debounced next-token suggestions as you type.
   - One-touch <kbd>Tab</kbd> completion for top candidate word.
   - Multi-word continuation with configurable decoding strategies (`greedy`, `top_k`, `sampling`).
   - Softmax temperature control (0.1 deterministic to 1.8 creative).

2. **Model Intelligence:**
   - Detailed neural pipeline breakdown: 100d Embedding, 150-unit BiLSTM (300d concatenated representation), and 7,299-dim Linear head.
   - Comprehensive parameter audit: 3,229,299 total trainable parameters.
   - Leak-free title-level data partitioning methodology disclosure.

3. **Live Operational Analytics:**
   - Real-time prediction and generation counters.
   - Separate measurement of model-only forward pass latency (1.44 ms) versus full HTTP end-to-end service latency (1.84 ms).
   - Comparative benchmark tables against unigram frequency baselines.

4. **Production FastAPI Backend:**
   - Clean lifespan architecture with instant model loading and zero per-request initialization overhead.
   - Full input sanitization, error boundaries, CORS handling, and Pydantic v2 validation.
   - 13 automated backend tests (PyTest) covering root, health, prediction, generation, and integration.

---

## Local Development & Setup

### Prerequisites
- Python 3.11+
- Node.js 18+ and npm

### 1. Backend Setup
```bash
# Clone the repository
git clone https://github.com/UMARMUKTHAR03/predictive-text-generator.git
cd predictive-text-generator

# Install backend dependencies
pip install -r backend/requirements.txt

# Run backend tests (13/13 passing)
python -m pytest backend/tests -v

# Start FastAPI server
python -m uvicorn backend.app.main:app --reload --port 8000
```
API will be live at `http://127.0.0.1:8000` with interactive Swagger docs at `http://127.0.0.1:8000/docs`.

### 2. Frontend Setup
```bash
cd frontend

# Install frontend dependencies
npm install

# Run frontend vitest suite
npm test

# Start development server
npm run dev
```

---

## Verification & Testing Summary

- **Backend PyTest Suite:** 13/13 passed (API endpoints, schemas, model loading, vocab integrity, forward passes).
- **Frontend Vitest Suite:** 2/2 passed (Navbar rendering, Dashboard benchmark metric rendering).
- **Frontend Production Build:** Built with Vite and TypeScript without errors (`dist/`).
- **Original Artifacts Preserved:** [`Final.ipynb`](Final.ipynb), [`RNN and LSTM.ipynb`](RNN%20and%20LSTM.ipynb), [`RNN and LSTM.pdf`](RNN%20and%20LSTM.pdf), and original static assets archived in [`legacy_archive/`](legacy_archive/).

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
