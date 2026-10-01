# LLM RAG Evaluation Engine 🚀

> **Comprehensive Two-Stage Retrieval Evaluation & Synthetic Dataset Generation Suite**
> Powered by **DeepEval**, **LangChain**, **ChromaDB**, **BAAI/bge-m3**, **Cross-Encoder Reranking**, and **Mistral AI**.

---

## 📋 Overview

This repository implements a production-grade **Retrieval-Augmented Generation (RAG) Evaluation Framework**. It is designed to evaluate, benchmark, and optimize document retrieval performance across video transcripts (VTT format) from multi-session AI/LLM courses.

The framework combines a **two-stage retrieval pipeline** (Dense Bi-Encoder over-retrieval + Cross-Encoder re-scoring) with automated **DeepEval Contextual Metrics** (Contextual Recall & Contextual Precision) judged by `open-mistral-7b`.

---

## 🏗 System Architecture

```mermaid
flowchart TD
    subgraph Ingestion ["1. Data Ingestion & Indexing"]
        VTT["📄 VTT Lecture Transcripts (Data/*.vtt)"] --> TextSplitter["RecursiveCharacterTextSplitter (size=1000, overlap=150)"]
        TextSplitter --> BGE["Embeddings: BAAI/bge-m3"]
        BGE --> ChromaDB[("Chroma Vector Store (chroma_store/)")]
    end

    subgraph TwoStageRetriever ["2. Two-Stage Retrieval Pipeline"]
        Query["❓ User / Golden Query"] --> BiEncoder["Bi-Encoder Search (fetch_k=23)"]
        ChromaDB --> BiEncoder
        BiEncoder --> Candidates["23 Candidate Chunks"]
        Candidates --> CrossEncoder["Cross-Encoder Reranker\n(ms-marco-MiniLM-L-6-v2)"]
        CrossEncoder --> TopK["Top-6 Reranked Chunks"]
    end

    subgraph DeepEvalHarness ["3. DeepEval Evaluation Suite"]
        TopK --> TestCases["LLMTestCase (Input, Context, Ideal Answer)"]
        Goldens["🎯 Golden Dataset (Goldens/retriever_goldens.json)"] --> TestCases
        TestCases --> Recall["Contextual Recall Metric (threshold=0.70)"]
        TestCases --> Precision["Contextual Precision Metric (threshold=0.70)"]
        Recall --> MistralJudge["Judge Model: MistralLLM (open-mistral-7b)"]
        Precision --> MistralJudge
        MistralJudge --> JSONRepair["clean_json_output (json_repair)"]
        JSONRepair --> Report["📊 Evaluation Report & Summary"]
    end
```

---

## 📊 Evaluation Performance Benchmarks

Evaluated across **15 human-curated reference golden test cases** covering lecture sessions 1 to 9:

| Retrieval Configuration | Contextual Precision | Contextual Recall | Pass Rate (Recall) | Pass Rate (Precision) |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline Vector Search (`k=5`)** | ~0.82 | ~0.61 | ~40.0% | ~80.0% |
| **Reranked (`fetch_k=20, top_k=6`)** | 0.9176 | 0.7311 | 46.7% (7/15) | 93.3% (14/15) |
| **Reranked (`fetch_k=25, top_k=7`)** | 0.8742 | 0.7507 | 60.0% (9/15) | 93.3% (14/15) |
| **Reranked (`fetch_k=25, top_k=6`, overlap=250)** | 0.8955 | 0.7577 | 73.3% (11/15) | 93.3% (14/15) |
| **Reranked Active Default (`fetch_k=23, top_k=6`, overlap=150)** 🏆 | **0.9047** | **0.7493** | **60.0% (9/15)** | **93.3% (14/15)** |

### Key Hyperparameter Sweep Insights:
- **Optimal Candidate Pool (`fetch_k=23`)**: Over-retrieving 23 candidates gives the Cross-Encoder the necessary depth to capture multi-sentence lecture concepts.
- **Chunking Strategy (`size=1000, overlap=150`)**: Fine-tuned chunking prevents text redundancy across context windows, yielding high contextual precision (**0.9047**, **93.33% pass rate**).
- **Top-K Cutoff Tuning (`top_k=6`)**: Keeping top 6 reranked chunks balances precision while filtering out noisy background chunks.
- **Benchmark Stability**: Maintains **93.33% Contextual Precision Pass Rate** (14/15 passed) and **60.0% Contextual Recall Pass Rate** (9/15 passed) with an average recall score of **0.7493**.

---

## ⭐ Key Technical Features & Optimizations

1. **Two-Stage Reranking Pipeline**:
   - **Stage 1 (Bi-Encoder Over-Retrieval)**: Uses `BAAI/bge-m3` to fast-retrieve `fetch_k = 23` candidate chunks from Chroma vector store.
   - **Stage 2 (Cross-Encoder Re-Scoring)**: Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to perform pairwise query-document re-scoring, filtering down to the top `top_k = 6` chunks.

2. **Apple Silicon (Mac M2) Hardware Safety**:
   - Caps PyTorch CPU thread usage (`torch.set_num_threads(4)`) and sets batch encoding size (`batch_size = 32`) to prevent CPU thread starvation and system freezing during heavy embedding tasks.

3. **Robust LLM Judge & JSON Repair System**:
   - Custom `MistralLLM` class extending DeepEval's `DeepEvalBaseLLM`.
   - Built-in `clean_json_output()` pipeline using `json_repair` and brace-stack tracking (`{ ... }`) to sanitize LLM responses containing unescaped newlines, control characters, or markdown codeblocks.
   - Graceful schema fallback: Raises `TypeError` on native schema calls to trigger DeepEval's automatic fallback Pydantic deserialization.

4. **Synthetic Golden Generation**:
   - Automated creation of synthetic test cases using DeepEval's `Synthesizer` with evolution strategies (`Goldens/generate_goldens.py`).

---

## 📁 Repository Structure

```
.
├── Data/                             # VTT lecture transcripts
│   ├── YT Sandbox LLM Evals Session 1.vtt
│   ├── ...
│   └── YT Sandbox LLM Evals Session 9.vtt
├── Evals/                            # Evaluation harness & execution
│   ├── eval_retriever.py            # Primary evaluation script (DeepEval metrics)
│   └── harness.py                   # Metric summarization & reporting utilities
├── Goldens/                          # Reference and synthetic golden datasets
│   ├── retriever_goldens.json       # 15 human-authored reference goldens
│   ├── retriever_deepeval_goldens.json # Generated synthetic golden draft dataset
│   └── generate_goldens.py          # DeepEval Synthesizer script
├── src/                              # Core RAG implementation
│   ├── retriever.py                 # Ingestion, VTT parser, Chroma vector store
│   └── reranker.py                  # Two-stage Cross-Encoder reranker
├── chroma_store/                     # Local persisted ChromaDB store
├── pyproject.toml                    # Dependencies (uv / pyproject configuration)
└── README.md                         # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python **3.14** (or 3.10+)
- [`uv`](https://github.com/astral-sh/uv) package manager
- Mistral AI API Key

### 2. Environment Setup

Clone the repository and install dependencies:
```bash
uv sync
```

Create a `.env` file in the project root:
```env
MISTRAL_API_KEY=your_mistral_api_key_here
```

### 3. Build Vector Store & Run Basic Retrieval

To ingest transcripts from `Data/*.vtt` and build/test the Chroma vector store:
```bash
uv run python src/retriever.py
```

To test the standalone Cross-Encoder reranker:
```bash
uv run python src/reranker.py
```

---

## 🧪 Running Evaluations

### Run Two-Stage Reranked Retriever Evaluation

Execute the full evaluation suite with `ContextualRecallMetric` and `ContextualPrecisionMetric`:
```bash
PYTHONPATH=. uv run python Evals/eval_retriever.py --rerank
```

### Run Baseline Retrieval (Single-Stage Vector Search)
```bash
PYTHONPATH=. uv run python Evals/eval_retriever.py
```

---

## 🧬 Generating Synthetic Golden Datasets

To synthesize new test cases from transcript context using DeepEval's `Synthesizer` and Mistral 7B:
```bash
"/Users/shlokbam/Documents/Code/LLM RAG Evaluation/.venv/bin/python" "Goldens/generate_goldens.py"
```
The resulting synthetic goldens are saved to `Goldens/retriever_deepeval_goldens.json`.

---

## 🛠 Tech Stack

- **Frameworks**: [DeepEval](https://github.com/confident-ai/deepeval), [LangChain](https://github.com/langchain-ai/langchain)
- **Vector DB**: [ChromaDB](https://www.trychroma.com/)
- **Embeddings**: [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3)
- **Reranker**: [`cross-encoder/ms-marco-MiniLM-L-6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)
- **LLM Judge**: `open-mistral-7b` via [`langchain_mistralai`](https://github.com/langchain-ai/langchain-mistralai)
- **JSON Sanitization**: [`json-repair`](https://github.com/maitool/json-repair)
