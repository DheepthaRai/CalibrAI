# CalibrAI

CalibrAI was developed and demonstrated on Dell GB10 hardware, using locally hosted large language models including Meta's LLaMA and DeepSeek.

The system is designed to evaluate how real, locally deployed LLMs behave under different safety configurations, rather than relying on abstract or API-only models.

At its core, CalibrAI generates test prompts, runs them against an LLM backend at five enforcement levels, and surfaces both model behavior and system metrics through a React + FastAPI interface.

## What CalibrAI Does

CalibrAI helps answer a simple but uncomfortable question:

> How safe is this model, really, and what happens when I turn the knobs?

Specifically, it allows you to:

- Generate industry-specific test queries across benign, adversarial, and edge-case categories
- Run the same query across five safety enforcement levels (L1 Very Strict → L5 Very Permissive)
- Compare responses for refusal behavior, consistency, and threshold-sensitive degradation
- Identify the crossover point where tighter enforcement starts increasing false positives
- Monitor CPU, RAM, and GPU usage during inference
- Demo AI safety concepts even when the backend is mocked or unstable

## Project Structure

```text
calibrai/
│
├── app.py                        # Legacy Streamlit prototype (preserved for reference)
├── run_experiment.py             # Standalone calibration wave runner
├── test_pipeline.py              # End-to-end pipeline tests
├── requirements.txt
├── Makefile                      # Dev startup commands
│
├── backend/                      # FastAPI backend (v2)
│   ├── main.py                   # App entry point, CORS, router registration
│   ├── routers/
│   │   ├── calibration.py        # Calibration wave endpoints
│   │   ├── evaluate.py           # Single query evaluation
│   │   ├── policy.py             # Policy input and threshold recommendation
│   │   ├── audit.py              # Governance audit log
│   │   └── inspector.py          # Live model inspector
│   ├── services/
│   │   ├── calibration_service.py
│   │   ├── llm_service.py        # LLM interface (real + mock fallback)
│   │   └── policy_service.py
│   └── db/
│       ├── database.py
│       └── models.py
│
├── frontend/                     # React + Vite frontend
│   └── src/
│       ├── pages/
│       │   ├── Dashboard.tsx
│       │   └── Onboarding.tsx
│       └── components/
│           ├── AuditLog.tsx
│           ├── FailureAnalysis.tsx
│           ├── LiveInspector.tsx
│           ├── ModelStatus.tsx
│           ├── TradeoffChart.tsx
│           └── Sidebar.tsx
│
└── src/
    └── models/
        ├── llm_backend.py        # LLM interface (real)
        └── mock_backend.py       # LLM interface (mocked)
```

## Backend (FastAPI)

`backend/main.py` is the v2 entry point and handles:

- FastAPI app initialization and database setup via lifespan context
- CORS configuration for local frontend development
- Router registration across five functional areas: policy, calibration, audit, inspector, evaluate

### Key Design Choices

- Tokenizer parallelism is explicitly disabled to avoid HuggingFace deadlocks
- A mock backend (`mock_backend.py`) ensures demos do not fail if models are offline
- Database is initialized on startup via `init_db()`, no manual migration step required

## LLM Interface

`llm_service.py` abstracts the underlying model so the rest of the system does not care whether it is talking to:

- A local LLaMA or DeepSeek model via Ollama
- A vLLM or CPU-based backend
- A mocked response generator for testing and demos

Core interface methods:

- `generate_test_query(industry, category)` — produces a domain-specific prompt for a given behavioral category
- `test_with_safety_level(query, level)` — evaluates a query under a specific enforcement posture (L1–L5)

This separation allows models to be swapped without touching the API or frontend.

## System Metrics & Observability

CalibrAI surfaces live system metrics alongside model outputs:

- CPU utilization
- RAM usage (GB)
- GPU availability, utilization, and memory
- GPU name (if present)

Performance and stability tradeoffs are visible instead of hypothetical.

## Hardware & Model Stack

CalibrAI was built and tested on **Dell Pro Max GB10** hardware, leveraging its local compute capabilities for running and evaluating large language models without cloud dependencies.

### Models Used

- **LLaMA 3.1 8B** (via Ollama) — prompt generation and instruction-following
- **DeepSeek R1 7B** (via Ollama) — safety response evaluation across L1–L5 enforcement levels

Both models run locally through a modular backend interface, allowing CalibrAI to:

- Swap models without changing the API or frontend
- Compare safety behavior across different architectures
- Surface real-world deployment constraints such as memory pressure and inference latency

## Running the App

### 1. Install dependencies

```bash
make install
```

Or separately:

```bash
pip install -r backend/requirements.txt
cd frontend && npm install
```

### 2. Run (backend + frontend together)

```bash
make dev
```

This starts the FastAPI backend on `http://localhost:8000` and the React frontend on `http://localhost:5173`.

### 3. Run individually

```bash
make backend    # FastAPI only
make frontend   # React only
```

### Legacy Streamlit prototype

The original Streamlit interface is preserved in `app.py` and can still be run:

```bash
pip install -r requirements.txt
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```
