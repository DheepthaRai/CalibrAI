# CalibrAI

CalibrAI is an AI chatbot safety calibration tool designed to test, stress, and visualize how large language models behave under different safety and risk settings.

It is built for fast experimentation, live demos, and hackathon environments, including the kind where systems restart at the worst possible moment and you still need to explain your architecture with a straight face.

At its core, CalibrAI generates test prompts, runs them against an LLM backend at varying safety levels, and surfaces both model behavior and system metrics through a Streamlit interface.

## What CalibrAI Does
CalibrAI helps answer a simple but uncomfortable question:

> How safe is this model, really, and what happens when I turn the knobs?

Specifically, it allows you to:
- Generate industry-specific or generic test queries
- Run the same query across multiple safety or moderation levels
- Compare responses for refusal behavior, consistency, and degradation
- Monitor CPU, RAM, and GPU usage during inference
- Demo AI safety concepts even when the backend is mocked or unstable

## Project Structure
Project Structure

## Project Structure

```text
calibrai/
│
├── app.py                # Streamlit UI and orchestration
├── requirements.txt
├── README.md
│
└── src/
    └── models/
        └── llm_backend.py  # LLM interface (real)
        └── llm_backend.py  # LLM interface (mocked)
```

## app.py (Streamlit UI)
`app.py` is the main entry point and handles:
- Streamlit layout and user interaction
- Safety-level selection and iteration
- System metric collection (CPU, RAM, GPU via psutil and GPUtil)
- Parallel execution of safety tests using ThreadPoolExecutor
- Graceful fallback to a mock backend if the real LLM backend is unavailable

### Key Design Choices
- Tokenizer parallelism is explicitly disabled to avoid common HuggingFace deadlocks
- Threaded execution keeps the UI responsive during multiple test runs
- A mock backend ensures demos do not completely fail if models disconnect

## llm_backend.py (Model Interface)
`llm_backend.py` abstracts the underlying LLM so the UI does not care whether it is talking to:
- A local LLaMA or DeepSeek model
- A vLLM or CPU-based backend
- A mocked response generator for testing and demos

### Expected Backend Methods
The backend exposes methods such as:
- `generate_test_query(industry)`
- `test_with_safety_level(query, safety_level)`
This separation allows models to be swapped without rewriting the UI.

## System Metrics & Observability
CalibrAI surfaces live system metrics alongside model outputs:
- CPU utilization
- RAM usage (GB)
- GPU availability, utilization, and memory
- GPU name (if present)

Performance and stability tradeoffs are visible instead of hypothetical.

## Running the App
### 1. Install dependencies
```bash
pip install -r requirements.txt
```
### 2. Run
```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```
