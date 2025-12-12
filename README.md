CalibrAI

CalibrAI is an AI chatbot safety calibration tool designed to test, stress, and visualize how large language models behave under different safety and risk settings. It was built for fast experimentation, live demos, and hackathon-grade chaos, where systems crash five minutes before judging and still need to tell a coherent story.

At its core, CalibrAI generates test prompts, runs them against an LLM backend at varying safety levels, and surfaces system, model, and response metrics through a Streamlit interface.

⸻

What CalibrAI Does

CalibrAI helps answer a very specific question:

How safe is this model, really, and what happens when I turn the knobs?

Specifically, it allows you to:
	•	Generate industry-specific or generic test queries
	•	Run the same query across multiple safety or moderation levels
	•	Compare responses for consistency, refusal behavior, and degradation
	•	Monitor system health (CPU, RAM, GPU) during inference
	•	Demo AI safety concepts even when the backend is fragile or mocked

⸻

Project Structure

calibrai/
├── app.py                # Streamlit UI and orchestration
├── src/
│   └── models/
│       └── llm_backend.py # LLM interface (real or mocked)
├── requirements.txt
└── README.md


⸻

app.py (Streamlit UI)

app.py is the main entry point. It handles:
	•	Streamlit layout and user interaction
	•	Safety-level selection and iteration
	•	System metric collection (CPU, RAM, GPU via psutil and GPUtil)
	•	Threaded execution for running multiple safety tests in parallel
	•	Graceful fallback to a mock backend if the real LLM backend is unavailable

Key Design Choices
	•	Tokenizer parallelism disabled to avoid common HuggingFace deadlocks
	•	ThreadPoolExecutor used to keep the UI responsive
	•	MockBackend fallback ensures demos do not fail completely if models disconnect

⸻

llm_backend.py (Model Interface)

llm_backend.py abstracts away the actual LLM implementation so the UI does not care whether it is talking to:
	•	A local LLaMA / DeepSeek model
	•	A vLLM or CPU-based backend
	•	A mocked response generator for testing and demos

Expected Backend Responsibilities

The backend exposes methods such as:
	•	generate_test_query(industry)
	•	test_with_safety_level(query, safety_level)

This separation makes it easy to swap models without rewriting the UI.

⸻

System Metrics & Observability

CalibrAI surfaces live system metrics alongside model output:
	•	CPU utilization
	•	RAM usage (GB)
	•	GPU availability, utilization, and memory
	•	GPU name (if present)

This makes performance and stability tradeoffs visible, not theoretical.

⸻

Running the App

1. Install dependencies

pip install -r requirements.txt

2. Launch Streamlit

streamlit run app.py

The app will automatically fall back to a mock backend if the real LLM backend cannot be imported.

⸻

Use Cases
	•	AI safety and alignment demos
	•	Comparing refusal vs compliance behavior across safety levels
	•	Hackathons and live judging environments
	•	Teaching AI governance and moderation concepts
	•	Stress-testing local or edge-deployed LLMs

⸻

Known Limitations
	•	Model restarts will invalidate in-memory sessions
	•	GPU metrics depend on system drivers and permissions
	•	Safety evaluation is qualitative, not a formal benchmark

This tool is meant for exploration, not certification.

⸻

Why CalibrAI Exists

Because “trust me, it worked earlier” should not be your demo strategy.

CalibrAI makes AI safety behavior observable, comparable, and explainable under pressure.

⸻

License

MIT (or whatever you decide when you’re done winning arguments with judges)
