# Person Search Agent 🕵️

An AI-powered person search agent using MCP (Model Context Protocol), LangChain, and Groq.

## Getting Started

This project uses **`uv`** for dependency management. It is faster and more reliable than `pip` or `conda`.

### Installation

1.  **Install `uv`** (if you haven't already):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
2.  **Sync Dependencies**:
    ```bash
    uv sync
    ```
3.  **Install Playwright Browsers**:
    ```bash
    uv run playwright install chromium
    ```

### How to Run

- **Run Agent (CLI)**:
  ```bash
  uv run agent/agent.py
  ```
- **Run Streamlit UI**:
  ```bash
  uv run streamlit run agent/agent_streamlit_gpt.py
  ```

### Fixing "Yellow Underlines" in VS Code

If you see yellow underlines under your imports (like `dotenv` or `langchain`), it's because VS Code hasn't selected the correct Python interpreter.

1.  Press `Ctrl + Shift + P` in VS Code.
2.  Search for **"Python: Select Interpreter"**.
3.  Choose the one that points to `./.venv/bin/python` (the environment created by `uv`).

Once selected, the warnings will disappear! 🚀
