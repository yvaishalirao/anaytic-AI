# Data Analyst Agent

A small local data analysis agent that accepts a CSV file, runs an autonomous reasoning loop, generates charts, and writes a structured analysis report. The system is designed to analyze local datasets only and does not send your data to external storage.

## Prerequisites

- Python 3.11 or later
- A MyGroq API key
- A working terminal on Windows or macOS/Linux

## Quickstart

1. Install the package in editable mode with development dependencies:

   ```bash
   pip install -e ".[dev]"
   ```

2. Copy the example environment file and set your OpenAI key:

   ```bash
   copy .env.example .env
   ```

3. Run the Streamlit UI:

   ```bash
   streamlit run src/ui/app.py
   ```

## Notes

- Analysis runs on local data only. The uploaded CSV and generated outputs remain on your machine.

