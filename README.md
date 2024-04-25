# Omnivore summarizer

## Installation

*(Optional)* Install [uv package manager](https://github.com/astral-sh/uv)

**Then run**:

```bash
make init # or copy the pip commands from the Makefile if no uv
```

**After that**:

1. Fill in `.env-example` and rename it to `.env`.
   - `OMNIVORE_API_KEY` - your Omnivore API key
   - `OLLAMA_MODEL_ID` - your Ollama model ID
2. Create `summarize` label in your Omnivore APP (will be automated in the future)
3. Download LLaMa3 model:
   - [Ollama LlaMa3](https://ollama.com/library/llama3)
4. run `ollama serve` to start the Ollama server

## Usage

```bash
make run # or python main.py
```

## Tech Stack

- Omnivore ReaderAPP
- omnivoreQL - graphQL API for Omnivore APP [(link)](https://github.com/yazdipour/OmnivoreQL)
- Ollama [(link)](https://ollama.com/)
