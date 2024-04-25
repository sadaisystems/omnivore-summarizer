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
   - `LM_STUDIO_MODEL_ID` - your LM Studio model ID used for the server
2. Create `summarize` label in your Omnivore APP (will be automated in the future)
3. Download LLama3 model:
   - [Ollama](https://ollama.com/library/llama3)
   - [LM Studio](https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF)
4. Initialize LM Studio. Server/Ollama

## Usage

```bash
make run # or python main.py
```

## Tech Stack

- Omnivore ReaderAPP
- omnivoreQL - graphQL API for Omnivore APP [(link)](https://github.com/yazdipour/OmnivoreQL)
- LM Studio [(link)](https://lmstudio.ai/)
- OpenAI client to communicate with LM Studio API server
