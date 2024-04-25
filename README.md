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
2. Create `summarized` and `read later` labels in your Omnivore App (will be automated in the future)
3. *(Optional)* Create saved searches in your Omnivore App for the labels you created
4. Download LLaMa3 model:
   - [Ollama LlaMa3](https://ollama.com/library/llama3)
   - also works with other models, but tokenization estimation might be off (for now)
5. run `ollama serve` to start the Ollama server

## Usage

```bash
make run # or python main.py
```

### Flow description

**Work in progress!**

1. Fetches all subscriptions from the last 24 hours
2. Filters out artcles that are larger then context of the model (8192)
3. Summarizes the articles with later options to `read later` or `archive`
4. TBA

## Tech Stack

- Omnivore App [(link)](https://omnivore.app/)
- omnivoreQL - graphQL API for Omnivore APP [(link)](https://github.com/yazdipour/OmnivoreQL)
- Ollama [(link)](https://ollama.com/)
