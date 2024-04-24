# Omnivore summarizer

## Installation

*(Optional)* Install [uv package manager](https://github.com/astral-sh/uv)

**Then run**:

```bash
make init # or copy the pip commands from the Makefile if no uv
```

**After that, fill in `.env-example` and rename it to `.env`.**

## Usage

```bash
make run # or python main.py
```

## Tech Stack

- omnivoreQL - graphQL API for Omnivore APP [(link)](https://github.com/yazdipour/OmnivoreQL)
- LM Studio [(link)](https://lmstudio.ai/)
- OpenAI client to communicate with LM Studio API server
