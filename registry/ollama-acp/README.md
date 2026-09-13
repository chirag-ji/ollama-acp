# Ollama ACP

A local, ACP-native coding agent for JetBrains IDEs backed by Ollama.

It can inspect, edit, and test your code, run shell commands, and iterate until the task is done. All filesystem, terminal, and permission operations run through the ACP client (IntelliJ / JetBrains AI Assistant).

## Install

```bash
npx ollama-acp
```

Or build from source:

```bash
git clone https://github.com/chirag-ji/ollama-acp.git
cd ollama-acp
npm install
npm run build
```

## Setup

Run the interactive terminal setup to configure the Ollama server URL and an optional API key for remote/authenticated servers:

```bash
npx ollama-acp --setup
```

Credentials are stored in `~/.ollama-acp/state.json`.

## Requirements

- Node.js 20+
- Ollama running at `http://127.0.0.1:11434` (or a remote Ollama server)
- A tool-capable model (e.g. `qwen3-coder`)
- IntelliJ IDEA with ACP support in AI Assistant