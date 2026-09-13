# Ollama ACP Agent

A local, ACP-native coding agent for IntelliJ IDEA / JetBrains AI Assistant backed by Ollama.

It is deliberately **not chat-only**. The agent loop can:

- inspect workspace files through ACP `fs/read_text_file`
- edit/create files through ACP `fs/write_text_file`
- run builds, tests, git, grep/rg, package-manager commands through ACP terminals
- ask IntelliJ for permission before edits/commands
- stream agent messages, thoughts, and tool-call status into the IDE
- keep the IDE's thinking shimmer above the input visible while the agent works (progress heartbeat, tunable via `OLLAMA_PROGRESS_HEARTBEAT_MS`)
- expose **Agent** and **Plan** session modes
- use a local Ollama model with function/tool calling
- keep multi-turn session history in-process
- keep context sized to the selected model: options are labeled by size and auto-clamped to the model's max context

## Requirements

- Node.js 20+
- Ollama running locally at `http://127.0.0.1:11434`
- a tool-capable coding model. `qwen3-coder` is a good starting point.
- IntelliJ IDEA with ACP support in AI Assistant

Ollama's local API does not require authentication. For remote/authenticated servers, a Bearer token can be configured at setup. Ollama supports tool calling and multi-turn agent loops through `/api/chat`. See the official docs for the API and tool-calling behavior.

## Install from the ACP registry

`ollama-acp` is listed in the [ACP registry](https://github.com/agentclientprotocol/registry) and can be run directly with `npx`:

```bash
npx ollama-acp
```

When a JetBrains IDE with ACP support asks for the agent command, use:

```text
npx ollama-acp@latest
```

The first run through the IDE's authentication flow launches an interactive terminal setup where you can configure the Ollama server URL and an optional API key. You can also run the same setup manually:

```bash
npx ollama-acp --setup
```

## Build from source

```bash
npm install
npm run build
```

Try the agent directly:

```bash
OLLAMA_MODEL=qwen3-coder npm start
```

ACP uses stdin/stdout for JSON-RPC, so **do not print logs to stdout**. If you add logging, use stderr.

## IntelliJ / JetBrains setup

JetBrains AI Assistant supports external ACP agents. Add a custom ACP agent and point it at the built launcher:

**Command**

```text
node /absolute/path/to/ollama-acp/dist/index.js
```

Environment variables:

```text
OLLAMA_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3-coder
MAX_AGENT_STEPS=40
```

Working shimmer: while a turn is in flight, the agent sends a progress heartbeat on a timer so the IDE keeps its thinking shimmer above the input visible even during long tool calls or silent model thinking. The heartbeat interval in milliseconds can be tuned with `OLLAMA_PROGRESS_HEARTBEAT_MS` (default `4000`, minimum `1000`).

Optional environment variable for remote/authenticated Ollama servers:

```text
OLLAMA_API_KEY=
```

The API key, when set, is sent as a `Bearer` token on every request. It can also be stored via the terminal setup (`--setup`) in `~/.ollama-acp/state.json` instead of an environment variable.

The exact UI location for adding a custom ACP agent can vary by JetBrains release. ACP is the integration layer; this process is the agent, while IntelliJ remains the client that owns files, terminals, permissions, and presentation.

## Modes

### Agent

Autonomous mode. The model is instructed to inspect, edit, run commands, test, and iterate. File writes and command execution go through IntelliJ's ACP permission flow.

### Plan

Read-only mode. File writes are rejected, and only clearly read-oriented commands are allowed. Use this to make the agent produce an implementation plan before switching to Agent mode.

## Configuration

Live agent settings appear in the IDE's agent configuration panel as ACP config options. Changes apply immediately and persist to `~/.ollama-acp/state.json`:

- **Ollama Model** — choose from the models the current server reports.
- **Ollama URL** — switch between saved server URLs, or pick "Add new URL..." to type a new one.
- **Thinking / Reasoning** — enable or disable reasoning tokens. Automatically disabled when the selected model reports no thinking support.
- **Context Size** — the `num_ctx` sent with each request. Options are labeled by size: `Very small (4k)`, `Small (8k)`, `Medium (16k)`, `Large (32k)`, `Very large (64k)`, `Extra large (128k)`.

The context size stays in sync with the selected model in real time. On every session, model switch, or connection retry, the agent reads the model's maximum context length (`context_length` from Ollama's `/api/show`, reported in `_meta.maxContext`), hides options above that limit, and automatically clamps the active context size down to the model's max — so a small model is never asked to run an oversized context window.

## Architecture

```text
IntelliJ IDEA / AI Assistant
        |
        | ACP / JSON-RPC over stdio
        v
ollama-acp
        |
        +-- ACP session + permission + tool-call updates
        |
        +-- Agent loop
        |     |
        |     +-- read_file  -> IntelliJ fs/read_text_file
        |     +-- write_file -> IntelliJ fs/write_text_file
        |     +-- run_command-> IntelliJ terminal/*
        |
        +-- Ollama /api/chat
              |
              +-- local model + tool calls
```

The important design choice is that the agent does **not** directly manipulate the IntelliJ project filesystem or shell. It asks the ACP client to perform those operations, which is what makes the integration compatible with IDE-side permission and tool UI.

## Extending this toward a full coding agent

The current project is a strong ACP baseline, not a reimplementation of every JetBrains-native feature. The next high-value additions are:

1. session persistence/load/resume
2. ACP config options for approval policy and auto-run
3. structured plans (`plan` session updates)
4. richer file diffs in `tool_call_update`
5. parallel tool calls
6. IntelliJ diagnostics/test/build result ingestion
7. optional MCP server forwarding from `session/new`
8. image/vision prompt support for vision-capable Ollama models
9. background tasks/subagents where supported by the ACP client

These are additions to the agent; they do not require turning it into an IntelliJ plugin. ACP is specifically intended to let an IDE connect to a coding agent without a bespoke IDE-agent integration.
