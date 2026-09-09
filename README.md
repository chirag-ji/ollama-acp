# Ollama IntelliJ ACP Agent

A local, ACP-native coding agent for IntelliJ IDEA / JetBrains AI Assistant backed by Ollama.

It is deliberately **not chat-only**. The agent loop can:

- inspect workspace files through ACP `fs/read_text_file`
- edit/create files through ACP `fs/write_text_file`
- run builds, tests, git, grep/rg, package-manager commands through ACP terminals
- ask IntelliJ for permission before edits/commands
- stream agent messages, thoughts, and tool-call status into the IDE
- expose **Agent** and **Plan** session modes
- use a local Ollama model with function/tool calling
- keep multi-turn session history in-process

## Requirements

- Node.js 20+
- Ollama running locally at `http://127.0.0.1:11434`
- a tool-capable coding model. `qwen3-coder` is a good starting point.
- IntelliJ IDEA with ACP support in AI Assistant

Ollama's local API does not require authentication. Ollama supports tool calling and multi-turn agent loops through `/api/chat`. See the official docs for the API and tool-calling behavior.

## Install

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
node /absolute/path/to/ollama-intellij-acp/dist/index.js
```

Environment variables:

```text
OLLAMA_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3-coder
MAX_AGENT_STEPS=40
```

The exact UI location for adding a custom ACP agent can vary by JetBrains release. ACP is the integration layer; this process is the agent, while IntelliJ remains the client that owns files, terminals, permissions, and presentation.

## Modes

### Agent

Autonomous mode. The model is instructed to inspect, edit, run commands, test, and iterate. File writes and command execution go through IntelliJ's ACP permission flow.

### Plan

Read-only mode. File writes are rejected, and only clearly read-oriented commands are allowed. Use this to make the agent produce an implementation plan before switching to Agent mode.

## Architecture

```text
IntelliJ IDEA / AI Assistant
        |
        | ACP / JSON-RPC over stdio
        v
ollama-intellij-acp
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
2. model selector backed by live Ollama model metadata
3. ACP config options for model, thinking level, approval policy, and auto-run
4. structured plans (`plan` session updates)
5. richer file diffs in `tool_call_update`
6. cancellation propagation into Ollama HTTP requests
7. parallel tool calls
8. context compaction/summarization for long sessions
9. IntelliJ diagnostics/test/build result ingestion
10. optional MCP server forwarding from `session/new`
11. image/vision prompt support for vision-capable Ollama models
12. background tasks/subagents where supported by the ACP client

These are additions to the agent; they do not require turning it into an IntelliJ plugin. ACP is specifically intended to let an IDE connect to a coding agent without a bespoke IDE-agent integration.
