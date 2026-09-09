import * as acp from "@agentclientprotocol/sdk";
import {ndJsonStream} from "@agentclientprotocol/sdk";
import {OllamaClient, type OllamaMessage, type OllamaTool} from "./ollama.js";
import {randomUUID} from "node:crypto";
import {readFileSync, writeFileSync, existsSync, mkdirSync, appendFileSync} from "node:fs";
import {join} from "node:path";
import {homedir} from "node:os";
import {Readable, Writable} from "node:stream";

const CONFIG_DIR = join(homedir(), ".ollama-intellij-acp");
const CONFIG_FILE = join(CONFIG_DIR, "config.json");
const LOG_FILE = join(CONFIG_DIR, "agent.log");

function log(...args: unknown[]): void {
    try {
        if (!existsSync(CONFIG_DIR)) mkdirSync(CONFIG_DIR, {recursive: true});
        const ts = new Date().toISOString();
        const msg = args.map(a => (typeof a === "string" ? a : JSON.stringify(a, null, 2))).join(" ");
        appendFileSync(LOG_FILE, `[${ts}] ${msg}\n`);
    } catch (err) {
        process.stderr.write(`[log error] ${err instanceof Error ? err.message : String(err)}\n`);
    }
}

function loadConfig(): { model?: string; baseUrl?: string; thinking?: boolean } {
    try {
        if (existsSync(CONFIG_FILE)) {
            return JSON.parse(readFileSync(CONFIG_FILE, "utf-8"));
        }
    } catch {
    }
    return {};
}

function saveConfig(config: { model?: string; baseUrl?: string; thinking?: boolean }): void {
    try {
        if (!existsSync(CONFIG_DIR)) {
            mkdirSync(CONFIG_DIR, {recursive: true});
        }
        writeFileSync(CONFIG_FILE, JSON.stringify(config, null, 2));
    } catch {
    }
}

const savedConfig = loadConfig();
const ollama = new OllamaClient(savedConfig.baseUrl, savedConfig.model, savedConfig.thinking);

type Mode = "agent" | "plan";

type Session = {
    id: string;
    cwd: string;
    mode: Mode;
    messages: OllamaMessage[];
    abort?: AbortController;
};

const sessions = new Map<string, Session>();

const TOOLS: OllamaTool[] = [
    {
        type: "function",
        function: {
            name: "read_file",
            description: "Read a text file from the IntelliJ workspace. Use absolute paths.",
            parameters: {
                type: "object",
                required: ["path"],
                properties: {path: {type: "string"}, line: {type: "integer"}, limit: {type: "integer"}}
            }
        }
    },
    {
        type: "function",
        function: {
            name: "write_file",
            description: "Create or replace a text file in the IntelliJ workspace. Use absolute paths.",
            parameters: {
                type: "object",
                required: ["path", "content"],
                properties: {path: {type: "string"}, content: {type: "string"}}
            }
        }
    },
    {
        type: "function",
        function: {
            name: "run_command",
            description: "Run a shell command in the IntelliJ workspace. Use for builds, tests, git, grep/rg, package managers, and other development tasks.",
            parameters: {
                type: "object",
                required: ["command"],
                properties: {
                    command: {type: "string"},
                    args: {type: "array", items: {type: "string"}},
                    cwd: {type: "string"}
                }
            }
        }
    }
];

export function textFromPrompt(prompt: acp.PromptRequest["prompt"]): string {
    return prompt
        .map((p: any) => {
            if (p.type === "text") {
                return p.text;
            }
            if (p.type === "resource") {
                const res = p.resource;
                if (res && typeof res === "object") {
                    const uri = res.uri ?? "";
                    const text = res.text ?? "";
                    return `[File: ${uri}]\n${text}`;
                }
            }
            if (p.type === "resource_link") {
                const uri = p.uri ?? "";
                const name = p.name ?? "";
                return `[File: ${name} (${uri})]`;
            }
            return "";
        })
        .filter(Boolean)
        .join("\n");
}

function modeState(mode: Mode): any {
    return {
        currentModeId: mode,
        availableModes: [
            {
                id: "agent",
                name: "Agent",
                description: "Autonomous coding mode: inspect, edit, run commands, test, and iterate."
            },
            {
                id: "plan",
                name: "Plan",
                description: "Read-only planning mode. The agent can inspect the workspace and propose changes without modifying files."
            }
        ]
    };
}

async function emitUpdate(client: acp.AgentContext, sessionId: string, update: any) {
    await client.notify(acp.methods.client.session.update, {sessionId, update});
}

async function requestPermission(
    client: acp.AgentContext,
    sessionId: string,
    title: string,
    kind: "edit" | "execute",
    location?: string
): Promise<boolean> {
    const result = await client.request(acp.methods.client.session.requestPermission, {
        sessionId,
        toolCall: {
            toolCallId: randomUUID(),
            title,
            kind,
            status: "pending",
            locations: location ? [{path: location}] : []
        },
        options: [
            {optionId: "allow_once", kind: "allow_once", name: "Allow once"},
            {optionId: "reject_once", kind: "reject_once", name: "Reject"}
        ]
    });
    return result.outcome.outcome !== "cancelled" && result.outcome.optionId === "allow_once";
}

async function executeTool(
    session: Session,
    client: acp.AgentContext,
    sessionId: string,
    name: string,
    args: Record<string, unknown>
): Promise<string> {
    if (name === "read_file") {
        const path = String(args.path);
        const line = typeof args.line === "number" ? args.line : undefined;
        const limit = typeof args.limit === "number" ? args.limit : undefined;
        const response = await client.request(acp.methods.client.fs.readTextFile, {
            sessionId, path, line, limit
        });
        return response.content;
    }

    if (name === "write_file") {
        if (session.mode === "plan") return "DENIED: plan mode is read-only.";
        const path = String(args.path);
        const content = String(args.content);
        const allowed = await requestPermission(client, sessionId, `Write ${path}`, "edit", path);
        if (!allowed) return "DENIED by user.";
        await client.request(acp.methods.client.fs.writeTextFile, {sessionId, path, content});
        return `Wrote ${path}`;
    }

    if (name === "run_command") {
        const command = String(args.command);
        const rawArgs = Array.isArray(args.args) ? args.args.map(String) : [];
        const cwd = typeof args.cwd === "string" ? args.cwd : session.cwd;

        if (session.mode === "plan") {
            // Plan mode only permits explicitly read-only commands.
            const readOnly = /^(git\s+(status|diff|log|show|branch)|rg\b|grep\b|find\b|ls\b|pwd\b|cat\b|head\b|tail\b|sed\b|npm\s+(list|outdated)|pnpm\s+(list|outdated)|yarn\s+(list|outdated)|mvn\s+help:)/i.test(command);
            if (!readOnly) return "DENIED: command is not read-only in plan mode.";
        }

        if (session.mode === "agent") {
            const allowed = await requestPermission(client, sessionId, `Run: ${command} ${rawArgs.join(" ")}`, "execute");
            if (!allowed) return "DENIED by user.";
        }

        const created = await client.request(acp.methods.client.terminal.create, {
            sessionId,
            command,
            args: rawArgs,
            cwd,
            outputByteLimit: 200000
        });
        try {
            const waited = await client.request(acp.methods.client.terminal.waitForExit, {
                sessionId,
                terminalId: created.terminalId
            });
            const output = await client.request(acp.methods.client.terminal.output, {
                sessionId,
                terminalId: created.terminalId
            });
            return JSON.stringify({exitCode: waited.exitCode, signal: waited.signal, output: output.output}, null, 2);
        } finally {
            await client.request(acp.methods.client.terminal.release, {
                sessionId,
                terminalId: created.terminalId
            }).catch(() => undefined);
        }
    }

    return `Unknown tool: ${name}`;
}

async function runAgentTurn(session: Session, client: acp.AgentContext, userText: string): Promise<acp.StopReason> {
    const system = [
        "You are Ollama IntelliJ ACP, a local autonomous coding agent running inside IntelliJ IDEA.",
        "You are not a chat-only assistant. You can inspect files, modify files, run commands, run tests/builds, and iterate.",
        `Workspace: ${session.cwd}`,
        `Mode: ${session.mode}`,
        session.mode === "plan"
            ? "PLAN MODE: do not modify files or run mutating commands. Inspect and produce a concrete implementation plan."
            : "AGENT MODE: autonomously work toward the user's goal. Inspect first, make focused edits, run relevant checks, fix failures, and summarize the result.",
        "Prefer small, verifiable changes. Never invent file contents when you can read them.",
        "Use tools instead of merely telling the user what they could do."
    ].join("\n");

    if (session.messages.length === 0) session.messages.push({role: "system", content: system});
    session.messages.push({role: "user", content: userText});

    const maxSteps = Number(process.env.MAX_AGENT_STEPS ?? 40);
    for (let step = 0; step < maxSteps; step++) {
        if (session.abort?.signal.aborted) return "cancelled";

        await emitUpdate(client, session.id, {
            sessionUpdate: "agent_thought_chunk",
            content: {type: "text", text: `Step ${step + 1}: reasoning about the next action…`}
        });

        const response = await ollama.chat(session.messages, TOOLS);
        const assistant = response.message;
        log("ollama response", session.id, "step:", step + 1, "tool_calls:", assistant.tool_calls?.length ?? 0, "content:", (assistant.content ?? "").slice(0, 200));
        if (assistant.thinking) {
            await emitUpdate(client, session.id, {
                sessionUpdate: "agent_thought_chunk",
                content: {type: "text", text: assistant.thinking}
            });
        }
        session.messages.push({
            role: "assistant",
            content: assistant.content,
            ...(assistant.tool_calls?.length ? {tool_calls: assistant.tool_calls} : {})
        });

        if (assistant.content) {
            await emitUpdate(client, session.id, {
                sessionUpdate: "agent_message_chunk",
                content: {type: "text", text: assistant.content}
            });
        }

        const calls = assistant.tool_calls ?? [];
        if (calls.length === 0) return "end_turn";

        for (const call of calls) {
            const id = randomUUID();
            const title = `${call.function.name}`;
            await emitUpdate(client, session.id, {
                sessionUpdate: "tool_call",
                toolCallId: id,
                title,
                kind: call.function.name === "read_file" ? "read" : call.function.name === "write_file" ? "edit" : "execute",
                status: "in_progress",
                rawInput: call.function.arguments
            });

            let result: string;
            try {
                log("tool_call", session.id, call.function.name, JSON.stringify(call.function.arguments).slice(0, 300));
                result = await executeTool(session, client, session.id, call.function.name, call.function.arguments);
            } catch (error) {
                result = `Tool error: ${error instanceof Error ? error.message : String(error)}`;
                log("tool_error", session.id, call.function.name, result);
            }
            log("tool_result", session.id, call.function.name, result.slice(0, 300));

            await emitUpdate(client, session.id, {
                sessionUpdate: "tool_call_update",
                toolCallId: id,
                status: result.startsWith("Tool error:") || result.startsWith("DENIED") ? "failed" : "completed",
                content: [{type: "content", content: {type: "text", text: result.slice(0, 12000)}}]
            });

            session.messages.push({
                role: "tool",
                tool_name: call.function.name,
                content: result
            });
        }
    }

    await emitUpdate(client, session.id, {
        sessionUpdate: "agent_message_chunk",
        content: {type: "text", text: `Stopped after ${maxSteps} agent steps. Continue the task to resume.`}
    });
    return "max_turn_requests";
}

const input = Writable.toWeb(process.stdout) as unknown as WritableStream<Uint8Array>;
const output = Readable.toWeb(process.stdin) as unknown as ReadableStream<Uint8Array>;
const stream = ndJsonStream(input, output);
const app = acp.agent({name: "ollama-intellij-acp"});

app.onRequest("initialize", (_ctx: any) => {
    log("initialize");
    return {
        protocolVersion: acp.PROTOCOL_VERSION,
        agentCapabilities: {
            loadSession: false,
            promptCapabilities: {image: false, audio: false, embeddedContext: true}
        },
        authMethods: []
    };
});

function configOptions(models: string[] = []): any[] {
    const currentModel = ollama.getModel();
    const allModels = Array.from(new Set([currentModel, ...models]));
    return [
        {
            id: "ollama_model",
            type: "select",
            name: "Ollama Model",
            description: "Model to use for chat completions",
            category: "model",
            currentValue: currentModel,
            options: allModels.map(m => ({value: m, name: m}))
        },
        {
            id: "ollama_url",
            type: "select",
            name: "Ollama URL",
            description: "Base URL of the Ollama server",
            currentValue: ollama.getBaseUrl(),
            options: [
                {value: ollama.getBaseUrl(), name: ollama.getBaseUrl()}
            ]
        },
        {
            id: "ollama_thinking",
            type: "select",
            name: "Thinking / Reasoning",
            description: "Enable reasoning tokens (think mode). Disable for models that do not support thinking (e.g. qwen-coder2.5).",
            category: "thought_level",
            currentValue: ollama.isThinking() ? "true" : "false",
            options: [
                {value: "true", name: "Enabled"},
                {value: "false", name: "Disabled"}
            ]
        }
    ];
}

app.onRequest("session/new", async (ctx: any) => {
    const id = randomUUID();
    log("session/new", id, "cwd:", ctx.params.cwd);
    const models = await ollama.listModels().catch(() => []);
    const session: Session = {
        id,
        cwd: ctx.params.cwd,
        mode: "agent",
        messages: []
    };
    sessions.set(id, session);
    return {
        sessionId: id,
        modes: modeState(session.mode),
        configOptions: configOptions(models)
    };
});

app.onRequest("session/set_config_option", async (ctx: any) => {
    const session = sessions.get(ctx.params.sessionId);
    if (!session) throw acp.RequestError.invalidParams(undefined, "Unknown session");
    if (ctx.params.configId === "ollama_url" && typeof ctx.params.value === "string") {
        ollama.setBaseUrl(ctx.params.value);
        saveConfig({model: ollama.getModel(), baseUrl: ctx.params.value, thinking: ollama.isThinking()});
    }
    if (ctx.params.configId === "ollama_model" && typeof ctx.params.value === "string") {
        ollama.setModel(ctx.params.value);
        saveConfig({model: ctx.params.value, baseUrl: ollama.getBaseUrl(), thinking: ollama.isThinking()});
    }
    if (ctx.params.configId === "ollama_thinking" && typeof ctx.params.value === "string") {
        const thinking = ctx.params.value === "true";
        ollama.setThinking(thinking);
        saveConfig({model: ollama.getModel(), baseUrl: ollama.getBaseUrl(), thinking});
    }
    const models = await ollama.listModels().catch(() => []);
    return {configOptions: configOptions(models)};
});

app.onRequest("session/set_mode", (ctx: any) => {
    const session = sessions.get(ctx.params.sessionId);
    if (!session) throw acp.RequestError.invalidParams(undefined, "Unknown session");
    const next = ctx.params.modeId as Mode;
    if (next !== "agent" && next !== "plan") throw acp.RequestError.invalidParams(undefined, "Unknown mode");
    session.mode = next;
    return {};
});

app.onRequest("session/prompt", async (ctx: any) => {
    const session = sessions.get(ctx.params.sessionId);
    if (!session) throw acp.RequestError.invalidParams(undefined, "Unknown session");
    session.abort = new AbortController();
    try {
        const text = textFromPrompt(ctx.params.prompt);
        if (!text.trim()) throw acp.RequestError.invalidParams(undefined, "Empty prompt");
        log("session/prompt", session.id, "mode:", session.mode, "text:", text.slice(0, 200));
        const stopReason = await runAgentTurn(session, ctx.client, text);
        log("session/prompt done", session.id, "stopReason:", stopReason);
        return {stopReason};
    } catch (err) {
        log("session/prompt error", session.id, err);
        throw err;
    } finally {
        session.abort = undefined;
    }
});

app.onNotification("session/cancel", (ctx: any) => {
    log("session/cancel", ctx.params.sessionId);
    sessions.get(ctx.params.sessionId)?.abort?.abort();
});

app.connect(stream);
