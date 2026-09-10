#!/usr/bin/env node
import * as acp from "@agentclientprotocol/sdk";
import {ndJsonStream} from "@agentclientprotocol/sdk";
import {OllamaClient, type OllamaMessage, type OllamaTool} from "./ollama.js";
import {randomUUID} from "node:crypto";
import {readFileSync, writeFileSync, existsSync, mkdirSync, appendFileSync, readdirSync, statSync} from "node:fs";
import {join} from "node:path";
import {homedir} from "node:os";
import {Readable, Writable} from "node:stream";
import {execFile} from "node:child_process";
import * as readline from "node:readline";

export const AGENT_NAME = "ollama-acp";
export const CONFIG_DIR_NAME = ".ollama-acp";

const CONFIG_DIR = join(homedir(), CONFIG_DIR_NAME);
const STATE_FILE = join(CONFIG_DIR, "state.json");
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

type State = { model?: string; thinking?: boolean; contextSize?: number; urls?: string[]; activeUrl?: string; apiKey?: string };

const CONTEXT_SIZES = [4096, 8192, 16384, 32768, 65536, 131072];
const DEFAULT_URL = "http://127.0.0.1:11434";
const ADD_URL_OPTION = "__acp_add_url__";

let supportsElicitationForm = false;
let clientName = "unknown";

function normalizeUrl(raw: string): string {
    const trimmed = raw.trim();
    if (!trimmed) return "";
    return /^https?:\/\//i.test(trimmed) ? trimmed : `http://${trimmed}`;
}

const SKIP_DIRS = new Set(["node_modules", ".git", ".venv", "__pycache__", "build", "dist", ".idea", "target", ".gradle"]);

function globMatch(name: string, pattern: string): boolean {
    const regex = pattern
        .replace(/[.+^${}()|[\]\\]/g, "\\$&")
        .replace(/\*\*/g, "{{GLOBSTAR}}")
        .replace(/\*/g, "[^/]*")
        .replace(/\?/g, "[^/]")
        .replace(/\{\{GLOBSTAR\}\}/g, ".*");
    return new RegExp(`^${regex}$`, "i").test(name);
}

function listDirSync(dirPath: string, pattern?: string): string {
    const entries = readdirSync(dirPath, {withFileTypes: true});
    const lines: string[] = [];
    for (const entry of entries) {
        if (entry.name.startsWith(".") && !pattern) continue;
        if (pattern && !globMatch(entry.name, pattern)) continue;
        lines.push(entry.isDirectory() ? `${entry.name}/` : entry.name);
    }
    lines.sort((a, b) => a.toLowerCase().localeCompare(b.toLowerCase()));
    return lines.length ? lines.join("\n") : "(empty directory)";
}

function searchFilesSync(root: string, pattern: string): string {
    const results: string[] = [];
    function walk(dir: string) {
        if (results.length >= 100) return;
        let entries;
        try { entries = readdirSync(dir, {withFileTypes: true}); } catch { return; }
        for (const entry of entries) {
            if (results.length >= 100) return;
            if (SKIP_DIRS.has(entry.name)) continue;
            const full = join(dir, entry.name);
            if (entry.isDirectory()) {
                walk(full);
            } else if (globMatch(entry.name, pattern) || globMatch(full.replace(root, "").replace(/^\//, ""), pattern)) {
                results.push(full);
            }
        }
    }
    walk(root);
    return results.length ? results.join("\n") : "(no matching files)";
}

function searchContentSync(searchPath: string, query: string, filePattern?: string): string {
    const results: string[] = [];
    let regex: RegExp;
    try { regex = new RegExp(query, "gi"); } catch { regex = new RegExp(query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "gi"); }
    function walk(dir: string) {
        if (results.length >= 100) return;
        let entries;
        try { entries = readdirSync(dir, {withFileTypes: true}); } catch { return; }
        for (const entry of entries) {
            if (results.length >= 100) return;
            if (SKIP_DIRS.has(entry.name)) continue;
            const full = join(dir, entry.name);
            if (entry.isDirectory()) {
                walk(full);
            } else if (filePattern && filePattern !== "*" && !globMatch(entry.name, filePattern)) {
                continue;
            } else if (entry.isFile()) {
                try {
                    const content = readFileSync(full, "utf-8");
                    const lines = content.split("\n");
                    for (let i = 0; i < lines.length; i++) {
                        if (results.length >= 100) return;
                        regex.lastIndex = 0;
                        if (regex.test(lines[i])) {
                            results.push(`${full}:${i + 1}:${lines[i]}`);
                        }
                    }
                } catch {}
            }
        }
    }
    walk(searchPath);
    return results.length ? results.join("\n") : "(no matches found)";
}

function loadState(): State {
    try {
        if (existsSync(STATE_FILE)) {
            return JSON.parse(readFileSync(STATE_FILE, "utf-8"));
        }
    } catch {
    }
    return {};
}

function saveState(state: State): void {
    try {
        if (!existsSync(CONFIG_DIR)) {
            mkdirSync(CONFIG_DIR, {recursive: true});
        }
        writeFileSync(STATE_FILE, JSON.stringify(state, null, 2));
    } catch {
    }
}

function persistUrlsState(urls: string[], activeUrl: string): void {
    saveState({model: ollama.getModel(), thinking: ollama.isThinking(), contextSize: ollama.getNumCtx(), urls, activeUrl, apiKey: ollama.getApiKey()});
}

function ensureUrls(state: State): string[] {
    if (!state.urls || state.urls.length === 0) {
        return [DEFAULT_URL];
    }
    return state.urls;
}

function getActiveUrl(state: State): string {
    const urls = ensureUrls(state);
    if (state.activeUrl && urls.includes(state.activeUrl)) {
        return state.activeUrl;
    }
    return urls[0];
}

const savedState = loadState();
const initialUrls = ensureUrls(savedState);
const initialActiveUrl = getActiveUrl(savedState);
const ollama = new OllamaClient(initialActiveUrl, savedState.model, savedState.thinking, savedState.contextSize, savedState.apiKey);

type Mode = "agent" | "plan";

type Session = {
    id: string;
    cwd: string;
    mode: Mode;
    messages: OllamaMessage[];
    abort?: AbortController;
};

const sessions = new Map<string, Session>();

function localExec(command: string, args: string[], cwd: string): Promise<string> {
    return new Promise((resolve) => {
        const timeoutMs = 120_000;
        const child = execFile(command, args, {cwd, timeout: timeoutMs, maxBuffer: 5 * 1024 * 1024}, (error, stdout, stderr) => {
            const exitCode = error ? (error.code ?? 1) : 0;
            const parts: string[] = [];
            if (stdout) parts.push(String(stdout));
            if (stderr) parts.push(String(stderr));
            const output = parts.join("").trim();
            if (error && !stdout && !stderr) {
                resolve(JSON.stringify({exitCode, error: error.message}, null, 2));
            } else {
                resolve(JSON.stringify({exitCode, output}, null, 2));
            }
        });
        child.on("error", (e) => {
            resolve(JSON.stringify({exitCode: -1, error: e.message}, null, 2));
        });
    });
}

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
    },
    {
        type: "function",
        function: {
            name: "add_url",
            description: "Add a new Ollama server URL to the saved list. The URL will be normalized (http:// prepended if no scheme). After adding, it becomes the active URL.",
            parameters: {
                type: "object",
                required: ["url"],
                properties: {url: {type: "string"}}
            }
        }
    },
    {
        type: "function",
        function: {
            name: "remove_url",
            description: "Remove an Ollama server URL from the saved list. Cannot remove the last URL. If the removed URL was active, the first remaining URL becomes active.",
            parameters: {
                type: "object",
                required: ["url"],
                properties: {url: {type: "string"}}
            }
        }
    },
    {
        type: "function",
        function: {
            name: "list_urls",
            description: "List all saved Ollama server URLs and show which one is currently active.",
            parameters: {
                type: "object",
                properties: {}
            }
        }
    },
    {
        type: "function",
        function: {
            name: "list_directory",
            description: "List files and subdirectories in a directory. Use absolute paths. Returns file names with / for directories.",
            parameters: {
                type: "object",
                required: ["path"],
                properties: {
                    path: {type: "string", description: "Absolute path to the directory to list"},
                    pattern: {type: "string", description: "Optional glob pattern to filter results, e.g. '*.ts' or '**/*.java'"}
                }
            }
        }
    },
    {
        type: "function",
        function: {
            name: "search_files",
            description: "Find files by name pattern in the workspace. Returns matching file paths. Use glob patterns like '**/*.ts' or 'src/**/*.java'.",
            parameters: {
                type: "object",
                required: ["pattern"],
                properties: {
                    pattern: {type: "string", description: "Glob pattern to match files, e.g. '**/*.ts' or 'src/**/test*'"}
                }
            }
        }
    },
    {
        type: "function",
        function: {
            name: "search_content",
            description: "Search for text content across files in the workspace. Returns matching lines with file paths and line numbers.",
            parameters: {
                type: "object",
                required: ["query"],
                properties: {
                    query: {type: "string", description: "Text or regex pattern to search for"},
                    path: {type: "string", description: "Optional directory to search within (absolute path)"},
                    pattern: {type: "string", description: "Optional file pattern filter, e.g. '*.ts' or '*.java'"}
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

async function emitConfigUpdate(client: acp.AgentContext, sessionId: string, models: string[] = []) {
    try {
        const opts = await configOptions(models);
        await client.notify(acp.methods.client.session.update, {
            sessionId,
            update: {sessionUpdate: "config_option_update", configOptions: opts}
        });
    } catch (err) {
        log("emitConfigUpdate failed:", err);
    }
}

async function promptForUrl(client: acp.AgentContext, sessionId: string): Promise<string | null> {
    if (!supportsElicitationForm) {
        throw acp.RequestError.invalidParams(
            undefined,
            `This client (${clientName}) does not support interactive input. Ask the agent in chat instead, e.g.: "add url http://127.0.0.1:11434"`
        );
    }
    try {
        const result = await client.request(acp.methods.client.elicitation.create, {
            mode: "form",
            sessionId,
            message: "Enter the base URL of the Ollama server you want to connect to.",
            requestedSchema: {
                type: "object",
                properties: {
                    url: {
                        type: "string",
                        title: "Ollama URL",
                        description: "e.g. http://127.0.0.1:11434 or https://ollama.example.com",
                        format: "uri",
                        default: DEFAULT_URL
                    }
                },
                required: ["url"]
            }
        });
        if (result.action === "accept" && result.content && typeof result.content.url === "string") {
            return normalizeUrl(result.content.url) || null;
        }
        log("promptForUrl: not accepted:", JSON.stringify(result));
        return null;
    } catch (err) {
        log("promptForUrl failed:", err);
        throw acp.RequestError.invalidParams(
            undefined,
            `Unable to collect a URL from this client (${clientName}). Ask the agent in chat instead, e.g.: "add url http://127.0.0.1:11434"`
        );
    }
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

        try {
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
        } catch (err) {
            const msg = err instanceof Error ? err.message : String(err);
            if (/terminal|shell|not available/i.test(msg)) {
                log("terminal API unavailable, falling back to local exec:", command, rawArgs.join(" "));
                return await localExec(command, rawArgs, cwd);
            }
            throw err;
        }
    }

    if (name === "add_url") {
        const rawUrl = String(args.url).trim();
        if (!rawUrl) return "Error: URL cannot be empty.";
        const url = normalizeUrl(rawUrl);
        const urls = currentUrls();
        if (urls.includes(url)) return `URL ${url} already exists. Current URLs: ${urls.join(", ")}`;
        urls.push(url);
        ollama.setBaseUrl(url);
        ollama.invalidateCapabilities();
        persistUrlsState(urls, url);
        await emitConfigUpdate(client, sessionId);
        return `Added ${url}. It is now the active URL. All saved URLs: ${urls.join(", ")}`;
    }

    if (name === "remove_url") {
        const rawUrl = String(args.url).trim();
        if (!rawUrl) return "Error: URL cannot be empty.";
        const url = normalizeUrl(rawUrl);
        const urls = currentUrls();
        if (!urls.includes(url)) return `URL ${url} not found. Current URLs: ${urls.join(", ")}`;
        if (urls.length === 1) return "Error: cannot remove the last URL. Add another URL first.";
        const updated = urls.filter(u => u !== url);
        const activeUrl = currentActiveUrl();
        let newActive = activeUrl;
        if (activeUrl === url) {
            newActive = updated[0];
            ollama.setBaseUrl(newActive);
            ollama.invalidateCapabilities();
        }
        persistUrlsState(updated, newActive);
        await emitConfigUpdate(client, sessionId);
        return `Removed ${url}. Active URL: ${newActive}. Remaining URLs: ${updated.join(", ")}`;
    }

    if (name === "list_urls") {
        const urls = currentUrls();
        const active = currentActiveUrl();
        const lines = urls.map(u => u === active ? `${u} (active)` : u);
        return `Saved URLs:\n${lines.join("\n")}`;
    }

    if (name === "list_directory") {
        const path = String(args.path);
        const pattern = typeof args.pattern === "string" ? args.pattern : undefined;
        try {
            return listDirSync(path, pattern);
        } catch (err) {
            return `Tool error: ${err instanceof Error ? err.message : String(err)}`;
        }
    }

    if (name === "search_files") {
        const pattern = String(args.pattern);
        try {
            return searchFilesSync(session.cwd, pattern);
        } catch (err) {
            return `Tool error: ${err instanceof Error ? err.message : String(err)}`;
        }
    }

    if (name === "search_content") {
        const query = String(args.query);
        const searchPath = typeof args.path === "string" ? args.path : session.cwd;
        const filePattern = typeof args.pattern === "string" ? args.pattern : undefined;
        try {
            return searchContentSync(searchPath, query, filePattern);
        } catch (err) {
            return `Tool error: ${err instanceof Error ? err.message : String(err)}`;
        }
    }

    return `Unknown tool: ${name}`;
}

async function runAgentTurn(session: Session, client: acp.AgentContext, userText: string): Promise<acp.StopReason> {
    const system = [
        "You are Ollama ACP, a local autonomous coding agent running inside IntelliJ IDEA.",
        "You are not a chat-only assistant. You can inspect files, modify files, run commands, run tests/builds, and iterate.",
        `Workspace: ${session.cwd}`,
        `Mode: ${session.mode}`,
        session.mode === "plan"
            ? "PLAN MODE: do not modify files or run mutating commands. Inspect and produce a concrete implementation plan."
            : "AGENT MODE: autonomously work toward the user's goal. Inspect first, make focused edits, run relevant checks, fix failures, and summarize the result.",
        "CRITICAL WORKFLOW: Before making any changes, explore the workspace thoroughly. Use list_directory and search_files to understand the project structure. Use search_content to find all usages of functions/classes you plan to modify. Use read_file to read related files that might need updates. Never assume file contents — always read them first.",
        "When modifying a file, check for imports, usages, tests, and related code that may also need changes. Use search_content to find all references before editing.",
        "Prefer small, verifiable changes. Never invent file contents when you can read them.",
        "Use tools instead of merely telling the user what they could do.",
        "URL management: the \"Ollama URL\" dropdown in the IDE config UI has an \"Add new URL...\" entry that lets the user type a new server URL directly. You can also use list_urls to show saved URLs, add_url to add a server URL, remove_url to remove one. When the user asks to connect/switch to an Ollama server or mentions a URL, prefer list_urls/add_url."
    ].join("\n");

    if (session.messages.length === 0) session.messages.push({role: "system", content: system});
    session.messages.push({role: "user", content: userText});

    const maxSteps = Number(process.env.MAX_AGENT_STEPS ?? 200);
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
                kind: call.function.name === "read_file" || call.function.name === "list_urls" || call.function.name === "list_directory" || call.function.name === "search_files" || call.function.name === "search_content" ? "read" : call.function.name === "write_file" || call.function.name === "add_url" || call.function.name === "remove_url" ? "edit" : "execute",
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
        content: {type: "text", text: `Reached the maximum of ${maxSteps} agent steps. Send another message to continue, or stop if the task is complete.`}
    });
    return "max_turn_requests";
}

const app = acp.agent({name: AGENT_NAME});

async function applySetup(url: string, apiKey: string): Promise<void> {
    const state = loadState();
    const normalized = normalizeUrl(url) || DEFAULT_URL;
    const urls = state.urls && state.urls.length > 0 ? state.urls : [DEFAULT_URL];
    if (!urls.includes(normalized)) urls.push(normalized);

    const cleanedApiKey = apiKey.trim() || undefined;
    const newState: State = {
        model: state.model,
        thinking: state.thinking,
        contextSize: state.contextSize,
        urls,
        activeUrl: normalized,
        apiKey: cleanedApiKey
    };
    saveState(newState);

    process.stderr.write(`\nSaved to ${STATE_FILE}\n`);
    process.stderr.write(`  URL: ${normalized}\n`);
    process.stderr.write(`  API key: ${cleanedApiKey ? "(set)" : "(none)"}\n`);
    process.stderr.write("\nSetup complete. You can now use this agent with your IDE.\n\n");
}

async function runSetup(): Promise<void> {
    process.stderr.write("\n=== Ollama ACP — Setup ===\n\n");

    if (process.stdin.isTTY) {
        const rl = readline.createInterface({input: process.stdin, output: process.stderr});
        try {
            const question = (q: string): Promise<string> => new Promise(resolve => rl.question(q, resolve));
            const url = await question(`Ollama server URL [${DEFAULT_URL}]: `);
            const apiKey = await question("API key (leave empty for local Ollama, no auth): ");
            await applySetup(url, apiKey);
        } finally {
            rl.close();
        }
        return;
    }

    const chunks: Buffer[] = [];
    for await (const chunk of process.stdin) chunks.push(chunk);
    const input = Buffer.concat(chunks).toString("utf-8");
    const lines = input.split("\n");
    const url = lines[0]?.trim() ?? "";
    const apiKey = lines[1]?.trim() ?? "";
    await applySetup(url, apiKey);
}

app.onRequest("initialize", (ctx: any) => {
    log("initialize", "client:", JSON.stringify(ctx.params?.clientInfo));
    const caps = ctx.params?.clientCapabilities;
    supportsElicitationForm = Boolean(caps?.elicitation?.form);
    clientName = ctx.params?.clientInfo?.name ?? "unknown";
    return {
        protocolVersion: acp.PROTOCOL_VERSION,
        agentCapabilities: {
            loadSession: false,
            promptCapabilities: {image: false, audio: false, embeddedContext: true}
        },
        authMethods: [{
            id: "ollama-acp-setup",
            name: "Set up Ollama connection",
            description: "Run an interactive setup to configure the Ollama server URL and an optional API key for authenticated/remote servers.",
            type: "terminal",
            args: ["--setup"]
        }]
    };
});

function currentUrls(): string[] {
    return ensureUrls(loadState());
}

function currentActiveUrl(): string {
    return getActiveUrl(loadState());
}

async function configOptions(models: string[] = []): Promise<any[]> {
    const currentModel = ollama.getModel();
    const allModels = Array.from(new Set([currentModel, ...models]));
    log("configOptions:", "current:", currentModel, "fromApi:", JSON.stringify(models), "merged:", JSON.stringify(allModels));

    let capabilities: string[] = [];
    try {
        const info = await ollama.getModelCapabilities(currentModel);
        capabilities = info.capabilities;
    } catch {
    }

    const supportsThinking = capabilities.includes("thinking");

    if (!supportsThinking && ollama.isThinking()) {
        ollama.setThinking(false);
        saveState({
            model: currentModel,
            thinking: false,
            contextSize: ollama.getNumCtx(),
            urls: currentUrls(),
            activeUrl: currentActiveUrl(),
            apiKey: ollama.getApiKey()
        });
    }

    const thinkingDescription = supportsThinking
        ? "Enable reasoning tokens (think mode)."
        : "This model does not support thinking. Reasoning tokens are disabled.";

    const urls = currentUrls();
    const activeUrl = currentActiveUrl();

    return [
        {
            id: "ollama_model",
            type: "select",
            name: "Ollama Model",
            description: "Model to use for chat completions",
            category: "model",
            currentValue: currentModel,
            options: allModels.map(m => ({value: m, name: m})),
            _meta: {capabilities}
        },
        {
            id: "ollama_url",
            type: "select",
            name: "Ollama URL",
            description: "Base URL of the Ollama server. Choose an existing URL, or pick \"Add new URL...\" to enter a new one.",
            currentValue: activeUrl,
            options: [
                ...urls.map(u => ({value: u, name: u})),
                {value: ADD_URL_OPTION, name: "Add new URL..."}
            ]
        },
        {
            id: "ollama_thinking",
            type: "select",
            name: "Thinking / Reasoning",
            description: thinkingDescription,
            category: "thought_level",
            currentValue: ollama.isThinking() ? "true" : "false",
            options: [
                {value: "true", name: "Enabled"},
                {value: "false", name: "Disabled"}
            ],
            _meta: {capabilities, supported: supportsThinking}
        },
        {
            id: "ollama_context_size",
            type: "select",
            name: "Context Size",
            description: "Number of context tokens (num_ctx) sent to the model per request.",
            category: "context",
            currentValue: String(ollama.getNumCtx()),
            options: CONTEXT_SIZES.map(s => ({value: String(s), name: String(s)}))
        }
    ];
}

app.onRequest("session/new", async (ctx: any) => {
    const id = randomUUID();
    log("session/new", id, "cwd:", ctx.params.cwd);
    const models = await ollama.listModels().catch((err) => { log("listModels failed:", err); return []; });
    log("session/new models:", JSON.stringify(models), "current:", ollama.getModel());
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
        configOptions: await configOptions(models)
    };
});

app.onRequest("session/set_config_option", async (ctx: any) => {
    const session = sessions.get(ctx.params.sessionId);
    if (!session) throw acp.RequestError.invalidParams(undefined, "Unknown session");
    if (ctx.params.configId === "ollama_url" && typeof ctx.params.value === "string") {
        const urls = currentUrls();
        if (ctx.params.value === ADD_URL_OPTION) {
            const newUrl = await promptForUrl(ctx.client, ctx.params.sessionId);
            if (newUrl) {
                const updated = urls.includes(newUrl) ? urls : [...urls, newUrl];
                ollama.setBaseUrl(newUrl);
                ollama.invalidateCapabilities();
                persistUrlsState(updated, newUrl);
                log("set_config_option added url:", newUrl);
            }
        } else if (!urls.includes(ctx.params.value)) {
            throw acp.RequestError.invalidParams(undefined, "Unknown URL. Use add_url to add new URLs.");
        } else {
            ollama.setBaseUrl(ctx.params.value);
            ollama.invalidateCapabilities();
            persistUrlsState(urls, ctx.params.value);
        }
    }
    if (ctx.params.configId === "ollama_model" && typeof ctx.params.value === "string") {
        ollama.setModel(ctx.params.value);
        ollama.invalidateCapabilities(ctx.params.value);
        let thinking = ollama.isThinking();
        try {
            const info = await ollama.getModelCapabilities(ctx.params.value);
            if (thinking && !info.capabilities.includes("thinking")) {
                thinking = false;
                ollama.setThinking(false);
            }
        } catch {}
        saveState({model: ctx.params.value, thinking, contextSize: ollama.getNumCtx(), urls: currentUrls(), activeUrl: currentActiveUrl(), apiKey: ollama.getApiKey()});
    }
    if (ctx.params.configId === "ollama_thinking" && typeof ctx.params.value === "string") {
        const thinking = ctx.params.value === "true";
        ollama.setThinking(thinking);
        saveState({model: ollama.getModel(), thinking, urls: currentUrls(), activeUrl: currentActiveUrl(), apiKey: ollama.getApiKey()});
    }
    if (ctx.params.configId === "ollama_context_size" && typeof ctx.params.value === "string") {
        const contextSize = Number(ctx.params.value);
        ollama.setNumCtx(contextSize);
        saveState({model: ollama.getModel(), thinking: ollama.isThinking(), contextSize, urls: currentUrls(), activeUrl: currentActiveUrl(), apiKey: ollama.getApiKey()});
    }
    const models = await ollama.listModels().catch((err) => { log("listModels failed on config change:", err); return []; });
    log("set_config_option models:", JSON.stringify(models));
    const opts = await configOptions(models);
    await emitConfigUpdate(ctx.client, ctx.params.sessionId, models);
    return {configOptions: opts};
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

if (process.argv.includes("--setup")) {
    runSetup().then(() => process.exit(0)).catch(err => {
        process.stderr.write(`Setup failed: ${err instanceof Error ? err.message : err}\n`);
        process.exit(1);
    });
} else {
    const input = Writable.toWeb(process.stdout) as unknown as WritableStream<Uint8Array>;
    const output = Readable.toWeb(process.stdin) as unknown as ReadableStream<Uint8Array>;
    const stream = ndJsonStream(input, output);
    app.connect(stream);
}
