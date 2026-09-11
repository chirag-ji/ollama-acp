import {describe, it, expect} from "vitest";
import {spawn} from "node:child_process";
import {createServer, type Server} from "node:http";
import {mkdtempSync, readFileSync, existsSync, mkdirSync, writeFileSync, appendFileSync, rmSync} from "node:fs";
import {tmpdir} from "node:os";
import {join} from "node:path";
import type {PromptRequest} from "@agentclientprotocol/sdk";
import {
    textFromPrompt,
    AGENT_NAME,
    CONFIG_DIR_NAME,
    snapshotFiles,
    diffSnapshots,
    mergeFileChanges,
    estimateMessagesTokens,
    shouldCompressMessages,
    compressMessages
} from "../src/index.js";
import {OllamaClient} from "../src/ollama.js";

describe("textFromPrompt", () => {
    it("extracts plain text", () => {
        const prompt: PromptRequest["prompt"] = [{type: "text", text: "hello world"}];
        expect(textFromPrompt(prompt)).toBe("hello world");
    });

    it("extracts multiple text blocks", () => {
        const prompt: PromptRequest["prompt"] = [
            {type: "text", text: "first line"},
            {type: "text", text: "second line"}
        ];
        expect(textFromPrompt(prompt)).toBe("first line\nsecond line");
    });

    it("extracts embedded resource (file content)", () => {
        const prompt: PromptRequest["prompt"] = [{
            type: "resource",
            resource: {uri: "file:///path/to/file.ts", text: "const x=1;"}
        }];
        const result = textFromPrompt(prompt);
        expect(result).toContain("file:///path/to/file.ts");
        expect(result).toContain("const x=1;");
    });

    it("extracts resource_link", () => {
        const prompt: PromptRequest["prompt"] = [{
            type: "resource_link",
            name: "index.ts",
            uri: "file:///src/index.ts"
        }];
        const result = textFromPrompt(prompt);
        expect(result).toContain("index.ts");
        expect(result).toContain("file:///src/index.ts");
    });

    it("handles mixed content types", () => {
        const prompt: PromptRequest["prompt"] = [
            {type: "text", text: "look at this file:"},
            {type: "resource", resource: {uri: "file:///src/app.ts", text: "export const app=()=>{};"}},
            {type: "resource_link", name: "utils.ts", uri: "file:///src/utils.ts"}
        ];
        const result = textFromPrompt(prompt);
        expect(result).toContain("look at this file:");
        expect(result).toContain("file:///src/app.ts");
        expect(result).toContain("export const app=()=>{};");
        expect(result).toContain("utils.ts");
    });

    it("ignores unknown content types", () => {
        const prompt: PromptRequest["prompt"] = [
            {type: "text", text: "visible"},
            {type: "image", data: "base64data", mimeType: "image/png"}
        ];
        expect(textFromPrompt(prompt)).toBe("visible");
    });

    it("handles empty prompt", () => {
        expect(textFromPrompt([])).toBe("");
    });
});

describe("renamed project identity", () => {
    it("uses ollama-acp as the agent name", () => {
        expect(AGENT_NAME).toBe("ollama-acp");
    });

    it("stores config under the .ollama-acp directory", () => {
        expect(CONFIG_DIR_NAME).toBe(".ollama-acp");
    });

    it("does not reference the old project name", () => {
        expect(AGENT_NAME).not.toContain("intellij");
        expect(CONFIG_DIR_NAME).not.toContain("intellij");
    });
});

describe("OllamaClient context size", () => {
    it("defaults to 32768", () => {
        const c = new OllamaClient();
        expect(c.getNumCtx()).toBe(32768);
    });

    it("accepts custom context size via constructor", () => {
        const c = new OllamaClient(undefined, undefined, undefined, 16384);
        expect(c.getNumCtx()).toBe(16384);
    });

    it("updates via setNumCtx", () => {
        const c = new OllamaClient();
        c.setNumCtx(65536);
        expect(c.getNumCtx()).toBe(65536);
    });

    it("setNumCtx does not affect other fields", () => {
        const c = new OllamaClient("http://localhost:8080", "mymodel", false, 8192);
        c.setNumCtx(131072);
        expect(c.getNumCtx()).toBe(131072);
        expect(c.getBaseUrl()).toBe("http://localhost:8080");
        expect(c.getModel()).toBe("mymodel");
        expect(c.isThinking()).toBe(false);
    });
});

const tmpHome = () => mkdtempSync(join(tmpdir(), "acp-test-"));

describe("ACP", () => {
    it("answers initialize", async () => {
        const p = spawn(process.execPath, ["dist/index.js"], {
            env: {...process.env, HOME: tmpHome()},
            stdio: ["pipe", "pipe", "pipe"]
        });
        const x: any = await new Promise((resolve, reject) => {
            let b = "";
            const timer = setTimeout(() => reject(new Error("timeout")), 3000);
            p.stdout.on("data", d => {
                b += d.toString();
                for (const line of b.split("\n").slice(0, -1)) {
                    try {
                        const j = JSON.parse(line);
                        if (j.id === 1) {
                            clearTimeout(timer);
                            p.kill();
                            resolve(j);
                            return;
                        }
                    } catch {
                    }
                }
                b = b.split("\n").pop() ?? "";
            });
            p.stdin.write(JSON.stringify({
                jsonrpc: "2.0",
                id: 1,
                method: "initialize",
                params: {protocolVersion: 1, clientCapabilities: {}, clientInfo: {name: "test", version: "1"}}
            }) + "\n");
        });
        expect(x.result).toBeTruthy();
        expect(x.result.protocolVersion).toBe(1);
        expect(x.result.agentCapabilities).toBeTruthy();
    });

    it("declares a terminal auth method for the registry", async () => {
        const p = spawn(process.execPath, ["dist/index.js"], {
            env: {...process.env, HOME: tmpHome()},
            stdio: ["pipe", "pipe", "pipe"]
        });
        const x: any = await new Promise((resolve, reject) => {
            let b = "";
            const timer = setTimeout(() => reject(new Error("timeout")), 3000);
            p.stdout.on("data", d => {
                b += d.toString();
                for (const line of b.split("\n").slice(0, -1)) {
                    try {
                        const j = JSON.parse(line);
                        if (j.id === 1) {
                            clearTimeout(timer);
                            p.kill();
                            resolve(j);
                            return;
                        }
                    } catch {
                    }
                }
                b = b.split("\n").pop() ?? "";
            });
            p.stdin.write(JSON.stringify({
                jsonrpc: "2.0",
                id: 1,
                method: "initialize",
                params: {
                    protocolVersion: 1,
                    clientCapabilities: {auth: {terminal: true}},
                    clientInfo: {name: "test", version: "1"}
                }
            }) + "\n");
        });
        expect(Array.isArray(x.result.authMethods)).toBe(true);
        expect(x.result.authMethods.length).toBeGreaterThan(0);
        expect(x.result.authMethods[0].type).toBe("terminal");
        expect(x.result.authMethods[0].args).toContain("--setup");
    });

    it("runs the setup flow when invoked with --setup", async () => {
        const p = spawn(process.execPath, ["dist/index.js", "--setup"], {
            env: {...process.env, HOME: tmpHome()},
            stdio: ["pipe", "pipe", "pipe"]
        });
        p.stdin.end("http://127.0.0.1:11434\n\n");
        const exitCode: number = await new Promise(resolve => {
            p.on("exit", code => resolve(code ?? -1));
            setTimeout(() => {
                p.kill();
                resolve(-999);
            }, 3000);
        });
        expect(exitCode).toBe(0);
    });
});

function waitForInitialize(p: any): Promise<any> {
    return new Promise((resolve, reject) => {
        let b = "";
        const timer = setTimeout(() => reject(new Error("timeout waiting for initialize")), 5000);
        p.stdout.on("data", (d: string) => {
            b += d.toString();
            for (const line of b.split("\n").slice(0, -1)) {
                try {
                    const j = JSON.parse(line);
                    if (j.id === 1) {
                        clearTimeout(timer);
                        resolve(j);
                        return;
                    }
                } catch {
                }
            }
            b = b.split("\n").pop() ?? "";
        });
        p.stdin.write(JSON.stringify({
            jsonrpc: "2.0",
            id: 1,
            method: "initialize",
            params: {protocolVersion: 1, clientCapabilities: {}, clientInfo: {name: "test", version: "1"}}
        }) + "\n");
    });
}

function waitForExit(p: any): Promise<number | null> {
    return new Promise(resolve => {
        p.on("exit", code => resolve(code));
        setTimeout(() => {
            if (p.exitCode === null) {
                p.kill("SIGKILL");
                resolve(-999);
            }
        }, 8000);
    });
}

function runStatus(home: string): Promise<string> {
    return new Promise((resolve, reject) => {
        const p = spawn(process.execPath, ["dist/index.js", "--status"], {
            env: {...process.env, HOME: home},
            stdio: ["pipe", "pipe", "pipe"]
        });
        let out = "";
        const timer = setTimeout(() => reject(new Error("timeout waiting for --status")), 5000);
        p.stdout.on("data", (d: string) => {
            out += d.toString();
        });
        p.on("exit", code => {
            clearTimeout(timer);
            if (code === 0) resolve(out); else reject(new Error(`--status exited ${code}: ${out}`));
        });
    });
}

describe("process lifecycle and termination detection", () => {
    it("--status reports nothing recorded on a fresh config", async () => {
        const home = mkdtempSync(join(tmpdir(), "acp-life-"));
        const out = await runStatus(home);
        expect(out).toContain("No agent instance has been recorded yet.");
        expect(out).toContain(CONFIG_DIR_NAME);
    });

    it("records an unclean state when the process is killed with SIGKILL", async () => {
        const home = mkdtempSync(join(tmpdir(), "acp-life-"));
        const p = spawn(process.execPath, ["dist/index.js"], {
            env: {...process.env, HOME: home},
            stdio: ["pipe", "pipe", "pipe"]
        });
        await waitForInitialize(p);
        await new Promise<void>(resolve => setTimeout(resolve, 100));
        p.kill("SIGKILL");
        const killed = await waitForExit(p);
        expect(killed).not.toBe(0);

        const runtime = JSON.parse(readFileSync(join(home, ".ollama-acp", "runtime.json"), "utf-8"));
        expect(runtime.lastStatus).toBe("running");
        expect(runtime.pid).toBe(p.pid);

        const out = await runStatus(home);
        expect(out).toContain("not running");
        expect(out).toContain("killed/crashed without a clean shutdown");
    });

    it("marks the exit unclean when terminated by SIGTERM", async () => {
        const home = mkdtempSync(join(tmpdir(), "acp-life-"));
        const p = spawn(process.execPath, ["dist/index.js"], {
            env: {...process.env, HOME: home},
            stdio: ["pipe", "pipe", "pipe"]
        });
        await waitForInitialize(p);
        p.kill("SIGTERM");
        await waitForExit(p);
        const runtime = JSON.parse(readFileSync(join(home, ".ollama-acp", "runtime.json"), "utf-8"));
        expect(runtime.lastStatus).toBe("unclean");
        expect(runtime.reason).toContain("SIGTERM");
        expect(runtime.pid).toBe(p.pid);
        expect(existsSync(join(home, ".ollama-acp", "runtime.json"))).toBe(true);
    });

    it("warns on the next start that the previous run was killed or crashed", async () => {
        const home = mkdtempSync(join(tmpdir(), "acp-life-"));
        const first = spawn(process.execPath, ["dist/index.js"], {
            env: {...process.env, HOME: home},
            stdio: ["pipe", "pipe", "pipe"]
        });
        await waitForInitialize(first);
        first.kill("SIGKILL");
        await waitForExit(first);

        let stderr = "";
        const second = spawn(process.execPath, ["dist/index.js"], {
            env: {...process.env, HOME: home},
            stdio: ["pipe", "pipe", "pipe"]
        });
        second.stderr.on("data", (d: string) => {
            stderr += d.toString();
        });
        await waitForInitialize(second);
        second.kill("SIGKILL");
        await waitForExit(second);

        expect(stderr).toContain("Previous agent process");
        expect(stderr).toContain("killed or crashed");
    });
});

describe("OllamaClient auth", () => {
    it("reads an api key from the constructor", () => {
        const c = new OllamaClient("http://localhost:11434", undefined, undefined, undefined, "sk-test");
        expect(c.getApiKey()).toBe("sk-test");
    });

    it("returns undefined api key when none is set", () => {
        const c = new OllamaClient();
        expect(c.getApiKey()).toBeUndefined();
    });

    it("setApiKey updates the stored key", () => {
        const c = new OllamaClient();
        c.setApiKey("sk-updated");
        expect(c.getApiKey()).toBe("sk-updated");
        c.setApiKey(undefined);
        expect(c.getApiKey()).toBeUndefined();
    });

    it("sends the bearer token to all endpoints", async () => {
        const seen: Array<{ path: string; auth?: string }> = [];
        let server: Server | undefined;
        server = createServer((req, res) => {
            seen.push({path: req.url ?? "", auth: req.headers.authorization});
            res.setHeader("content-type", req.url === "/api/chat" ? "application/x-ndjson" : "application/json");
            res.end(req.url === "/api/tags" ? "[]" : req.url === "/api/show" ? '{"capabilities":[]}' : '{"message":{"role":"assistant","content":"hi"}}\n{"done":true}\n');
        });
        await new Promise<void>(resolve => server.listen(0, "127.0.0.1", resolve));
        const port = (server.address() as any).port;
        try {
            const c = new OllamaClient(`http://127.0.0.1:${port}`, undefined, undefined, undefined, "secret-token");
            await c.listModels();
            await c.getModelCapabilities();
            await c.chat([{role: "user", content: "x"}], [{
                type: "function",
                function: {name: "f", description: "", parameters: {}}
            }]);
        } finally {
            await new Promise<void>(resolve => server.close(() => resolve()));
        }
        expect(seen.length).toBe(3);
        for (const s of seen) {
            expect(s.auth).toBe("Bearer secret-token");
        }
    });

    it("omits the auth header when no api key is set", async () => {
        const seen: Array<string> = [];
        const server = createServer((req, res) => {
            seen.push(req.headers.authorization ?? "__none__");
            res.setHeader("content-type", "application/json");
            res.end("[]");
        });
        await new Promise<void>(resolve => server.listen(0, "127.0.0.1", resolve));
        const port = (server.address() as any).port;
        try {
            const c = new OllamaClient(`http://127.0.0.1:${port}`);
            await c.listModels();
        } finally {
            await new Promise<void>(resolve => server.close(() => resolve()));
        }
        expect(seen).toEqual(["__none__"]);
    });

    it("does not send a bearer token for unauthenticated requests", () => {
        const c = new OllamaClient("http://localhost:11434", undefined, undefined, undefined, "");
        expect(c.getApiKey()).toBeUndefined();
    });

    it("surfaces a clear hint when a server rejects auth", async () => {
        const server = createServer((req, res) => {
            res.statusCode = 401;
            res.end('{"error":"unauthorized"}');
        });
        await new Promise<void>(resolve => server.listen(0, "127.0.0.1", resolve));
        const port = (server.address() as any).port;
        try {
            const c = new OllamaClient(`http://127.0.0.1:${port}`, undefined, undefined, undefined, "bad-key");
            await expect(c.chat([{role: "user", content: "x"}], [])).rejects.toThrow(/authentication|401/i);
        } finally {
            await new Promise<void>(resolve => server.close(() => resolve()));
        }
    });

    it("does not add an auth header with no api key even when challenged", async () => {
        const seen: Array<string> = [];
        const server = createServer((req, res) => {
            seen.push(req.headers.authorization ?? "__none__");
            res.statusCode = 401;
            res.end("{}");
        });
        await new Promise<void>(resolve => server.listen(0, "127.0.0.1", resolve));
        const port = (server.address() as any).port;
        try {
            const c = new OllamaClient(`http://127.0.0.1:${port}`);
            await expect(c.listModels()).rejects.toThrow();
        } finally {
            await new Promise<void>(resolve => server.close(() => resolve()));
        }
        expect(seen).toEqual(["__none__"]);
    });
});

describe("OllamaClient streaming chat", () => {
    it("streams NDJSON chunks, invokes onChunk per chunk, and aggregates the full response", async () => {
        const chunks = [
            '{"message":{"role":"assistant","content":"Hello","thinking":"let me think"}}',
            '{"message":{"role":"assistant","content":" world"}}',
            '{"message":{"role":"assistant","tool_calls":[{"function":{"name":"read_file","arguments":{"path":"a"}}}]}}',
            '{"prompt_eval_count":1234,"done":true}'
        ];
        const server = createServer((req, res) => {
            res.setHeader("content-type", "application/x-ndjson");
            res.end(chunks.join("\n") + "\n");
        });
        await new Promise<void>(r => server.listen(0, "127.0.0.1", r));
        const port = (server.address() as any).port;
        try {
            const c = new OllamaClient(`http://127.0.0.1:${port}`);
            const seen: Array<{ content?: string; thinking?: string }> = [];
            const out = await c.chat([{role: "user", content: "x"}], [], chunk => {
                if (chunk.message?.content || chunk.message?.thinking) seen.push({
                    content: chunk.message.content,
                    thinking: chunk.message.thinking
                });
            });
            expect(out.message.content).toBe("Hello world");
            expect(out.message.thinking).toBe("let me think");
            expect(out.message.tool_calls?.[0]?.function.name).toBe("read_file");
            expect(out.prompt_eval_count).toBe(1234);
            expect(seen.length).toBe(2);
        } finally {
            await new Promise<void>(r => server.close(() => r()));
        }
    });

    it("aggregates content delivered across partial lines", async () => {
        const server = createServer((req, res) => {
            res.writeHead(200, {"content-type": "application/x-ndjson"});
            res.write('{"message":{"role":"assistant","content":"Hel');
            setTimeout(() => {
                res.write('lo, world"}}\n{"done":true}\n');
                res.end();
            }, 20);
        });
        await new Promise<void>(r => server.listen(0, "127.0.0.1", r));
        const port = (server.address() as any).port;
        try {
            const c = new OllamaClient(`http://127.0.0.1:${port}`);
            const out = await c.chat([{role: "user", content: "x"}], []);
            expect(out.message.content).toBe("Hello, world");
        } finally {
            await new Promise<void>(r => server.close(() => r()));
        }
    });

    it("merges tool calls that arrive split across chunks", async () => {
        const chunks = [
            '{"message":{"tool_calls":[{"function":{"name":"read_file","arguments":{"path":"a.ts"}}}]}}',
            '{"message":{"tool_calls":[{"function":{"name":"write_file","arguments":{"path":"b.ts","content":"hello"}}}]}}',
            '{"done":true}'
        ];
        const server = createServer((req, res) => {
            res.setHeader("content-type", "application/x-ndjson");
            res.end(chunks.join("\n") + "\n");
        });
        await new Promise<void>(r => server.listen(0, "127.0.0.1", r));
        const port = (server.address() as any).port;
        try {
            const c = new OllamaClient(`http://127.0.0.1:${port}`);
            const out = await c.chat([{role: "user", content: "x"}], []);
            const names = out.message.tool_calls?.map(tc => tc.function.name);
            expect(names).toContain("read_file");
            expect(names).toContain("write_file");
        } finally {
            await new Promise<void>(r => server.close(() => r()));
        }
    });

    it("fills in previously partial arguments for the same tool call", async () => {
        const chunks = [
            '{"message":{"tool_calls":[{"function":{"name":"write_file","arguments":{}}}]}}',
            '{"message":{"tool_calls":[{"function":{"name":"write_file","arguments":{"path":"b.ts","content":"hello"}}}]}}',
            '{"done":true}'
        ];
        const server = createServer((req, res) => {
            res.setHeader("content-type", "application/x-ndjson");
            res.end(chunks.join("\n") + "\n");
        });
        await new Promise<void>(r => server.listen(0, "127.0.0.1", r));
        const port = (server.address() as any).port;
        try {
            const c = new OllamaClient(`http://127.0.0.1:${port}`);
            const out = await c.chat([{role: "user", content: "x"}], []);
            const call = out.message.tool_calls?.[0];
            expect(call?.function.name).toBe("write_file");
            expect(call?.function.arguments).toEqual({path: "b.ts", content: "hello"});
        } finally {
            await new Promise<void>(r => server.close(() => r()));
        }
    });

    it("rejects with AbortError when aborted mid-stream", async () => {
        const controller = new AbortController();
        const server = createServer((req, res) => {
            res.writeHead(200, {"content-type": "application/x-ndjson"});
            res.write('{"message":{"role":"assistant","content":"first"}}\n');
            setTimeout(() => res.end('{"message":{"role":"assistant","content":"second"}}\n{"done":true}\n'), 500);
        });
        await new Promise<void>(r => server.listen(0, "127.0.0.1", r));
        const port = (server.address() as any).port;
        try {
            const c = new OllamaClient(`http://127.0.0.1:${port}`);
            let got = 0;
            const p = c.chat([{role: "user", content: "x"}], [], () => {
                got++;
                if (got === 1) controller.abort();
            }, controller.signal);
            await expect(p).rejects.toSatisfy((e: any) => e?.name === "AbortError");
        } finally {
            await new Promise<void>(r => server.close(() => r()));
        }
    });

    it("falls back to non-thinking mode when the server rejects the think flag", async () => {
        let calls = 0;
        const server = createServer((req, res) => {
            res.setHeader("content-type", "application/x-ndjson");
            calls++;
            if (calls === 1) {
                res.statusCode = 400;
                res.end('{"error":"model does not support thinking"}');
            } else {
                res.end('{"message":{"role":"assistant","content":"ok without thinking"}}\n{"done":true}\n');
            }
        });
        await new Promise<void>(r => server.listen(0, "127.0.0.1", r));
        const port = (server.address() as any).port;
        try {
            const c = new OllamaClient(`http://127.0.0.1:${port}`, undefined, true);
            const out = await c.chat([{role: "user", content: "x"}], []);
            expect(out.message.content).toBe("ok without thinking");
            expect(c.isThinking()).toBe(false);
            expect(calls).toBe(2);
        } finally {
            await new Promise<void>(r => server.close(() => r()));
        }
    });
});

describe("OllamaClient connection health", () => {
    it("is reachable by default and reports a clean probe", async () => {
        const server = createServer((req, res) => {
            res.setHeader("content-type", "application/json");
            res.end('{"models":[{"name":"qwen3-coder"}]}');
        });
        await new Promise<void>(r => server.listen(0, "127.0.0.1", r));
        const port = (server.address() as any).port;
        try {
            const c = new OllamaClient(`http://127.0.0.1:${port}`);
            expect(c.isReachable()).toBe(true);
            expect(await c.probe(500)).toBe(true);
            expect(c.isReachable()).toBe(true);
            expect(c.getHealth().url).toBe(`http://127.0.0.1:${port}`);
            expect(c.getHealth().error).toBeUndefined();
        } finally {
            await new Promise<void>(r => server.close(() => r()));
        }
    });

    it("marks the server unreachable when a probe fails", async () => {
        const c = new OllamaClient("http://127.0.0.1:1");
        expect(await c.probe(500)).toBe(false);
        expect(c.isReachable()).toBe(false);
        expect(c.getHealth().error).toBeTruthy();
    });

    it("returns the last known model list while the server is unreachable", async () => {
        let server: Server | undefined;
        server = createServer((req, res) => {
            res.setHeader("content-type", "application/json");
            res.end('{"models":[{"name":"qwen3-coder"},{"name":"llama3"}]}');
        });
        await new Promise<void>(r => server.listen(0, "127.0.0.1", r));
        const port = (server.address() as any).port;
        const c = new OllamaClient(`http://127.0.0.1:${port}`);
        try {
            expect(await c.listModels()).toEqual(["qwen3-coder", "llama3"]);
        } finally {
            await new Promise<void>(r => server.close(() => r()));
        }
        expect(await c.listModels()).toEqual(["qwen3-coder", "llama3"]);
        expect(c.isReachable()).toBe(false);
    });

    it("recovers once the server comes back online", async () => {
        let up = false;
        const server = createServer((req, res) => {
            if (!up) {
                res.destroy();
                return;
            }
            res.setHeader("content-type", "application/json");
            res.end('{"models":[{"name":"qwen3-coder"}]}');
        });
        await new Promise<void>(r => server.listen(0, "127.0.0.1", r));
        const port = (server.address() as any).port;
        try {
            const c = new OllamaClient(`http://127.0.0.1:${port}`);
            expect(await c.probe(500)).toBe(false);
            expect(c.isReachable()).toBe(false);
            up = true;
            expect(await c.probe(500)).toBe(true);
            expect(c.isReachable()).toBe(true);
            expect(c.getHealth().error).toBeUndefined();
        } finally {
            await new Promise<void>(r => server.close(() => r()));
        }
    });

    it("summarize sends a non-streaming non-thinking request and returns content", async () => {
        let captured: any;
        const server = createServer((req, res) => {
            const chunks: Buffer[] = [];
            req.on("data", (c: Buffer) => chunks.push(c));
            req.on("end", () => {
                captured = JSON.parse(Buffer.concat(chunks).toString());
                res.setHeader("content-type", "application/json");
                res.end(JSON.stringify({message: {role: "assistant", content: "folded context summary"}}));
            });
        });
        await new Promise<void>(r => server.listen(0, "127.0.0.1", r));
        const port = (server.address() as any).port;
        try {
            const c = new OllamaClient(`http://127.0.0.1:${port}`);
            const out = await c.summarize([{role: "user", content: "summarize this"}]);
            expect(out).toBe("folded context summary");
            expect(captured.stream).toBe(false);
            expect(captured.think).toBe(false);
            expect(c.isReachable()).toBe(true);
        } finally {
            await new Promise<void>(r => server.close(() => r()));
        }
    });

    it("summarize marks the server unreachable when the request fails", async () => {
        const server = createServer((req, res) => {
            res.destroy();
        });
        await new Promise<void>(r => server.listen(0, "127.0.0.1", r));
        const port = (server.address() as any).port;
        try {
            const c = new OllamaClient(`http://127.0.0.1:${port}`);
            await expect(c.summarize([{role: "user", content: "x"}])).rejects.toThrow();
            expect(c.isReachable()).toBe(false);
            expect(c.getHealth().error).toBeTruthy();
        } finally {
            await new Promise<void>(r => server.close(() => r()));
        }
    });
});

describe("context compression", () => {
    const sys = {role: "system" as const, content: "system"};
    const user = (t: string) => ({role: "user" as const, content: t});
    const asst = (t: string) => ({role: "assistant" as const, content: t});

    it("estimates tokens from character length including tool arguments", () => {
        const msgs = [user("abcdefghijklmnopqrstuvwxyz")];
        expect(estimateMessagesTokens(msgs)).toBe(7);
        const withTools = [{
            role: "assistant" as const,
            content: "hi",
            tool_calls: [{function: {name: "read_file", arguments: {path: "/a/b/c.ts"}}}]
        }];
        expect(estimateMessagesTokens(withTools)).toBeGreaterThan(1);
        expect(estimateMessagesTokens([])).toBe(0);
    });

    it("does not compress short sessions or sessions under the threshold", () => {
        const short = [sys, user("a"), asst("b")];
        expect(shouldCompressMessages(short, 16384)).toBe(false);
        const many = [sys, ...Array.from({length: 10}, (_, i) => i % 2 ? asst(`a${i}`) : user(`u${i}`))];
        expect(shouldCompressMessages(many, 16384)).toBe(false);
        expect(shouldCompressMessages(many, 16384, 1)).toBe(false);
    });

    it("triggers compression when prompt tokens exceed the threshold", () => {
        const msgs = [sys, ...Array.from({length: 20}, (_, i) => i % 2 ? asst(`a${i}`) : user(`u${i}`))];
        expect(shouldCompressMessages(msgs, 16384, 14000)).toBe(true);
        expect(shouldCompressMessages(msgs, 16384, 13000)).toBe(false);
    });

    it("triggers compression from an estimated size when no token count is known", () => {
        const big = [sys, ...Array.from({length: 20}, (_, i) => i % 2 ? asst("x".repeat(2000)) : user("y".repeat(2000)))];
        expect(shouldCompressMessages(big, 4096)).toBe(true);
    });

    it("replaces older messages with a summary, keeping system and the recent tail", () => {
        const msgs = [sys];
        for (let i = 0; i < 20; i++) msgs.push(i % 2 ? asst(`a${i}`) : user(`u${i}`));
        const out = compressMessages(msgs, "FOLDED", 8);
        expect(out[0].role).toBe("system");
        expect(out.filter(m => String(m.content).startsWith("[Summary of earlier conversation]"))).toHaveLength(1);
        expect(out[out.length - 1]).toEqual(msgs[msgs.length - 1]);
        expect(out.length).toBeLessThan(msgs.length);
    });

    it("returns the messages unchanged for an empty summary", () => {
        const msgs = [sys, user("u0"), asst("a0")];
        expect(compressMessages(msgs, "   ")).toBe(msgs);
    });
});

describe("file change detection", () => {
    it("snapshots all files recursively and skips node_modules", () => {
        const root = mkdtempSync(join(tmpdir(), "acp-chg-"));
        mkdirSync(join(root, "a", "b"), {recursive: true});
        mkdirSync(join(root, "node_modules"), {recursive: true});
        writeFileSync(join(root, "top.txt"), "top");
        writeFileSync(join(root, "a", "one.ts"), "x");
        writeFileSync(join(root, "a", "b", "two.ts"), "y");
        writeFileSync(join(root, "node_modules", "pkg.txt"), "ignored");
        const snap = snapshotFiles(root);
        expect(snap.has(join(root, "top.txt"))).toBe(true);
        expect(snap.has(join(root, "a", "one.ts"))).toBe(true);
        expect(snap.has(join(root, "a", "b", "two.ts"))).toBe(true);
        expect(snap.has(join(root, "node_modules", "pkg.txt"))).toBe(false);
    });

    it("detects created, modified, and deleted files", async () => {
        const root = mkdtempSync(join(tmpdir(), "acp-chg-"));
        writeFileSync(join(root, "keep.txt"), "a");
        writeFileSync(join(root, "edit.ts"), "old");
        writeFileSync(join(root, "gone.ts"), "bye");
        const before = snapshotFiles(root);
        await new Promise(r => setTimeout(r, 20));
        appendFileSync(join(root, "edit.ts"), "newcontent");
        writeFileSync(join(root, "new.ts"), "fresh");
        rmSync(join(root, "gone.ts"));
        const changes = diffSnapshots(before, snapshotFiles(root));
        const byPath = new Map(changes.map(c => [c.path, c.change]));
        expect(byPath.get(join(root, "edit.ts"))).toBe("modified");
        expect(byPath.get(join(root, "new.ts"))).toBe("created");
        expect(byPath.get(join(root, "gone.ts"))).toBe("deleted");
        expect(byPath.get(join(root, "keep.txt"))).toBeUndefined();
    });

    it("reports no changes when nothing changed", () => {
        const root = mkdtempSync(join(tmpdir(), "acp-chg-"));
        writeFileSync(join(root, "f.txt"), "same");
        const before = snapshotFiles(root);
        expect(diffSnapshots(before, snapshotFiles(root))).toHaveLength(0);
    });

    it("merges tracked writes with snapshot changes and prefers created over modified", () => {
        const root = mkdtempSync(join(tmpdir(), "acp-chg-"));
        writeFileSync(join(root, "a.ts"), "x");
        const snapChanges = [
            {path: join(root, "a.ts"), change: "modified" as const},
            {path: join(root, "b.ts"), change: "modified" as const}
        ];
        const tracker = new Map<string, { path: string; kind: "created" | "modified" }>();
        tracker.set(join(root, "a.ts"), {path: join(root, "a.ts"), kind: "created"});
        tracker.set(join(root, "out.txt"), {path: join(root, "out.txt"), kind: "created"});
        const merged = mergeFileChanges(snapChanges, tracker);
        const byPath = new Map(merged.map(c => [c.path, c.change]));
        expect(byPath.get(join(root, "a.ts"))).toBe("created");
        expect(byPath.get(join(root, "b.ts"))).toBe("modified");
        expect(byPath.get(join(root, "out.txt"))).toBe("created");
    });
});