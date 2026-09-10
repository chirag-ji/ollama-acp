export type OllamaToolCall = { function: { name: string; arguments: Record<string, unknown> } };

export type OllamaMessage = {
  role: "system" | "user" | "assistant" | "tool";
  content?: string;
  tool_calls?: OllamaToolCall[];
  tool_name?: string;
};

export type OllamaTool = {
  type: "function";
  function: {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  };
};

export type OllamaChunk = {
  message?: {
    role?: string;
    content?: string;
    thinking?: string;
    tool_calls?: OllamaToolCall[];
  };
  done?: boolean;
  done_reason?: string;
};

export type OllamaResponse = {
  message: {
    role: "assistant";
    content: string;
    thinking?: string;
    tool_calls?: OllamaToolCall[];
  };
  done: boolean;
};

function log(...args: unknown[]): void {
  const msg = args.map(a => (typeof a === "string" ? a : JSON.stringify(a))).join(" ");
  process.stderr.write(`[ollama] ${msg}\n`);
}

function normalizeHost(host?: string): string | undefined {
  if (!host) return undefined;
  const trimmed = host.trim();
  if (!trimmed) return undefined;
  return /^https?:\/\//i.test(trimmed) ? trimmed : `http://${trimmed}`;
}

export type ModelCapabilities = {
  capabilities: string[];
  template?: string;
  details?: Record<string, unknown>;
};

function authErrorHint(status: number): string {
    if (status === 401 || status === 403) {
        return "Server rejected authentication. Run 'ollama-acp --setup' to configure an API key, or unset OLLAMA_API_KEY for a local Ollama server that requires no auth.";
    }
    return "";
}

export class OllamaClient {
  private baseUrl: string;
  private model: string;
  private thinking: boolean;
  private numCtx: number;
  private apiKey?: string;
  private cachedCapabilities: Map<string, ModelCapabilities> = new Map();

  constructor(
    baseUrl?: string,
    model?: string,
    thinking?: boolean,
    numCtx?: number,
    apiKey?: string,
  ) {
    this.baseUrl = baseUrl ?? process.env.OLLAMA_BASE_URL ?? process.env.OLLAMA_URL ?? normalizeHost(process.env.OLLAMA_HOST) ?? "http://127.0.0.1:11434";
    this.model = model ?? process.env.OLLAMA_MODEL ?? "qwen3-coder";
    this.thinking = thinking ?? process.env.OLLAMA_THINK !== "false";
    this.numCtx = numCtx ?? 32768;
    this.apiKey = (apiKey ?? process.env.OLLAMA_API_KEY)?.trim() || undefined;
  }

  getApiKey(): string | undefined { return this.apiKey; }

  setApiKey(apiKey: string | undefined): void { this.apiKey = apiKey; }

  private headers(): Record<string, string> {
    const h: Record<string, string> = {"content-type": "application/json"};
    if (this.apiKey) h["authorization"] = `Bearer ${this.apiKey}`;
    return h;
  }

  getModel(): string { return this.model; }

  getBaseUrl(): string { return this.baseUrl; }

  isThinking(): boolean { return this.thinking; }

  getNumCtx(): number { return this.numCtx; }

  setBaseUrl(url: string): void { this.baseUrl = url; }

  setModel(model: string): void { this.model = model; }

  setThinking(thinking: boolean): void { this.thinking = thinking; }

  setNumCtx(numCtx: number): void { this.numCtx = numCtx; }

  async getModelCapabilities(model?: string): Promise<ModelCapabilities> {
    const target = model ?? this.model;
    const cached = this.cachedCapabilities.get(target);
    if (cached) return cached;

    const res = await fetch(`${this.baseUrl}/api/show`, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify({model: target})
    });
    if (!res.ok) throw new Error(`Ollama /api/show failed: ${res.status}. ${authErrorHint(res.status)}`);
    const json = await res.json() as {
      capabilities?: string[];
      template?: string;
      details?: Record<string, unknown>;
    };
    const result: ModelCapabilities = {
      capabilities: json.capabilities ?? [],
      template: json.template,
      details: json.details
    };
    this.cachedCapabilities.set(target, result);
    return result;
  }

  invalidateCapabilities(model?: string): void {
    const target = model ?? this.model;
    this.cachedCapabilities.delete(target);
  }

  async listModels(): Promise<string[]> {
    const res = await fetch(`${this.baseUrl}/api/tags`, {
      headers: this.headers()
    });
    if (!res.ok) throw new Error(`Ollama /api/tags failed: ${res.status}. ${authErrorHint(res.status)}`);
    const json = await res.json() as { models?: Array<{ name: string }> };
    const names = (json.models ?? []).map(m => m.name);
    log(`listModels: got ${names.length} models: ${JSON.stringify(names)}`);
    return names;
  }

  private async postChat(messages: OllamaMessage[], tools: OllamaTool[], think: boolean, signal?: AbortSignal): Promise<Response> {
    return fetch(`${this.baseUrl}/api/chat`, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify({
        model: this.model,
        messages,
        tools,
        stream: true,
        think,
        options: {num_ctx: this.numCtx}
      }),
      signal
    });
  }

  private async readChatStream(
    res: Response,
    onChunk?: (chunk: OllamaChunk) => void | Promise<void>,
    signal?: AbortSignal
  ): Promise<OllamaResponse> {
    if (!res.body) throw new Error("Ollama /api/chat returned no body");
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let content = "";
    let thinking = "";
    let toolCalls: OllamaToolCall[] = [];

    const processLine = async (line: string): Promise<void> => {
      const trimmed = line.trim();
      if (!trimmed) return;
      let chunk: OllamaChunk;
      try {
        chunk = JSON.parse(trimmed);
      } catch {
        return;
      }
      if (chunk.message?.content) content += chunk.message.content;
      if (chunk.message?.thinking) thinking += chunk.message.thinking;
      if (chunk.message?.tool_calls?.length) toolCalls = chunk.message.tool_calls;
      if (onChunk) {
        try {
          await onChunk(chunk);
        } catch (err) {
          log("onChunk error:", err);
        }
      }
    };

    while (true) {
      if (signal?.aborted) {
        await reader.cancel().catch(() => undefined);
        const err = new Error("Aborted");
        err.name = "AbortError";
        throw err;
      }
      const {done, value} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});
      let nl = buffer.indexOf("\n");
      while (nl !== -1) {
        const line = buffer.slice(0, nl);
        buffer = buffer.slice(nl + 1);
        await processLine(line);
        nl = buffer.indexOf("\n");
      }
    }

    return {
      message: {
        role: "assistant",
        content,
        thinking: thinking || undefined,
        tool_calls: toolCalls.length ? toolCalls : undefined
      },
      done: true
    };
  }

  async chat(
    messages: OllamaMessage[],
    tools: OllamaTool[],
    onChunk?: (chunk: OllamaChunk) => void | Promise<void>,
    signal?: AbortSignal
  ): Promise<OllamaResponse> {
    let res = await this.postChat(messages, tools, this.thinking, signal);
    if (!res.ok) {
      const body = await res.text();
      if (this.thinking && body.toLowerCase().includes("thinking")) {
        log("thinking not supported, retrying without think flag");
        this.thinking = false;
        res = await this.postChat(messages, tools, false, signal);
        if (!res.ok) {
          const retryBody = await res.text();
          throw new Error(`Ollama /api/chat failed: ${res.status} ${retryBody} ${authErrorHint(res.status)}`);
        }
      } else {
        throw new Error(`Ollama /api/chat failed: ${res.status} ${body} ${authErrorHint(res.status)}`);
      }
    }
    return this.readChatStream(res, onChunk, signal);
  }
}
