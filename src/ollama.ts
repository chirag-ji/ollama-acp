export type OllamaMessage = {
  role: "system" | "user" | "assistant" | "tool";
  content?: string;
  tool_calls?: Array<{ function: { name: string; arguments: Record<string, unknown> } }>;
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

export type OllamaResponse = {
  message: {
    role: "assistant";
    content: string;
    thinking?: string;
    tool_calls?: Array<{ function: { name: string; arguments: Record<string, unknown> } }>;
  };
  done: boolean;
};

function log(...args: unknown[]): void {
  const msg = args.map(a => (typeof a === "string" ? a : JSON.stringify(a))).join(" ");
  process.stderr.write(`[ollama] ${msg}\n`);
}

export class OllamaClient {
  private baseUrl: string;
  private model: string;
  private thinking: boolean;

  constructor(
    baseUrl?: string,
    model?: string,
    thinking?: boolean,
  ) {
    this.baseUrl = baseUrl ?? process.env.OLLAMA_URL ?? "http://127.0.0.1:11434";
    this.model = model ?? process.env.OLLAMA_MODEL ?? "qwen3-coder";
    this.thinking = thinking ?? process.env.OLLAMA_THINK !== "false";
  }

  getModel(): string { return this.model; }

  getBaseUrl(): string { return this.baseUrl; }

  isThinking(): boolean { return this.thinking; }

  setBaseUrl(url: string): void { this.baseUrl = url; }

  setModel(model: string): void { this.model = model; }

  setThinking(thinking: boolean): void { this.thinking = thinking; }

  async listModels(): Promise<string[]> {
    const res = await fetch(`${this.baseUrl}/api/tags`);
    if (!res.ok) throw new Error(`Ollama /api/tags failed: ${res.status}`);
    const json = await res.json() as { models?: Array<{ name: string }> };
    return (json.models ?? []).map(m => m.name);
  }

  async chat(messages: OllamaMessage[], tools: OllamaTool[]): Promise<OllamaResponse> {
    let res = await fetch(`${this.baseUrl}/api/chat`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: this.model,
        messages,
        tools,
        stream: false,
        think: this.thinking
      })
    });
    if (!res.ok) {
      const body = await res.text();
      if (this.thinking && body.toLowerCase().includes("thinking")) {
        log("thinking not supported, retrying without think flag");
        this.thinking = false;
        res = await fetch(`${this.baseUrl}/api/chat`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            model: this.model,
            messages,
            tools,
            stream: false,
            think: false
          })
        });
        if (!res.ok) {
          const retryBody = await res.text();
          throw new Error(`Ollama /api/chat failed: ${res.status} ${retryBody}`);
        }
      } else {
        throw new Error(`Ollama /api/chat failed: ${res.status} ${body}`);
      }
    }
    return await res.json() as OllamaResponse;
  }
}
