import { spawn } from "node:child_process";
import { OllamaClient } from "../src/ollama.js";

async function main() {
  console.log("=== Ollama ACP debug ===");
  const ollama = new OllamaClient();

  console.log("1. Ollama health...");
  const models = await ollama.listModels();
  console.log(`   PASS: ${models.length} model(s)`);

  console.log("2. Ollama chat...");
  const c:any = await ollama.chat([{role:"user",content:"Reply with exactly ACP_OK"}], []);
  console.log(`   PASS: ${(c.message?.content ?? "").slice(0,200)}`);

  console.log("3. ACP initialize + session/new...");
  const p = spawn(process.execPath, ["dist/index.js"], {stdio:["pipe","pipe","pipe"], env:process.env});
  let out="", err="";
  p.stdout.on("data", d => out += d.toString());
  p.stderr.on("data", d => err += d.toString());

  p.stdin.write(JSON.stringify({jsonrpc:"2.0",id:1,method:"initialize",params:{
    protocolVersion:1,clientCapabilities:{},clientInfo:{name:"debug-client",version:"1"}
  }})+"\n");
  p.stdin.write(JSON.stringify({jsonrpc:"2.0",id:2,method:"session/new",params:{
    cwd:process.cwd(),mcpServers:[]
  }})+"\n");

  await new Promise(r=>setTimeout(r,1200));
  p.kill("SIGTERM");

  const lines = out.split("\n").filter(Boolean);
  console.log(`   Responses: ${lines.length}`);
  for (const l of lines) console.log(`   ${l.slice(0,500)}`);
  console.log("   stderr:", err.trim() || "(none)");

  if (lines.length < 2) throw new Error("ACP did not return initialize + session/new responses");
  console.log("4. RESULT: PASS");
}
main().catch(e => { console.error("DEBUG FAILED:", e); process.exit(1); });