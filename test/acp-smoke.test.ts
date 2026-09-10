import {describe,it,expect} from "vitest";
import {spawn} from "node:child_process";
import {createServer, type Server} from "node:http";
import type {PromptRequest} from "@agentclientprotocol/sdk";
import {textFromPrompt,AGENT_NAME,CONFIG_DIR_NAME} from "../src/index.js";
import {OllamaClient} from "../src/ollama.js";

describe("textFromPrompt",()=>{
  it("extracts plain text",()=>{
    const prompt:PromptRequest["prompt"]=[{type:"text",text:"hello world"}];
    expect(textFromPrompt(prompt)).toBe("hello world");
  });

  it("extracts multiple text blocks",()=>{
    const prompt:PromptRequest["prompt"]=[
      {type:"text",text:"first line"},
      {type:"text",text:"second line"}
    ];
    expect(textFromPrompt(prompt)).toBe("first line\nsecond line");
  });

  it("extracts embedded resource (file content)",()=>{
    const prompt:PromptRequest["prompt"]=[{
      type:"resource",
      resource:{uri:"file:///path/to/file.ts",text:"const x=1;"}
    }];
    const result=textFromPrompt(prompt);
    expect(result).toContain("file:///path/to/file.ts");
    expect(result).toContain("const x=1;");
  });

  it("extracts resource_link",()=>{
    const prompt:PromptRequest["prompt"]=[{
      type:"resource_link",
      name:"index.ts",
      uri:"file:///src/index.ts"
    }];
    const result=textFromPrompt(prompt);
    expect(result).toContain("index.ts");
    expect(result).toContain("file:///src/index.ts");
  });

  it("handles mixed content types",()=>{
    const prompt:PromptRequest["prompt"]=[
      {type:"text",text:"look at this file:"},
      {type:"resource",resource:{uri:"file:///src/app.ts",text:"export const app=()=>{};"}},
      {type:"resource_link",name:"utils.ts",uri:"file:///src/utils.ts"}
    ];
    const result=textFromPrompt(prompt);
    expect(result).toContain("look at this file:");
    expect(result).toContain("file:///src/app.ts");
    expect(result).toContain("export const app=()=>{};");
    expect(result).toContain("utils.ts");
  });

  it("ignores unknown content types",()=>{
    const prompt:PromptRequest["prompt"]=[
      {type:"text",text:"visible"},
      {type:"image",data:"base64data",mimeType:"image/png"}
    ];
    expect(textFromPrompt(prompt)).toBe("visible");
  });

  it("handles empty prompt",()=>{
    expect(textFromPrompt([])).toBe("");
  });
});

describe("renamed project identity",()=>{
  it("uses ollama-acp as the agent name",()=>{
    expect(AGENT_NAME).toBe("ollama-acp");
  });

  it("stores config under the .ollama-acp directory",()=>{
    expect(CONFIG_DIR_NAME).toBe(".ollama-acp");
  });

  it("does not reference the old project name",()=>{
    expect(AGENT_NAME).not.toContain("intellij");
    expect(CONFIG_DIR_NAME).not.toContain("intellij");
  });
});

describe("OllamaClient context size",()=>{
  it("defaults to 32768",()=>{
    const c=new OllamaClient();
    expect(c.getNumCtx()).toBe(32768);
  });

  it("accepts custom context size via constructor",()=>{
    const c=new OllamaClient(undefined,undefined,undefined,16384);
    expect(c.getNumCtx()).toBe(16384);
  });

  it("updates via setNumCtx",()=>{
    const c=new OllamaClient();
    c.setNumCtx(65536);
    expect(c.getNumCtx()).toBe(65536);
  });

  it("setNumCtx does not affect other fields",()=>{
    const c=new OllamaClient("http://localhost:8080","mymodel",false,8192);
    c.setNumCtx(131072);
    expect(c.getNumCtx()).toBe(131072);
    expect(c.getBaseUrl()).toBe("http://localhost:8080");
    expect(c.getModel()).toBe("mymodel");
    expect(c.isThinking()).toBe(false);
  });
});

describe("ACP",()=>{
  it("answers initialize",async()=>{
    const p=spawn(process.execPath,["dist/index.js"],{stdio:["pipe","pipe","pipe"]});
    const x:any=await new Promise((resolve,reject)=>{
      let b=""; const timer=setTimeout(()=>reject(new Error("timeout")),3000);
      p.stdout.on("data",d=>{
        b+=d.toString();
        for(const line of b.split("\n").slice(0,-1)){try{
          const j=JSON.parse(line); if(j.id===1){clearTimeout(timer);p.kill();resolve(j);return;}
        }catch{}}
        b=b.split("\n").pop()??"";
      });
      p.stdin.write(JSON.stringify({jsonrpc:"2.0",id:1,method:"initialize",params:{protocolVersion:1,clientCapabilities:{},clientInfo:{name:"test",version:"1"}}})+"\n");
    });
    expect(x.result).toBeTruthy();
    expect(x.result.protocolVersion).toBe(1);
    expect(x.result.agentCapabilities).toBeTruthy();
  });

  it("declares a terminal auth method for the registry",async()=>{
    const p=spawn(process.execPath,["dist/index.js"],{stdio:["pipe","pipe","pipe"]});
    const x:any=await new Promise((resolve,reject)=>{
      let b=""; const timer=setTimeout(()=>reject(new Error("timeout")),3000);
      p.stdout.on("data",d=>{
        b+=d.toString();
        for(const line of b.split("\n").slice(0,-1)){try{
          const j=JSON.parse(line); if(j.id===1){clearTimeout(timer);p.kill();resolve(j);return;}
        }catch{}}
        b=b.split("\n").pop()??"";
      });
      p.stdin.write(JSON.stringify({jsonrpc:"2.0",id:1,method:"initialize",params:{protocolVersion:1,clientCapabilities:{auth:{terminal:true}},clientInfo:{name:"test",version:"1"}}})+"\n");
    });
    expect(Array.isArray(x.result.authMethods)).toBe(true);
    expect(x.result.authMethods.length).toBeGreaterThan(0);
    expect(x.result.authMethods[0].type).toBe("terminal");
    expect(x.result.authMethods[0].args).toContain("--setup");
  });

  it("runs the setup flow when invoked with --setup",async()=>{
    const p=spawn(process.execPath,["dist/index.js","--setup"],{stdio:["pipe","pipe","pipe"]});
    p.stdin.end("http://127.0.0.1:11434\n\n");
    const exitCode:number=await new Promise(resolve=>{
      p.on("exit",code=>resolve(code??-1));
      setTimeout(()=>{p.kill();resolve(-999);},3000);
    });
    expect(exitCode).toBe(0);
  });
});

describe("OllamaClient auth",()=>{
  it("reads an api key from the constructor",()=>{
    const c=new OllamaClient("http://localhost:11434",undefined,undefined,undefined,"sk-test");
    expect(c.getApiKey()).toBe("sk-test");
  });

  it("returns undefined api key when none is set",()=>{
    const c=new OllamaClient();
    expect(c.getApiKey()).toBeUndefined();
  });

  it("setApiKey updates the stored key",()=>{
    const c=new OllamaClient();
    c.setApiKey("sk-updated");
    expect(c.getApiKey()).toBe("sk-updated");
    c.setApiKey(undefined);
    expect(c.getApiKey()).toBeUndefined();
  });

  it("sends the bearer token to all endpoints",async()=>{
    const seen:Array<{path:string;auth?:string}>=[]; let server:Server|undefined;
    server=createServer((req,res)=>{
      seen.push({path:req.url??"",auth:req.headers.authorization});
      res.setHeader("content-type","application/json");
      res.end(req.url==="/api/tags"?"[]":'{"message":{"role":"assistant","content":"hi"}}');
    });
    await new Promise<void>(resolve=>server.listen(0,"127.0.0.1",resolve));
    const port=(server.address() as any).port;
    try{
      const c=new OllamaClient(`http://127.0.0.1:${port}`,undefined,undefined,undefined,"secret-token");
      await c.listModels();
      await c.getModelCapabilities();
      await c.chat([{role:"user",content:"x"}],[{type:"function",function:{name:"f",description:"",parameters:{}}}]);
    }finally{
      await new Promise<void>(resolve=>server.close(()=>resolve()));
    }
    expect(seen.length).toBe(3);
    for(const s of seen){
      expect(s.auth).toBe("Bearer secret-token");
    }
  });

  it("omits the auth header when no api key is set",async()=>{
    const seen:Array<string>=[];
    const server=createServer((req,res)=>{
      seen.push(req.headers.authorization??"__none__");
      res.setHeader("content-type","application/json");
      res.end("[]");
    });
    await new Promise<void>(resolve=>server.listen(0,"127.0.0.1",resolve));
    const port=(server.address() as any).port;
    try{
      const c=new OllamaClient(`http://127.0.0.1:${port}`);
      await c.listModels();
    }finally{
      await new Promise<void>(resolve=>server.close(()=>resolve()));
    }
    expect(seen).toEqual(["__none__"]);
  });

  it("does not send a bearer token for unauthenticated requests",()=>{
    const c=new OllamaClient("http://localhost:11434",undefined,undefined,undefined,"");
    expect(c.getApiKey()).toBeUndefined();
  });
});