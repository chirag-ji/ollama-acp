import {describe,it,expect} from "vitest";
import {spawn} from "node:child_process";
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
});