// Use the pinned official demo unchanged; drive its local-file loading entry.
// Launches a fresh task-owned Chrome profile, never the user's browser profile.
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import {pathToFileURL} from 'node:url';
const [site,model,out,mode='run']=process.argv.slice(2);
const {chromium}=await import(pathToFileURL(process.env.PLAYWRIGHT_MODULE).href);
const save=(name,value)=>fs.writeFileSync(path.join(out,name),JSON.stringify(value,null,2)+'\n');
let browser;
const state={phase:'browser_start',generation_completed:false};
const checkpoint=()=>save('worker.json',state);
checkpoint();
const server=http.createServer((req,res)=>{
  const name=new URL(req.url,'http://localhost').pathname;
  if(name==='/'){
    res.setHeader('Content-Type','text/html');
    res.end('<!doctype html><meta charset="utf-8"><input id="model" type="file"><llm-chat></llm-chat><script src="bundle.js"></script>');
    return;
  }
  const file=path.resolve(site,'.'+name);
  if(!file.startsWith(path.resolve(site)+path.sep)||!fs.existsSync(file)){res.writeHead(404);res.end();return;}
  res.setHeader('Content-Type',file.endsWith('.wasm')?'application/wasm':file.endsWith('.js')?'text/javascript':'text/plain');
  res.setHeader('Content-Length',fs.statSync(file).size);fs.createReadStream(file).pipe(res);
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
process.on('SIGTERM',async()=>{state.stopped_by_guard=true;checkpoint();await browser?.close().catch(()=>{});server.close();process.exit(2);});
try{
  browser=await chromium.launch({channel:'chrome',headless:false});
  state.browser_version=browser.version();checkpoint();
  const context=await browser.newContext();
  await context.route('**/*',route=>new URL(route.request().url()).hostname==='127.0.0.1'?route.continue():route.abort());
  const page=await context.newPage();
  page.on('console',msg=>fs.appendFileSync(path.join(out,'console.jsonl'),JSON.stringify({type:msg.type(),text:msg.text()})+'\n'));
  page.on('pageerror',error=>fs.appendFileSync(path.join(out,'page-errors.log'),String(error)+'\n'));
  await page.exposeFunction('probeCheckpoint',value=>{Object.assign(state,value);checkpoint();});
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  await page.waitForFunction(()=>Boolean(document.querySelector('llm-chat')?.llmService));
  const adapter=await page.evaluate(async()=>{
    const a=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});
    if(!a)return null;
    const info=a.info;
    return {info:{vendor:info.vendor,architecture:info.architecture,device:info.device,description:info.description},
            features:[...a.features],limits:{maxBufferSize:a.limits.maxBufferSize,
                maxStorageBufferBindingSize:a.limits.maxStorageBufferBindingSize,
                maxStorageBuffersPerShaderStage:a.limits.maxStorageBuffersPerShaderStage}};
  });
  state.adapter=adapter;state.phase='adapter_ready';checkpoint();
  if(!adapter||!adapter.features.includes('shader-f16'))throw new Error('Required hardware WebGPU adapter / shader-f16 unavailable');
  if(mode!=='probe'){
    await page.locator('#model').setInputFiles(model);
    await page.evaluate(async()=>{
      const service=document.querySelector('llm-chat').llmService;
      const options={baseOptions:{modelAssetFile:document.querySelector('#model').files[0]},
                     maxTokens:128,numResponses:1,topK:1,temperature:0,randomSeed:42,forceF32:false};
      await window.probeCheckpoint({phase:'engine_creation',options:{...options,baseOptions:{localFile:true}}});
      service.loadingProgress$.subscribe(progress=>{
        if(progress!==null)window.probeCheckpoint({loading_progress:progress});
      });
      await service.setOptions(options);
      await window.probeCheckpoint({phase:'generation',engine_created:true});
      const prompt='<|turn>user\nReply with only the word OK.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>';
      const inputTokens=service.llmInference.sizeInTokens(prompt);
      let output='',callbacks=0,doneSeen=false,cancelled=false;
      const response=await service.llmInference.generateResponse(prompt,(part,done)=>{
        output+=part;callbacks++;doneSeen=doneSeen||done;
        window.probeCheckpoint({partial_output:output,callback_count:callbacks,done_seen:doneSeen});
        // Callback counts are not token counts. Bound a runaway response independently.
        if(output.length>512&&!done){cancelled=true;service.llmInference.cancelProcessing();}
      });
      await window.probeCheckpoint({phase:'cleanup',prompt,input_tokens:inputTokens,response,
          generation_completed:doneSeen&&!cancelled,response_cancelled:cancelled});
      service.llmInference.close();
      await window.probeCheckpoint({phase:'completed',engine_closed:true});
    });
  } else {state.phase='probe_completed';checkpoint();}
}catch(error){state.error=String(error);state.error_stack=error.stack;checkpoint();process.exitCode=2;}
finally{await browser?.close().catch(()=>{});await new Promise(resolve=>server.close(resolve));}
