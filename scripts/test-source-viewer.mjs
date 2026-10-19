// Test the MCP Apps bridge without a browser or external service.
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';
import {createHash, webcrypto} from 'node:crypto';

const html=await readFile(new URL('../rag3/source-viewer.html',import.meta.url),'utf8');
const script=html.match(/<script>([\s\S]*?)<\/script>/)[1];
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const partSize=384*1024;

assert.ok(!/<button\b|<a\b/.test(html),'no custom approval or download controls');
assert.ok(!/openExternal|uploadFile|getFileDownloadUrl/.test(script),'no external-link confirmation flow');

async function run({corrupt=false,wrongIdentity=false,denied=false,supported=true,mode='standard',page=false,scoped=false,noNative=false}={}){
  const bytes=Buffer.alloc(8*1024*1024+17);
  for(let i=0;i<bytes.length;i++)bytes[i]=i%251;
  const digest=sha(bytes), id='a'.repeat(64), messages=[], downloads=[];
  const elements=new Map();
  for(const name of ['filename','details','progress','status','error','page-image'])
    elements.set(name,{textContent:'',hidden:true});
  let receive;
  const sourceResult=page?{
    structuredContent:{filename:'원본.pdf',page:1,page_count:359},
    _meta:{'rag3/pageImage':{mime_type:'image/jpeg',data:'cHJldmlldw=='}}
  }:{structuredContent:{collection:'test',document_id:id,filename:'원본.pdf',file_size:bytes.length,
    mime_type:'application/pdf',source_sha256:digest,part_bytes:partSize,
    first_part_uri:`rag3://sources/test/${id}/0`}};
  function deliverSource(){receive({source:parent,data:{jsonrpc:'2.0',method:'ui/notifications/tool-result',params:sourceResult}});}
  function readPart(offset){
    const part=bytes.subarray(offset,offset+partSize),end=offset+part.length;
    return {offset,end_offset:end,next_offset:end<bytes.length?end:null,file_size:bytes.length,
      source_sha256:wrongIdentity?'b'.repeat(64):digest,part_sha256:sha(part),
      data_base64:corrupt?Buffer.from('corrupt').toString('base64'):part.toString('base64')};
  }
  function privatePart(args){
    assert.equal(args.document_id,id);assert.equal(args.collection,'test');
    return {structuredContent:{offset:args.offset},_meta:{'rag3/sourcePart':readPart(args.offset)}};
  }
  const parent={postMessage(message){
    messages.push(message);
    queueMicrotask(()=>{
      if(message.method==='ui/initialize'){
        // Host requests use their own ID space and may overlap an app request.
        receive({source:parent,data:{jsonrpc:'2.0',id:message.id,method:'ping',params:{}}});
        const capabilities=!supported?{}:scoped?{serverResources:{},serverTools:{}}:
          mode==='standard'?{serverResources:{}}:mode==='bridge'?{serverTools:{}}:{};
        if(supported&&!noNative)capabilities.downloadFile={};
        respond(message,{hostCapabilities:capabilities,hostContext:{theme:'light'}});
      }else if(message.method==='ui/notifications/initialized'){
        deliverSource();
      }else if(message.method==='resources/read'){
        if(scoped){receive({source:parent,data:{jsonrpc:'2.0',id:message.id,
          error:{code:-32602,message:'MCP app cannot read resource outside its widget scope'}}});return;}
        const body=readPart(Number(message.params.uri.split('/').at(-1)));
        const result={contents:[{uri:message.params.uri,mimeType:'application/json',text:JSON.stringify(body)}]};
        assert.ok(Buffer.byteLength(JSON.stringify(result))<1024*1024,'each MCP response stays below the SDK event cap');
        respond(message,result);
      }else if(message.method==='ui/download-file'){
        downloads.push(message.params.contents[0].resource);respond(message,{isError:denied});
      }else if(message.method==='tools/call'){
        assert.equal(message.params.name,'rag3_read_source_part');respond(message,privatePart(message.params.arguments));
      }
    });
  }};
  function respond(message,result){receive({source:parent,data:{jsonrpc:'2.0',id:message.id,result}});}
  const window={parent,addEventListener(name,handler){if(name==='message')receive=handler;}};
  if(supported&&mode!=='standard')window.openai={
    ...(mode==='legacy'?{callTool:async(name,args)=>{
      assert.equal(name,'rag3_read_source_part');messages.push({method:'legacy/callTool'});return privatePart(args);
    }}:{}),
    openExternal:async()=>assert.fail('must never open an external approval dialog')
  };
  const context={window,document:{getElementById:id=>elements.get(id),documentElement:{style:{}}},
    crypto:webcrypto,Uint8Array,Intl,Map,Promise,Error,JSON,Math,Array,Number,setTimeout,clearTimeout,
    atob:data=>Buffer.from(data,'base64').toString('binary')};
  vm.createContext(context);
  vm.runInContext(script,context,{filename:'source-viewer.html'});
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(elements.get('filename').textContent,'원본.pdf');
  if(page){assert.equal(elements.get('page-image').src,'data:image/jpeg;base64,cHJldmlldw==');
    assert.equal(elements.get('page-image').hidden,false);assert.equal(downloads.length,0);return;}
  if(!supported||noNative){assert.equal(downloads.length,0);
    assert.match(elements.get('status').textContent,/지원하지 않습니다/);return;}
  // Transfer must start from the tool result alone, with no simulated click.
  await vm.runInContext('transferTask',context);
  if(corrupt||wrongIdentity){assert.equal(downloads.length,0);assert.ok(elements.get('error').textContent);}
  else{
    assert.equal(downloads.length,1);
    assert.equal(downloads[0].mimeType,'application/pdf');
    assert.deepEqual(Buffer.from(downloads[0].blob,'base64'),bytes,'original bytes must survive multi-part download');
    const readMethod=mode==='standard'?'resources/read':mode==='bridge'?'tools/call':'legacy/callTool';
    assert.equal(messages.filter(message=>message.method===readMethod).length,Math.ceil(bytes.length/partSize));
    if(denied)assert.ok(elements.get('error').textContent);
    else assert.equal(elements.get('error').textContent,'');
    deliverSource();await new Promise(resolve=>setImmediate(resolve));
    assert.equal(downloads.length,1,'repeated host notifications must not download again');
  }
}
await run();
await run({corrupt:true});
await run({wrongIdentity:true});
await run({denied:true});
await run({supported:false});
await run({mode:'bridge'});
await run({mode:'legacy'});
await run({mode:'legacy',corrupt:true});
await run({mode:'bridge',scoped:true});
await run({mode:'legacy',noNative:true});
await run({page:true});
console.log('SOURCE_VIEWER_OK automatic original bytes, no clicks, no external dialog, deduplication, host bridges, integrity, cancellation and page preview');
