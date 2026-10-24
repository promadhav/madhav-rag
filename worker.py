"""Resident GPU model, document indexing, and local Unix-socket service."""
import os
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import fcntl
import hashlib
import json
import logging
import socketserver
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from cache_model import locate_model
from parsers import SUPPORTED
from retrieval import Index, chunks_for

ROOT=Path(__file__).resolve().parent
STATE=Path(os.environ.get('RAG_STATE',str(ROOT/'state')))
SOCKET=os.environ.get('RAG_SOCKET','/tmp/madhav-rag.sock')
EMPTY={'answer':'','citations':[]}
log=logging.getLogger('madhav-rag')




class Model:
    def __init__(self):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        if not torch.version.hip or not torch.cuda.is_available():
            raise RuntimeError('A working AMD ROCm GPU is required. CPU fallback disabled.')
        self.torch=torch
        path=locate_model()
        log.info('Loading cached model: %s',path)
        self.processor=AutoProcessor.from_pretrained(str(path),local_files_only=True,use_fast=True,
                       min_pixels=256*32*32,max_pixels=1024*32*32)
        self.model=Qwen3VLForConditionalGeneration.from_pretrained(
            str(path),local_files_only=True,dtype=torch.bfloat16,
            attn_implementation='sdpa').to('cuda').eval()
        self.generate('Reply OK.','Hello',max_tokens=4,seconds=25)

    def generate(self,system,prompt,image=None,max_tokens=256,seconds=19):
        content=[{'type':'text','text':prompt}]
        if image is not None:
            content.insert(0,{'type':'image'})
        messages=[{'role':'system','content':system},{'role':'user','content':content}]
        text=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        kwargs={'text':[text],'return_tensors':'pt','padding':True}
        if image is not None:
            kwargs['images']=[image]
        inputs=self.processor(**kwargs).to('cuda')
        inputs.pop('token_type_ids', None)
        generation={}
        if 'JSON' in system:
            from transformers import StoppingCriteria, StoppingCriteriaList
            tokenizer=self.processor.tokenizer
            bool_dtype=self.torch.bool
            prompt_length=inputs.input_ids.shape[1]
            class CompleteJSON(StoppingCriteria):
                def __call__(self,token_ids,scores,**kwargs):
                    output=tokenizer.decode(token_ids[0,prompt_length:],skip_special_tokens=True)
                    start=output.find('{')
                    done=False
                    if start>=0:
                        try:
                            value,_=json.JSONDecoder().raw_decode(output[start:])
                            done=isinstance(value,dict)
                        except ValueError:
                            pass
                    return token_ids.new_full((token_ids.shape[0],),done,dtype=bool_dtype)
            generation['stopping_criteria']=StoppingCriteriaList([CompleteJSON()])
        with self.torch.inference_mode():
            ids=self.model.generate(**inputs,max_new_tokens=max_tokens,do_sample=False,
                    use_cache=True,max_time=seconds,**generation)
        answer=self.processor.batch_decode(ids[:,inputs.input_ids.shape[1]:],
                            skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
        return answer.strip()

    def ocr(self,path,deadline):
        from vision_v3 import transcribe
        return transcribe(self,path,deadline)


class Engine:
    def __init__(self):
        self.boot=time.monotonic()
        self.model=Model()
        self.index=Index([])
        self.corpus=None
        self.indexed=False
        STATE.mkdir(parents=True,exist_ok=True)
        (STATE/'ocr').mkdir(exist_ok=True)

    def health(self):
        t=self.model.torch
        return {'ready':True,'indexed':self.indexed,'corpus':self.corpus,
                'chunks':len(self.index.chunks),'gpu':t.cuda.get_device_name(0),
                'torch':t.__version__,'hip':t.version.hip,
                'allocated_GiB':round(t.cuda.memory_allocated()/2**30,3),
                'peak_allocated_GiB':round(t.cuda.max_memory_allocated()/2**30,3),
                'reserved_GiB':round(t.cuda.memory_reserved()/2**30,3)}

    def build(self,corpus):
        root=Path(corpus).resolve()
        if not root.is_dir():
            raise ValueError('Corpus must be an existing directory')
        started=time.monotonic()
        deadline=started+500 if self.indexed else min(started+500,self.boot+550)
        # Do not leave the previous index active when a new corpus fails.
        self.index=Index([])
        self.corpus=None
        self.indexed=False
        chunks,report=[],[]
        for directory,dirs,files in os.walk(root,followlinks=False,
                                           onerror=lambda e:log.warning('Walk error: %s',e)):
            dirs[:]=sorted(d for d in dirs if not (Path(directory)/d).is_symlink())
            for name in sorted(files):
                path=Path(directory)/name
                source=path.relative_to(root).as_posix()
                if path.is_symlink() or path.suffix.lower() not in SUPPORTED:
                    report.append({'source':source,'skipped':'symlink or unknown extension'})
                    continue
                if time.monotonic()>deadline-2:
                    raise TimeoutError('Index budget exhausted; inspect corpus and timings before submission')
                try:
                    with tempfile.TemporaryDirectory(prefix='parse-',dir=STATE) as td:
                        proc=subprocess.run([sys.executable,str(ROOT/'parsers.py'),str(path),td],
                            capture_output=True,text=True,timeout=min(15,max(1,deadline-time.monotonic())))
                        if proc.returncode:
                            raise RuntimeError(proc.stderr[:500])
                        parsed=json.loads(proc.stdout)
                        if parsed.get('skipped'):
                            report.append({'source':source,'skipped':parsed['skipped']})
                            continue
                        parts=parsed['parts']
                        for img in parsed['images']:
                            key=hashlib.sha256(Path(img['path']).read_bytes()).hexdigest()
                            cached=STATE/'ocr'/f'qwen3-4b-layout-v3-{key}.txt'
                            if cached.is_file():
                                text=cached.read_text()
                            else:
                                text=self.model.ocr(img['path'],deadline)
                                cached.write_text(text)
                            if text:
                                parts.append({'label':img['label']+' [vision transcription]','text':text})
                        new=chunks_for(source,parts)
                        chunks.extend(new)
                        report.append({'source':source,'chunks':len(new),'images':len(parsed['images'])})
                except Exception as e:
                    log.warning('Skipped %s: %s',source,e)
                    report.append({'source':source,'skipped':str(e)})
        self.index=Index(chunks)
        self.corpus=str(root)
        self.indexed=True
        result={'corpus':str(root),'seconds':round(time.monotonic()-started,3),
                'startup_seconds':round(time.monotonic()-self.boot,3),
                'chunks':len(chunks),'files':report}
        (STATE/'index.json').write_text(json.dumps({'corpus':str(root),'chunks':self.index.chunks},ensure_ascii=False))
        (STATE/'index-report.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
        return result

    def answer(self,corpus,query):
        from vision_v3 import answer_question
        return answer_question(self,corpus,query,STATE)


def main():
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    # Lock before touching the socket: a second worker cannot disrupt the first.
    lock=open(SOCKET+'.lock','a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('A worker already owns this socket.')
    if os.path.exists(SOCKET):
        os.unlink(SOCKET)
    engine=Engine()

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            try:
                request=json.loads(self.rfile.readline(1_000_000))
                if request['op']=='health':
                    response=engine.health()
                elif request['op']=='index':
                    response=engine.build(request['corpus'])
                elif request['op']=='query':
                    response=engine.answer(request['corpus'],request['query'])
                else:
                    raise ValueError('Unknown operation')
            except Exception as e:
                log.exception('Worker request failed')
                response={'error':str(e)}
            try:
                self.wfile.write(json.dumps(response,ensure_ascii=False).encode()+b'\n')
            except BrokenPipeError:
                log.error('Client timed out before worker completed')

    with socketserver.UnixStreamServer(SOCKET,Handler) as server:
        os.chmod(SOCKET,0o600)
        log.info('READY %s',json.dumps(engine.health()))
        server.serve_forever(poll_interval=0.2)


if __name__=='__main__':
    main()
