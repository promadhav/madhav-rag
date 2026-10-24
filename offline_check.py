"""Packaging check only, intended for a CPU Docker host with network disabled."""
import hashlib,json
from pathlib import Path
import torch
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
from cache_model import locate_model,model_files
import app,worker,parsers,retrieval,answer_v2,vision_v3
from PIL import Image
p=locate_model()
files=model_files(p)
assert json.loads((p/'config.json').read_text())['model_type']=='qwen3_vl'
assert torch.version.hip and '+rocm' in torch.__version__
weights=[p/n for n in files if n.endswith('.safetensors')]
assert len(weights)==2
print('Model:',p,'| shards:',len(weights),'| weight GiB:',round(sum(x.stat().st_size for x in weights)/2**30,3),flush=True)
print('Torch:',torch.__version__,'| ROCm:',torch.version.hip,flush=True)
proc=AutoProcessor.from_pretrained(str(p),local_files_only=True,use_fast=True,min_pixels=256*32*32,max_pixels=1024*32*32)
msg=[{'role':'user','content':[{'type':'image'},{'type':'text','text':'Read text.'}]}]
prompt=proc.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
batch=proc(text=[prompt],images=[Image.new('RGB',(128,128),'white')],return_tensors='pt',padding=True)
assert 'pixel_values' in batch and batch['input_ids'].shape[0]==1
assert Path('/app/output').is_dir()
print('MODEL FILES: PASS\nOFFLINE PROCESSOR: PASS\nAPPLICATION IMPORTS: PASS',flush=True)
print('GPU INFERENCE: NOT TESTED ON THIS CPU HOST',flush=True)
