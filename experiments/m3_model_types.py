#!/usr/bin/env python3
"""Inspect tensor storage types in the E4B payloads without inferring kernel precision."""
import argparse
import collections
import hashlib
import importlib.metadata
import json
import mmap
import re
from pathlib import Path
import tflite
from m3_preflight import MODEL_SHA256

p=argparse.ArgumentParser(description=__doc__); p.add_argument('model'); p.add_argument('output'); p.add_argument('--schema',type=Path,required=True)
a=p.parse_args()
with open(a.model,'rb') as f:
    mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
    sha=hashlib.sha256(mm).hexdigest()
    if sha!=MODEL_SHA256: raise ValueError('unexpected model')
    schema=a.schema.read_text()
    enum=re.search(r'enum TensorType : byte \{(.*?)\}',schema,re.S).group(1)
    types={int(v):k for k,v in re.findall(r'^\s*(\w+)\s*=\s*(\d+)',enum,re.M)}
    rows=[]
    for offset in range(0,len(mm)-8,16384):
        if mm[offset+4:offset+8]!=b'TFL3': continue
        model=tflite.Model.GetRootAs(mm,offset)
        signatures=[model.SignatureDefs(i).SignatureKey().decode() for i in range(model.SignatureDefsLength())]
        # Inspect the first subgraph only; do not sum buffers reused by subgraphs.
        graph=model.Subgraphs(0)
        histogram=collections.Counter(); examples={}; inputs=[]
        for i in range(graph.TensorsLength()):
            tensor=graph.Tensors(i); dtype=types.get(tensor.Type(),str(tensor.Type()))
            buffer=model.Buffers(tensor.Buffer())
            if buffer.DataLength() or buffer.Size():
                histogram[dtype]+=1
                if dtype not in examples:
                    examples[dtype]={'name':tensor.Name().decode(),'buffer':tensor.Buffer(),'inline_bytes':buffer.DataLength(),'external_bytes':buffer.Size(),'scales':tensor.Quantization().ScaleLength() if tensor.Quantization() else 0}
        for i in range(graph.InputsLength()):
            t=graph.Tensors(graph.Inputs(i)); inputs.append({'name':t.Name().decode(),'type':types[t.Type()]})
        rows.append({'offset':offset,'signatures':signatures,'scope':'first subgraph constant tensors with nonempty buffers','constant_tensor_type_counts':dict(histogram),'examples':examples,'inputs':inputs})
    result={'model_sha256':sha,'file_bytes':len(mm),'schema_package':importlib.metadata.version('tflite'),'tensor_type_schema_sha256':hashlib.sha256(a.schema.read_bytes()).hexdigest(),'tensor_type_enum':types,'payloads':rows,'limitation':'Tensor storage types do not establish all weight bit widths, quantization recipe or executed kernel precision.'}
    with open(a.output,'x') as out: json.dump(result,out,indent=2);out.write('\n')
    for r in rows: print(r['signatures'],r['constant_tensor_type_counts'])
