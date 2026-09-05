#!/usr/bin/env python3
"""Record vision signature shapes for the exact M3/M4 model file."""
import json,mmap,hashlib,argparse
from pathlib import Path
import tflite
from m3_preflight import MODEL_SHA256
p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('output');a=p.parse_args()
with open(a.model,'rb') as f:
    mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
    if hashlib.sha256(mm).hexdigest()!=MODEL_SHA256:raise ValueError('model mismatch')
    signatures=[]
    for off in range(0,len(mm)-8,16384):
        if mm[off+4:off+8]!=b'TFL3':continue
        m=tflite.Model.GetRootAs(mm,off)
        for i in range(m.SignatureDefsLength()):
            sig=m.SignatureDefs(i); name=sig.SignatureKey().decode()
            if not name.startswith('vision'):continue
            g=m.Subgraphs(sig.SubgraphIndex())
            def tensor(v):
                t=g.Tensors(v.TensorIndex())
                return {'signature_name':v.Name().decode(),'tensor_name':t.Name().decode(),'shape':[t.Shape(i) for i in range(t.ShapeLength())],'type_enum':t.Type()}
            signatures.append({'name':name,'payload_offset':off,'subgraph':sig.SubgraphIndex(),'inputs':[tensor(sig.Inputs(j)) for j in range(sig.InputsLength())],'outputs':[tensor(sig.Outputs(j)) for j in range(sig.OutputsLength())]})
    result={'model_sha256':MODEL_SHA256,'signatures':signatures,'scope':'stored shapes, not a runtime tensor capture'}
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
