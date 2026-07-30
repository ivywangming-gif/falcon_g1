#!/usr/bin/env python3
import argparse
import numpy as np, onnxruntime

p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--output",required=True); p.add_argument("--onnx",required=True); a=p.parse_args()
x=np.load(a.input); session=onnxruntime.InferenceSession(a.onnx,providers=["CPUExecutionProvider"]); name=session.get_inputs()[0].name; output=[]
for row in x: output.append(session.run(None,{name:row[None].astype(np.float32)})[0][0])
np.save(a.output,np.asarray(output,dtype=np.float32))
