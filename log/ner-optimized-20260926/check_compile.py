"""Check the production public forward with CUDA full-graph compilation."""
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "scripts"))
from bench_ner import generated_class

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
rows = []
for dtype in (torch.float32, torch.float16):
    state = {k: v.to("cuda", dtype=dtype) for k, v in
             load_file(str(ROOT / "checkpoint/model.safetensors")).items()}
    model = generated_class(ROOT / "export/axon_torch.py").from_state_dict(state).eval()
    compiled = torch.compile(model, fullgraph=True, dynamic=True, mode="reduce-overhead")
    for context in (torch.no_grad, torch.inference_mode):
        with context():
            for batch, length in ((1, 32), (3, 37), (8, 128)):
                ids = torch.randint(1, 30522, (batch, length), device="cuda")
                mask = torch.ones_like(ids)
                mask[0, length // 2:] = 0
                expected = model(input_ids=ids, attn_mask=mask)
                actual = compiled(input_ids=ids, attn_mask=mask)
                atol, rtol = (1e-4, 1e-4) if dtype == torch.float32 else (0.03, 0.01)
                torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
                rows.append(dict(dtype=str(dtype), context=context.__name__, batch=batch,
                                 length=length, max_abs_diff=float((actual-expected).abs().max())))
                print(rows[-1], flush=True)
    del compiled, model, state
    torch._dynamo.reset()
    torch.cuda.empty_cache()
(ROOT / "compile-check.json").write_text(json.dumps(rows, indent=2) + "\n")
