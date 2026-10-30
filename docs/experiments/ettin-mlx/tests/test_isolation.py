"""Test 8: the runtime package never imports torch or transformers."""

import subprocess
import sys

CODE = """
import sys
import mlx.core as mx
import ettin_mlx
from ettin_mlx import cli, generate, load
model, tok = load("ettin-encoder-17m")
mx.eval(model(mx.array([tok.encode("Hello [MASK].")])))
dec, dtok = load("ettin-decoder-17m")
generate(dec, dtok.encode_prompt("Hello"), max_new_tokens=4)
leaked = sorted(m for m in sys.modules if m.split(".")[0] in ("torch", "transformers"))
assert not leaked, leaked
print("ok")
"""


def test_runtime_does_not_import_torch_or_transformers():
    out = subprocess.run([sys.executable, "-c", CODE], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("ok")
