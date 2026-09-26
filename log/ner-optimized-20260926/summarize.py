"""Publish complete production-export evidence, preserving all timing samples."""
import gzip
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]

def read(path):
    return json.loads(path.read_text())

def clean(path):
    data = read(path)
    data['arguments'] = {k: v for k, v in data['arguments'].items()
                         if k not in ('run_dir', 'output')}
    return data

configs = {}
for backend in ('hf', 'axon'):
    for device, dtype in (('cpu', 'fp32'), ('cuda', 'fp32'), ('cuda', 'fp16')):
        name = f'{backend}-{device}-{dtype}'
        quality_path = ROOT / 'results-host' / f'{name}-quality.json'
        if device == 'cpu':
            quality_path = (ROOT if backend == 'axon' else ROOT.parent / 'bert-ner-20260926') / 'results' / f'{name}-quality.json'
        runs = [clean(ROOT / 'results-host' / f'{name}-performance-r{r}.json') for r in range(3)]
        quality = clean(quality_path)
        assert quality['quality']['quality_gate_passed'], name
        configs[name] = dict(quality=quality, performance=runs,
                             quality_reused_from_original_run=(backend == 'hf' and device == 'cpu'))

frozen = read(ROOT / 'training-complete.json')['checkpoint_sha256']
for config in configs.values():
    for result in [config['quality'], *config['performance']]:
        assert result['checkpoint_sha256'] == frozen
    for run in config['performance']:
        assert len(run['performance']['shapes']) == 9
        assert len(run['performance']['end_to_end']) == 3
        assert all(len(row['samples_ms']) == 50 for rows in run['performance'].values()
                   if isinstance(rows, list) for row in rows if 'samples_ms' in row)
data = dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
            checkpoint_sha256=frozen, export=read(ROOT/'export/export.json'),
            manifest=read(ROOT/'manifest.json'), configurations=configs,
            validation=dict(tests_passed=198, tests_skipped=5, mlx_executed=False,
                            cuda_compile_parity=read(ROOT/'compile-check.json')),
            protocol='Three sequential shuffled fresh processes per configuration; 10 warmups and 50 samples per case; host arrays to completed host logits; eager Torch SDPA; four threads; TF32 disabled.')
text = json.dumps(data, indent=2) + '\n'
(ROOT/'public-results.json').write_text(text)
with gzip.open(ROOT/'public-results.json.gz', 'wb') as stream:
    stream.write(text.encode())

lines = ['# Optimized MiniLM / Few-NERD runtime measurements', '',
    'The same frozen public checkpoint is executed by freshly regenerated Axon code. '
    'No retraining or test-set selection was performed. These measurements supersede '
    'the earlier Axon execution timings; the [original report](bert-token-classification-results.md) '
    'retains the ONNX baselines and their protocol.', '',
    'Measured 2026-09-26 on Linux, Threadripper PRO 7965WX / RTX 4090. '
    + data['protocol'] + ' CPU affinity and clocks were not fixed. '
    'No MPS, MLX, or Core ML execution was performed.', '',
    '[All 10,800 samples, per-process results, quality, versions, and export hashes]'
    '(../log/ner-optimized-20260926/public-results.json.gz). '
    'Tables report medians of three process p50s; speedup is HF latency divided by Axon latency.', '',
    'Changes: packed independent linear projections at load, single evaluation of '
    'Torch LayerNorm inputs, broadcast padding masks, and a compile-compatible public '
    'Torch forward. Packing recognizes primitives and typed graph structure.', '',
    '## Quality', '',
    'Full official test split: 37,648 sentences and 921,118 scored words. '
    'FP32 uses atol=rtol=1e-4; reduced precision permits at most 0.005 absolute loss '
    'in both micro and macro F1. The frozen HF CPU reference is reused; all other '
    'quality rows were rerun.', '',
    '| Configuration | Micro F1 | Macro F1 | Word agreement | Gate |',
    '|---|---:|---:|---:|---|']
for name, config in sorted(configs.items()):
    q=config['quality']['quality']
    lines.append(f"| {name} | {q['f1']:.6f} | {q['macro_f1']:.6f} | {q['word_prediction_agreement']:.6f} | PASS |")
lines += ['', '## Eager model-call latency', '',
          '| Device / precision | Batch | Tokens | HF p50 ms | Axon p50 ms | HF / Axon |',
          '|---|---:|---:|---:|---:|---:|']
for device, dtype in (('cpu','fp32'), ('cuda','fp32'), ('cuda','fp16')):
    for index in range(9):
        hf=configs[f'hf-{device}-{dtype}']['performance']
        axon=configs[f'axon-{device}-{dtype}']['performance']
        shape=hf[0]['performance']['shapes'][index]
        h=statistics.median(r['performance']['shapes'][index]['p50_ms'] for r in hf)
        a=statistics.median(r['performance']['shapes'][index]['p50_ms'] for r in axon)
        lines.append(f"| {device} {dtype} | {shape['batch']} | {shape['length']} | {h:.3f} | {a:.3f} | {h/a:.2f}x |")
lines += ['', '## End-to-end word-to-span latency', '',
          'Includes tokenization, windowing, transfers, inference, and IO span decoding '
          'on 256 deterministic public test sentences. Inputs are presegmented words.', '',
          '| Device / precision | Batch | HF p50 ms | Axon p50 ms | HF / Axon |',
          '|---|---:|---:|---:|---:|']
for device, dtype in (('cpu','fp32'), ('cuda','fp32'), ('cuda','fp16')):
    for index, batch in enumerate((1,8,32)):
        def median(backend):
            return statistics.median(r['performance']['end_to_end'][index]['p50_ms']
                for r in configs[f'{backend}-{device}-{dtype}']['performance'])
        h,a=median('hf'),median('axon')
        lines.append(f'| {device} {dtype} | {batch} | {h:.3f} | {a:.3f} | {h/a:.2f}x |')
lines += ['', '## Validation and limits', '',
    '- 198 regression tests passed; five tinygrad cases skipped because clang was unavailable. '
    'Separate source-only MLX generation checks passed.',
    '- Generated public forward passed full-graph CUDA compilation parity in FP32/FP16, '
    'under no_grad and inference_mode, with padded inputs. This is a separate smoke check; '
    'the performance tables use eager execution.',
    '- The original CUDA attempt failed because the sandbox could not access the GPU. '
    'Its failed log is retained; successful CUDA runs used host GPU access.',
    '- Mac results remain pending. MLX packing requires unquantized FP32/FP16 weights. '
    'Compilation/export success is not evidence of Apple Silicon speed or quality.', '',
    f'Frozen checkpoint SHA-256: `{frozen}`.', '']
report='\n'.join(lines)
(ROOT/'public-report.md').write_text(report)
(REPO/'docs/bert-token-classification-optimized-results.md').write_text(report)
print(json.dumps({'configurations':len(configs), 'samples':10800,
                  'raw_sha256':hashlib.sha256(text.encode()).hexdigest()}))
