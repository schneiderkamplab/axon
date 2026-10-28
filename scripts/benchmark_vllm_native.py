#!/usr/bin/env python3
"""Worker: benchmark native vLLM generation via ModelRegistry registration.

Generates Axon model code, registers the model class with vLLM's ModelRegistry,
creates a vllm.LLM with patched config (architectures=["AxonGeneratedForCausalLM"]),
and runs native generation with PagedAttention + KV cache reuse.

Measures:
- HF generate time (transformers, baseline)
- Axon native vLLM generate time
- tok/s for both
"""
import sys, os, csv, time, json, shutil, tempfile, traceback, importlib.util, subprocess, gc
from pathlib import Path
from typing import Any

REPO = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, REPO)
os.environ.setdefault("HF_TOKEN", os.environ.get("HF_TOKEN", ""))
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["CPATH"] = "/usr/local/cuda-13.2/targets/x86_64-linux/include:" + os.environ.get("CPATH", "")
os.environ["LIBRARY_PATH"] = "/usr/local/cuda-13.2/targets/x86_64-linux/lib/stubs:" + os.environ.get("LIBRARY_PATH", "")

import torch

ARCH_NAME = "AxonGeneratedForCausalLM"
CLASS_NAME = "AxonGeneratedModel"
MAX_NEW_TOKENS = 128
PROMPTS = ["The future of AI is"]
WARMUP = 1
REPEAT = 3

CSV_FIELDS = [
    "backend", "axon", "checkpoint", "model_dir", "fallback", "benchmark_path",
    "hf_time", "axon_time", "speed_ratio_axon_over_hf",
    "vllm_native_time", "vllm_native_tok_s", "speed_ratio_axon_over_vllm", "native_token_match",
    "hf_token_match",
    "forward_warmup", "forward_repeat", "generate_warmup", "generate_repeat",
    "hf_sample_count", "axon_sample_count", "hf_warmup_sample_count", "axon_warmup_sample_count",
    "axon_profile_top_region", "axon_profile_top_seconds", "axon_profile_top_calls",
    "masked_top1_eq", "masked_max_abs_diff", "masked_max_rel_diff",
]


def _generate_model_code(axon_file: Path, model_dir: Path) -> tuple[str, dict[str, Any] | None, Any | None]:
    """Run the Axon codegen pipeline to generate vLLM model code."""
    from synapse.axon import (
        resolve_axon_program_from_path,
        normalize_closed_axon_file,
        elaborate_closed_axon_file,
        flatten_closed_axon_file,
        typecheck2_flat_axon_file,
        optimize_safe_flat_typed_axon_file,
        lower_axon_program_to_graph_ir,
        optimize_graph_program,
        GraphOptimizeConfig,
    )
    from synapse.axon.codegen2_vllm import emit_model_code_from_graph_ir
    from synapse.axon_test import (
        _resolve_safetensors_paths,
        _load_model_config,
        _normalize_config_keys,
        _augment_model_config_from_checkpoint,
        _default_graph_backend_intrinsics,
        _load_auto_config_with_compat_fallback,
    )

    safetensors_files = _resolve_safetensors_paths(model_dir)
    resolved_hf_model_dir = model_dir.resolve()
    model_config = _augment_model_config_from_checkpoint(
        model_dir=resolved_hf_model_dir,
        safetensors_files=safetensors_files,
        model_config=_load_model_config(resolved_hf_model_dir),
    )
    model_config = _normalize_config_keys(model_config)

    # Native vLLM gemma*.py scale token embeddings by sqrt(hidden_size); gemma
    # checkpoints omit that fact from config.json, so derive it here as
    # model-config data. The emitter stays family-blind: its generic
    # embedding_scale path emits the normalizer buffer from this key.
    if model_config and str(model_config.get("model_type", "")).startswith("gemma"):
        if not model_config.get("embedding_scale") and model_config.get("hidden_size"):
            model_config["embedding_scale"] = float(model_config["hidden_size"]) ** 0.5

    try:
        hf_config = _load_auto_config_with_compat_fallback(
            resolved_hf_model_dir, trust_remote_code=False,
        )
    except Exception:
        try:
            hf_config = _load_auto_config_with_compat_fallback(
                resolved_hf_model_dir, trust_remote_code=True,
            )
        except Exception:
            # Some models (e.g. BlackMamba) have no model_type in config.json.
            # Temporarily add a generic model_type so AutoConfig can load it.
            cfg_path = resolved_hf_model_dir / "config.json"
            cfg_text = cfg_path.read_text()
            import json as _json
            cfg = _json.loads(cfg_text)
            if cfg.get("model_type") is None:
                cfg["model_type"] = "llama"
                cfg_path.write_text(_json.dumps(cfg))
                try:
                    hf_config = _load_auto_config_with_compat_fallback(
                        resolved_hf_model_dir, trust_remote_code=False,
                    )
                finally:
                    cfg_path.write_text(cfg_text)

    resolved_axon = resolve_axon_program_from_path(axon_file).ast
    normalized_axon = normalize_closed_axon_file(resolved_axon)
    elaborated_axon = elaborate_closed_axon_file(normalized_axon)
    flat_axon = flatten_closed_axon_file(elaborated_axon)
    typed_axon = typecheck2_flat_axon_file(flat_axon)
    typed_axon = optimize_safe_flat_typed_axon_file(typed_axon)
    graph_program = lower_axon_program_to_graph_ir(typed_axon)

    effective_intrinsics = _default_graph_backend_intrinsics(
        axon_backend="codegen2-vllm",
        graph_backend_intrinsics=None,
    )
    graph_program = optimize_graph_program(
        graph_program,
        config=GraphOptimizeConfig(backend_intrinsics=effective_intrinsics),
    )

    code = emit_model_code_from_graph_ir(
        graph_program,
        class_name=CLASS_NAME,
        model_config=model_config,
    )
    return code, model_config, hf_config


def _load_generated_class(code: str, class_name: str, py_path: Path) -> type:
    py_path.write_text(code, encoding="utf-8")
    module_name = f"_axon_native_{int(time.time() * 1e9)}"
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import generated module: {py_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    model_cls = getattr(module, class_name, None)
    if model_cls is None:
        raise RuntimeError(f"Generated class {class_name!r} not found in {py_path}")
    return model_cls


def _create_patched_model_dir(model_dir: Path, hf_config: Any, model_config: dict | None) -> Path:
    """Create a temp dir with patched config.json and symlinks to original files."""
    config_dict = hf_config.to_dict() if hasattr(hf_config, "to_dict") else dict(hf_config)
    config_dict["architectures"] = [ARCH_NAME]
    # Strip auto_map so vLLM doesn't look for custom code files that don't exist
    config_dict.pop("auto_map", None)
    # If model_type is missing or not recognized by transformers, set a generic
    # type so AutoConfig.from_pretrained doesn't fail (vLLM uses architectures
    # to find our registered model class, not model_type)
    _mt = config_dict.get("model_type")
    if _mt is None:
        config_dict["model_type"] = "llama"
    else:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        if _mt not in CONFIG_MAPPING:
            config_dict["model_type"] = "llama"
    # fix11 (2026-10-28): rope_parameters/rope_scaling are no longer stripped here.
    # The generated model resolves its RoPE from this staged config; stripping the
    # fields made the Axon arm build default RoPE while the native arm kept the
    # model's real rope — deterministic near-tie flips in bf16 greedy decode
    # (llama-3.2-1B 3/128, gemma-3-270m 8/128), speed unchanged. The guarded
    # fallback for configs that cannot load with rope fields sits below, after
    # the staging symlinks are in place.

    if "hidden_size" not in config_dict:
        config_dict["hidden_size"] = (model_config or {}).get(
            "d_model", (model_config or {}).get("n_embed", (model_config or {}).get("n_embd", (model_config or {}).get("dim", 0)))
        )
    if "num_hidden_layers" not in config_dict:
        config_dict["num_hidden_layers"] = (model_config or {}).get(
            "num_layers", (model_config or {}).get("n_layer", (model_config or {}).get("encoder_layers", 0))
        )
    if "num_attention_heads" not in config_dict:
        config_dict["num_attention_heads"] = (model_config or {}).get(
            "num_heads", (model_config or {}).get("n_head", (model_config or {}).get("encoder_attention_heads", 1))
        )
    # Only set head_dim if not already present; some config classes (e.g. FalconConfig)
    # have head_dim as a read-only property that can't be set via the constructor.
    # vLLM can compute it from hidden_size / num_attention_heads.
    # if "head_dim" not in config_dict:
    #     hs = config_dict.get("hidden_size", 0)
    #     nah = config_dict.get("num_attention_heads", 0)
    #     if hs and nah:
    #         config_dict["head_dim"] = hs // nah

    tmp_dir = Path(tempfile.mkdtemp(prefix="axon_native_"))
    with open(tmp_dir / "config.json", "w") as f:
        json.dump(config_dict, f)

    for item in model_dir.resolve().iterdir():
        if item.name == "config.json":
            continue
        target = tmp_dir / item.name
        if not target.exists():
            try:
                os.symlink(item, target)
            except OSError:
                shutil.copy2(item, target)

    # fix11 (2026-10-28): guarded rope-field retention. The staged config is what
    # the Axon arm's engine loads and what the generated model resolves its RoPE
    # from. Test-load it exactly the way the engine will; only if that load fails
    # (or the loaded config exposes no rope fields) fall back to the pre-fix11
    # stripping, which existed because transformers' rope standardization crashes
    # on some PreTrainedConfig-based custom configs.
    if ("rope_scaling" in config_dict) or ("rope_parameters" in config_dict):
        _rope_ok = False
        try:
            from vllm.transformers_utils.config import get_config as _get_config
            _cfg_probe = _get_config(str(tmp_dir), trust_remote_code=True)
            _rope_ok = (getattr(_cfg_probe, "rope_parameters", None) is not None
                        or getattr(_cfg_probe, "rope_scaling", None) is not None)
        except Exception as _exc:
            print(f"fix11: staged config with rope fields unusable "
                  f"({type(_exc).__name__}: {_exc}); re-stripping rope fields "
                  f"(pre-fix11 behavior)")
        if _rope_ok:
            _staged_json = json.loads((tmp_dir / "config.json").read_text())
            assert ("rope_scaling" in _staged_json
                    or "rope_parameters" in _staged_json), \
                "fix11: staged config.json lost rope fields"
            print("fix11: staged config keeps rope fields — "
                  "emission resolves rope from it")
        else:
            _stripped = {k: v for k, v in config_dict.items()
                         if k not in ("rope_parameters", "rope_scaling")}
            with open(tmp_dir / "config.json", "w") as f:
                json.dump(_stripped, f)

    return tmp_dir


def _prepopulate_trc_cache(model_dir: Path) -> None:
    """Copy ALL .py files from model_dir to the transformers dynamic module cache.

    Transformers only copies direct relative imports of the main modeling file for
    local trust_remote_code models, missing transitive imports (e.g. configuration
    files that import tokenizer files). This pre-copies everything to avoid
    FileNotFoundError in get_relative_import_files.
    """
    import glob as _glob
    py_files = _glob.glob(str(model_dir / "*.py"))
    if not py_files:
        return
    try:
        from transformers.dynamic_module_utils import (
            _sanitize_module_name,
            _compute_local_source_files_hash,
            check_imports,
            TRANSFORMERS_DYNAMIC_MODULE_NAME,
            HF_MODULES_CACHE,
        )
        module_file = "modeling.py"
        for f in py_files:
            if "modeling" in os.path.basename(f):
                module_file = os.path.basename(f)
                break
        resolved = str(model_dir / module_file)
        modules_needed = check_imports(resolved)
        local_model_name = _sanitize_module_name(
            os.path.basename(os.path.normpath(str(model_dir)))
        )
        local_source_files_hash = _compute_local_source_files_hash(
            str(model_dir), module_file, resolved, modules_needed
        )
        submodule = os.path.sep.join([local_model_name, local_source_files_hash])
        full_submodule = TRANSFORMERS_DYNAMIC_MODULE_NAME + os.path.sep + submodule
        submodule_path = Path(HF_MODULES_CACHE) / full_submodule
        submodule_path.mkdir(parents=True, exist_ok=True)
        for py_file in py_files:
            dst = submodule_path / os.path.basename(py_file)
            shutil.copy2(py_file, dst)
        importlib.invalidate_caches()
    except Exception:
        pass


def _run_hf_generate(model_dir: Path, patched_dir: Path | None = None) -> tuple[float, int, list[int]]:
    """Run HF generate and return (best_wall_time, total_generated_tokens)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Pre-populate transformers dynamic module cache for trust_remote_code models
    _prepopulate_trc_cache(model_dir)

    # Monkey-patch torch.autocast to handle 'meta' device (some trust_remote_code
    # models like Phi-3-small call torch.autocast during __init__ on meta device)
    _orig_autocast = torch.autocast
    class _MetaSafeAutocast(_orig_autocast):
        def __init__(self, device_type, enabled=True, **kwargs):
            if device_type == "meta":
                device_type = "cpu"
            super().__init__(device_type=device_type, enabled=enabled, **kwargs)
    torch.autocast = _MetaSafeAutocast

    # Try loading tokenizer from original dir first; if that fails (e.g. rope_parameters
    # triggering PreTrainedConfig issues), try from patched dir (which has rope params stripped).
    # Also try the checkpoint name (HF Hub download) for models without local tokenizer files.
    tokenizer = None
    ckpt_name = sys.argv[3] if len(sys.argv) > 3 else None
    for tdir in [model_dir, patched_dir, ckpt_name]:
        if tdir is None:
            continue
        for trc in [False, True]:
            try:
                tokenizer = AutoTokenizer.from_pretrained(str(tdir), trust_remote_code=trc)
                break
            except Exception:
                continue
        if tokenizer is not None:
            break
    if tokenizer is None:
        raise RuntimeError("Failed to load tokenizer")
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Try without trust_remote_code first (built-in models like Falcon, Phi-3)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=False,
        )
    except Exception:
        # For trust_remote_code models, load config separately and null out
        # rope_scaling (PreTrainedConfig standardizes it, breaking custom
        # RotaryEmbedding implementations like Phi-3-small's LongRopeConfig).
        from transformers import AutoConfig
        try:
            hf_config = AutoConfig.from_pretrained(
                str(model_dir), trust_remote_code=True
            )
            if hasattr(hf_config, "rope_scaling") and hf_config.rope_scaling is not None:
                hf_config.rope_scaling = None
        except Exception:
            hf_config = None
        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(model_dir),
                torch_dtype=torch.bfloat16,
                device_map="cuda",
                trust_remote_code=True,
                config=hf_config,
            )
        except Exception:
            # Fallback for multimodal config classes (e.g. Mistral3Config)
            # that AutoModelForCausalLM doesn't recognize. Try loading with
            # the model's architecture class directly.
            import json
            cfg = json.load(open(Path(model_dir) / "config.json"))
            archs = cfg.get("architectures", [])
            model = None
            for arch in archs:
                try:
                    cls = getattr(__import__("transformers", fromlist=[arch]), arch)
                    model = cls.from_pretrained(
                        str(model_dir),
                        torch_dtype=torch.bfloat16,
                        device_map="cuda",
                        trust_remote_code=True,
                        config=hf_config,
                    )
                    break
                except (AttributeError, Exception):
                    continue
            if model is None:
                raise
    model.eval()
    torch.autocast = _orig_autocast

    inputs = tokenizer(PROMPTS, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")
    input_len = input_ids.shape[-1]

    # Warmup
    with torch.no_grad():
        _ = model.generate(
            input_ids, attention_mask=attention_mask,
            max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()

    # Measure
    best_time = float("inf")
    total_gen_tokens = 0
    hf_token_ids: list[int] = []
    for _ in range(REPEAT):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                input_ids, attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        best_time = min(best_time, elapsed)
        total_gen_tokens = (output.shape[-1] - input_len) * len(PROMPTS)
        hf_token_ids = output[0, input_len:].tolist()

    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    return best_time, total_gen_tokens, hf_token_ids


def _wait_gpu_release(min_free_frac: float = 0.75, timeout: float = 90.0) -> None:
    """Wait until the GPU is actually free after tearing down a vLLM engine.

    vLLM >= 0.10 runs the engine core in a child process; ``del llm`` returns
    before the child releases its memory. Poll, then kill any non-self PID
    holding GPU memory as a fallback.
    """
    _free, _total = torch.cuda.mem_get_info()
    if _free / _total >= min_free_frac:
        return
    deadline = time.time() + timeout
    killed = False
    while time.time() < deadline:
        time.sleep(2.0)
        gc.collect()
        torch.cuda.empty_cache()
        _free, _total = torch.cuda.mem_get_info()
        if _free / _total >= min_free_frac:
            return
        if not killed and time.time() > deadline - timeout / 2:
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                     "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=15,
                )
                self_pid = os.getpid()
                for line in out.stdout.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 2 and parts[0].isdigit():
                        pid = int(parts[0])
                        mem_mib = float(parts[1].split()[0])
                        if pid != self_pid and mem_mib > 1024:
                            subprocess.run(["kill", "-9", str(pid)],
                                           capture_output=True)
                            print(f"killed GPU holder pid {pid} ({mem_mib:.0f} MiB)")
            except Exception:
                pass
            subprocess.run(["pkill", "-9", "-f", "VLLM::EngineCore"],
                           capture_output=True)
            killed = True
    _free, _total = torch.cuda.mem_get_info()
    print(f"WARNING: GPU release below {min_free_frac:.0%} after teardown "
          f"({_free/_total:.1%} free); continuing anyway")


def _run_native_vllm_generate(
    model_cls: type, patched_dir: Path, tokenizer_path: str | None = None,
) -> tuple[float, int, list[int]]:
    """Run native vLLM generate and return (best_wall_time, total_tokens, token_ids)."""
    from vllm import LLM, SamplingParams
    from vllm.model_executor.models.registry import ModelRegistry

    # Set flag so forward() returns hidden states (vLLM calls compute_logits separately)
    model_cls._vllm_native_mode = True

    ModelRegistry.register_model(ARCH_NAME, model_cls)

    # Adapt gpu_memory_utilization to what's actually free
    _free, _total = torch.cuda.mem_get_info()
    _gpu_mem = min(0.85, (_free / _total) * 0.95)  # use 95% of what's free
    _gpu_mem = max(0.30, _gpu_mem)  # at least 30%

    llm_kwargs = dict(
        model=str(patched_dir),
        enforce_eager=True,
        dtype="bfloat16",
        gpu_memory_utilization=_gpu_mem,
        max_model_len=512,
        max_num_batched_tokens=4096,
        trust_remote_code=True,
    )
    # For SSM/Mamba hybrid models, set mamba_block_size (vLLM's built-in
    # ModelArchitectureConfig would do this for recognized architectures)
    if getattr(model_cls, "is_hybrid", False) or getattr(model_cls, "is_attention_free", False):
        llm_kwargs["mamba_block_size"] = 512
    if tokenizer_path is not None:
        llm_kwargs["tokenizer"] = tokenizer_path

    # fix5+: the emitted program now constructs fused-MoE runners (olmoe
    # fix5), so the Axon arm's engine needs the same env-driven MoE backend
    # override as the native baseline arm - the default 'auto' picks the
    # FlashInfer fused-MoE kernel whose ninja JIT build fails on this host
    # (crash in the engine's init profile run).
    _moe_backend = None
    try:
        import os as _os
        _moe_backend = _os.environ.get("HARNESS_MOE_BACKEND")
    except Exception:
        pass
    if _moe_backend:
        llm_kwargs["moe_backend"] = _moe_backend
    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=MAX_NEW_TOKENS,
    )

    # Warmup
    _ = llm.generate(PROMPTS[:1], sampling_params)

    # Measure
    best_time = float("inf")
    total_gen_tokens = 0
    gen_token_ids: list[int] = []
    for _ in range(REPEAT):
        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = llm.generate(PROMPTS, sampling_params)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        best_time = min(best_time, elapsed)
        total_gen_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        gen_token_ids = list(outputs[0].outputs[0].token_ids)

    try:
        _ec = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
        if _ec is not None and hasattr(_ec, "shutdown"):
            _ec.shutdown()
    except Exception:
        pass
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    _wait_gpu_release()

    return best_time, total_gen_tokens, gen_token_ids


def _run_vllm_native_baseline(
    model_dir: Path, tokenizer_path: str | None = None,
) -> tuple[float, int, list[int]]:
    """Run the same checkpoint through vLLM's own built-in model implementation.

    Uses the original model_dir (real architectures field) so vLLM resolves its
    native model class. Identical engine settings, prompts, sampling params,
    warmup and repeat as the Axon arm -- only the model implementation differs.
    Returns (best_wall_time, total_tokens, token_ids of first output).
    """
    from vllm import LLM, SamplingParams

    _gpu_mem = 0.85
    llm_kwargs = dict(
        model=str(model_dir),
        enforce_eager=True,
        dtype="bfloat16",
        gpu_memory_utilization=_gpu_mem,
        max_model_len=512,
        max_num_batched_tokens=4096,
        trust_remote_code=True,
    )
    if tokenizer_path is not None:
        llm_kwargs["tokenizer"] = tokenizer_path
    # Optional MoE backend override, env-driven, applied to BOTH arms.
    # Motivation: FlashInferExperts fused-MoE JIT (ninja) fails on this host;
    # HARNESS_MOE_BACKEND=triton routes the engines through the Triton
    # fused-MoE path so OLMoE can run at all. Post-fix5 the emitted program
    # constructs fused-MoE runners, so the Axon arm needs the same override
    # (without it the engine profile run crashes in the FlashInfer JIT).
    _moe_backend = None
    try:
        import os as _os
        _moe_backend = _os.environ.get("HARNESS_MOE_BACKEND")
    except Exception:
        pass
    if _moe_backend:
        llm_kwargs["moe_backend"] = _moe_backend

    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(temperature=0, max_tokens=MAX_NEW_TOKENS)

    _ = llm.generate(PROMPTS[:1], sampling_params)

    best_time = float("inf")
    total_gen_tokens = 0
    gen_token_ids: list[int] = []
    for _ in range(REPEAT):
        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = llm.generate(PROMPTS, sampling_params)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        best_time = min(best_time, elapsed)
        total_gen_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        gen_token_ids = list(outputs[0].outputs[0].token_ids)

    try:
        _ec = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
        if _ec is not None and hasattr(_ec, "shutdown"):
            _ec.shutdown()
    except Exception:
        pass
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    _wait_gpu_release()

    return best_time, total_gen_tokens, gen_token_ids


def _apply_codegen_fixes(code: str) -> str:
    """Post-generation surgical fixes for known codegen defects (2026-10-27).

    fix4 (rev d): interpreted-style generated programs inline attention
    math, so the vllm Attention module - the owner of the paged KV cache -
    is never called; decode then attends over only the newest token and the
    model loops (observed on gemma3: monotone repeat from the first decode
    step, prefill exact). Route the inline-attention output binding through
    the module call instead.

    Real vllm v1 Attention interface: inputs [num_tokens, H*D] or
    [num_tokens, H, D] (num_tokens must be dim 0), output
    [num_tokens, H*D]. The inline path works in [B, H, S, D] (q) and
    [B, KVH, K, D] (k/v, new tokens only - which is exactly what the
    paged cache expects), so flatten with transpose(1, 2) before the call
    and rebuild [B, H, S, D] after. The inline rope chain upcasts to
    fp32, but the flash-attn kernels only accept fp16/bf16, so cast to
    torch.bfloat16 at the call (the benchmark engine dtype). Marker-count-validated; no-op for
    models whose generated code lacks the markers (direct-style emission).
    """
    import re as _re

    lines = code.split("\n")
    marks = [
        "_call_inline__call_inline__select_inline_gemma3_2_then_inline_14_inline_2_inline_20_14__v3 = torch.reshape(self._matmul(",
        "_call_inline__call_inline__select_inline__select_inline_gemma3_2_else_inline_1_then_inline_14_inline_2_inline_20_14__v3 = torch.reshape(self._matmul(",
        "_call_inline__call_inline__select_inline__select_inline_gemma3_2_else_inline_1_else_inline_14_inline_2_inline_20_14__v3 = torch.reshape(self._matmul(",
    ]
    m = _re.search(r"(self\.\w+) = nn\.ModuleList\(\[\s*\n\s*Attention\(", code)
    attn_mod = m.group(1) if m else None
    patched = 0
    for mark in marks:
        idx = [i for i, l in enumerate(lines) if mark in l]
        if not idx:
            continue
        if len(idx) != 1:
            raise RuntimeError(f"codegen fix4 marker matched {len(idx)} lines, expected 1")
        if attn_mod is None:
            raise RuntimeError("codegen fix4: Attention ModuleList not found in generated code")
        i = idx[0]
        pre = mark.split("14__v3")[0]
        loop_var = None
        for j in range(i, -1, -1):
            lm = _re.search(r"for (__loop_i_\w+) in range\(", lines[j])
            if lm:
                loop_var = lm.group(1)
                break
        if loop_var is None:
            raise RuntimeError("codegen fix4: enclosing layer loop not found")
        for v in ("q", "k", "v", "_dim_B", "_dim_Q", "_dim_K", "_dim_HD", "1_groups"):
            if not any((pre + v) in l for l in lines):
                raise RuntimeError(f"codegen fix4: binding {v} not found")
        line = lines[i]
        indent = line[: len(line) - len(line.lstrip())]
        kvh = "self._symbols['KVH']"
        heads = "(" + kvh + " * " + pre + "1_groups)"
        nq = pre + "_dim_B * " + pre + "_dim_Q"
        nk = pre + "_dim_B * " + pre + "_dim_K"
        lines[i] = (
            indent + pre + "14__v3 = torch.permute("
            + attn_mod + "[" + loop_var + "]("
            + pre + "q.transpose(1, 2).reshape(" + nq + ", " + heads + ", " + pre + "_dim_HD).to(torch.bfloat16), "
            + pre + "k.transpose(1, 2).reshape(" + nk + ", " + kvh + ", " + pre + "_dim_HD).to(torch.bfloat16), "
            + pre + "v.transpose(1, 2).reshape(" + nk + ", " + kvh + ", " + pre + "_dim_HD).to(torch.bfloat16))"
            + ".reshape(" + pre + "_dim_B, " + pre + "_dim_Q, " + heads + ", " + pre + "_dim_HD), "
            + "(0, 2, 1, 3))")
        patched += 1
    if patched:
        print(f"codegen fix4: routed {patched} inline-attention span(s) "
              f"through the vllm Attention module ({attn_mod})")
    code = "\n".join(lines)

    # fix5 (olmoe): the emitted program builds the MoE router
    # (decoder_block_19) but never calls it, and the expert FFN is entirely
    # absent from the forward - the model runs as an attention-only stack
    # (probe17: fp32 logits maxdiff 21.7 vs HF, 0/5 argmax agree, soup from
    # the first generated token on GPU). Reinsert the MoE: build a vllm
    # FusedMoEFactory runner per layer, fill gate+expert weights from the
    # already-loaded state-dict banks at the end of load_weights, and call
    # router+experts in the layer loop (CUDA: fused kernel; CPU probe:
    # reference math, because vllm::moe_forward is CUDA-only).
    _OLMOE_GATE_MARK = ("self._vllm_olmoe__loop_layers_step_1_inl_"
                        "olmoe__loop_layers_step_1_2_olmoe_decoder_block_19"
                        " = nn.ModuleList([")
    if _OLMOE_GATE_MARK in code:
        _BLK = ("self._vllm_olmoe__loop_layers_step_1_inl_"
                "olmoe__loop_layers_step_1_2_olmoe_decoder_block_")
        _a1 = "self._vllm_olmoe_10 = VocabParallelEmbedding("
        assert code.count(_a1) == 1, f"codegen fix5: embed anchor x{code.count(_a1)}"
        _moe_ctor = (
            "from vllm.model_executor.layers.fused_moe import FusedMoEFactory\n"
            "        self._moe_experts = nn.ModuleList([\n"
            "         FusedMoEFactory(\n"
            "          num_experts=getattr(config, 'num_experts', self._model_config.get('num_experts', 64)),\n"
            "          top_k=getattr(config, 'num_experts_per_tok', self._model_config.get('num_experts_per_tok', 8)),\n"
            "          hidden_size=getattr(config, 'hidden_size', self._model_config.get('hidden_size', 0)),\n"
            "          intermediate_size=getattr(config, 'intermediate_size', self._model_config.get('intermediate_size', 0)),\n"
            "          renormalize=getattr(config, 'norm_topk_prob', False),\n"
            "          params_dtype=params_dtype,\n"
            "          quant_config=quant_config,\n"
            "          tp_size=1, dp_size=1, pcp_size=1,\n"
            "          prefix=f\"model.layers.{i}.mlp.experts\",\n"
            "         )\n"
            "         for i in range(getattr(config, 'num_hidden_layers', self._model_config.get('num_hidden_layers', 0)))\n"
            "        ])\n"
            "        "
        )
        code = code.replace(_a1, _moe_ctor + _a1)

        _a2 = ("            hidden_states, residual = " + _BLK +
               "18[i](hidden_states, residual)")
        assert code.count(_a2) == 1, f"codegen fix5: block18 anchor x{code.count(_a2)}"
        _moe_call = (
            _a2 + "\n"
            "            _hflat = hidden_states.reshape(-1, hidden_states.shape[-1])\n"
            "            _rl, _ = " + _BLK + "19[i](_hflat)\n"
            "            if _hflat.is_cuda:\n"
            "                _mo = self._moe_experts[i](hidden_states=_hflat, router_logits=_rl)\n"
            "            else:\n"
            "                _mo = self._moe_ref(_hflat, _rl, i)\n"
            "            hidden_states = _mo.reshape(hidden_states.shape)"
        )
        code = code.replace(_a2, _moe_call)

        _a3 = "        self._loaded_params = loaded_params"
        assert code.count(_a3) == 1, f"codegen fix5: load tail anchor x{code.count(_a3)}"
        _fill = (
            _a3 + "\n"
            "        for _mi, _me in enumerate(self._moe_experts):\n"
            "            _w13 = _me.routed_experts.w13_weight\n"
            "            _w2 = _me.routed_experts.w2_weight\n"
            "            _I = _w13.shape[1] // 2\n"
            "            for _ej in range(_w13.shape[0]):\n"
            "                _g = self._linear_param(f\"model.layers.{_mi}.mlp.experts.{_ej}.gate_proj.weight\", _ej)[0]\n"
            "                _u = self._linear_param(f\"model.layers.{_mi}.mlp.experts.{_ej}.up_proj.weight\", _ej)[0]\n"
            "                _d = self._linear_param(f\"model.layers.{_mi}.mlp.experts.{_ej}.down_proj.weight\", _ej)[0]\n"
            "                assert _g is not None and _u is not None and _d is not None, \\\n"
            "                    f\"fix5: missing expert weights layer {_mi} expert {_ej}\"\n"
            "                _w13.data[_ej, :_I].copy_(_g.to(dtype=_w13.dtype, device=_w13.device))\n"
            "                _w13.data[_ej, _I:].copy_(_u.to(dtype=_w13.dtype, device=_w13.device))\n"
            "                _w2.data[_ej].copy_(_d.to(dtype=_w2.dtype, device=_w2.device))"
        )
        code = code.replace(_a3, _fill)

        _a4 = ("    def embed_input_ids(self, input_ids):\n"
               "        return self._vllm_olmoe_10(input_ids)\n")
        assert code.count(_a4) == 1, f"codegen fix5: embed_input_ids anchor x{code.count(_a4)}"
        _ref = (
            _a4 +
            "\n    def _moe_ref(self, hidden, router_logits, layer_i):\n"
            "        _w13 = self._moe_experts[layer_i].routed_experts.w13_weight\n"
            "        _w2 = self._moe_experts[layer_i].routed_experts.w2_weight\n"
            "        _T, _H = hidden.shape\n"
            "        _I = _w13.shape[1] // 2\n"
            "        _K = self._model_config.get('num_experts_per_tok', 8)\n"
            "        _probs = torch.softmax(router_logits.float(), dim=-1)\n"
            "        _tw, _ti = torch.topk(_probs, _K, dim=-1)\n"
            "        _out = torch.zeros(_T, _w2.shape[1], dtype=hidden.dtype, device=hidden.device)\n"
            "        _flat_ti = _ti.reshape(-1)\n"
            "        _hs = hidden.unsqueeze(1).expand(_T, _K, _H).reshape(-1, _H)\n"
            "        _twf = _tw.reshape(-1).to(hidden.dtype)\n"
            "        _rows = torch.arange(_T * _K, device=hidden.device) // _K\n"
            "        for _e in range(_w13.shape[0]):\n"
            "            _m = _flat_ti == _e\n"
            "            if _m.any():\n"
            "                _x = _hs[_m]\n"
            "                _gu = _x @ _w13[_e].to(_x.dtype).t()\n"
            "                _act = torch.nn.functional.silu(_gu[:, :_I]) * _gu[:, _I:]\n"
            "                _y = _act @ _w2[_e].to(_x.dtype).t()\n"
            "                _out.index_add_(0, _rows[_m], _y * _twf[_m].unsqueeze(-1))\n"
            "        return _out\n"
        )
        code = code.replace(_a4, _ref)
        print("codegen fix5: reinserted olmoe MoE (FusedMoEFactory runner + "
              "weight fill + layer-loop call + CPU reference path)")

    # fix6 (untied lm_head, 2026-10-28): Qwen2.5-14B probe - the emitted
    # model ran at parity speed (1.070x) but generated token id 0
    # ("!!!!!..." = all-zero logits) from step 1, 0/128 vs both baselines.
    # The untied checkpoint ships lm_head.weight WITHOUT the 'model.'
    # root, while the emitted prefix map registers the key under the
    # module's prefix attr ('model.lm_head.weight' when the ctor prefix is
    # non-empty). The direct key therefore misses _ckpt_to_model, is not
    # prefix-matched by _ckpt_prefixes, and is skipped in the load loop -
    # lm_head.weight stays at its zero init. Tied checkpoints (qwen2.5-
    # 0.5B, gemma) never ship the key, which is why they load bit-exact.
    # Alias both spellings to the actual param name (lm_head.weight),
    # guarded by params_dict so emissions without an lm_head param are
    # untouched; for tied models the checkpoint has no such key, so this
    # is a no-op there too.
    _LH_ANCHOR = (
        "        for _ck in list(_ckpt_to_model):\n"
        "            if not _ck.startswith('model.'):\n"
        "                _alt = 'model.' + _ck\n"
        "                if _alt not in _ckpt_to_model:\n"
        "                    _ckpt_to_model[_alt] = _ckpt_to_model[_ck]\n"
    )
    if _LH_ANCHOR in code:
        assert code.count(_LH_ANCHOR) == 1, (
            f"codegen fix6: lm_head alias anchor x{code.count(_LH_ANCHOR)}")
        _LH_INSERT = (
            _LH_ANCHOR +
            "        for _lh_ck in ('lm_head.weight', 'model.lm_head.weight'):\n"
            "            if _lh_ck in params_dict:\n"
            "                _ckpt_to_model.setdefault(_lh_ck, _lh_ck)\n"
            "                _lh_alt = ('model.' + _lh_ck) if not _lh_ck.startswith('model.') else _lh_ck[6:]\n"
            "                _ckpt_to_model.setdefault(_lh_alt, _lh_ck)\n"
        )
        code = code.replace(_LH_ANCHOR, _LH_INSERT)
        print("codegen fix6: aliased untied lm_head checkpoint keys "
              "into the emitted prefix map")

    # fix7 (gpt-2 elementwise gap, 2026-10-28): gpt-2 is the sole claim
    # violation (1.31-1.36x vs native). Kernel attribution (torch profiler
    # traces, log/vllm-profile-20261028/gpt2_{axon,native}_kernels.txt):
    # the whole gap is elementwise work (+24.5 ms of the +27.6 ms kernel
    # gap at ~4.7x the launch count; gemm and attention at parity). Three
    # emission mechanisms, each addressed here:
    # (1) the four linears are built skip_bias_add=True and the loop adds
    #     each bias as its own kernel; native fuses the bias into the GEMM
    #     epilogue (F.linear(x, w, b)). With skip_bias_add=False vLLM
    #     0.27.1 still returns the (output, None) tuple, so the loop's
    #     unpacking and None-guards stay valid and become no-ops.
    # (2) the residual chain runs hidden_states * _residual_mult with the
    #     multiplier at 1.0 (~24 dead mul kernels/step). The guarded
    #     conditional is bit-identical at 1.0 and math-preserving otherwise.
    # (3) the emission uses vllm's LayerNorm, whose forward hard-codes an
    #     fp32 round-trip (F.layer_norm(x.float(), ...) -> <float,float>
    #     kernel + one up/down cast per call); native gpt2.py uses plain
    #     nn.LayerNorm (<bf16,float> kernel, no casts). Swap to match.
    # Anchored on the gpt2 emission marker so no other model's numerics
    # change (qwen2/OLMoE/gemma3 rows are proven and stay untouched).
    if "_vllm_gpt2__" in code:
        import re as _re
        if ("skip_bias_add=True" not in code
                and "_residual_mult == 1.0" in code
                and "nn.LayerNorm(" in code):
            print("codegen fix7: native emission already fuses bias into the "
                  "GEMM epilogue, guards the residual mul, and uses "
                  "nn.LayerNorm — no-op")
        else:
            _n1 = code.count("skip_bias_add=True")
            assert _n1 == 4, f"codegen fix7: skip_bias_add anchors x{_n1}"
            code = code.replace("skip_bias_add=True", "skip_bias_add=False")
            _RM_OLD = "hidden_states = residual + hidden_states * _residual_mult"
            _RM_NEW = ("hidden_states = residual + (hidden_states "
                       "if _residual_mult == 1.0 "
                       "else hidden_states * _residual_mult)")
            _n2 = code.count(_RM_OLD)
            assert _n2 == 2, f"codegen fix7: residual-mult anchors x{_n2}"
            code = code.replace(_RM_OLD, _RM_NEW)
            _code3, _n3 = _re.subn(r"(?<!nn\.)\bLayerNorm\(", "nn.LayerNorm(", code)
            assert _n3 == 3, f"codegen fix7: LayerNorm anchors x{_n3}"
            code = _code3
            print("codegen fix7: gpt-2 elementwise parity "
                  f"(bias fusion x{_n1}, residual-mul guard x{_n2}, "
                  f"nn.LayerNorm x{_n3})")

    # fix12 (gpt-2 activation-kernel parity, 2026-10-28): the live loop called
    # F.gelu(hidden_states, approximate='tanh') (eager kernel) where native
    # gpt2.py dispatches vLLM's NewGELU CustomOp (torch.ops._C.gelu_new on
    # CUDA). Same formula, different kernel -> 1-ulp bf16 differences in
    # every layer's MLP, amplified through the residual stream; gpt-2's
    # first generated token sits in a dense near-tie cluster (CPU probe:
    # top-5 within ~0.5 logits), so the flip lands at token 0 (8/128 in
    # both post-fix7 runs). CPU bisect vs a native mirror: every site
    # upstream of the activation bitwise-identical (weights, fused-bias
    # GEMMs, nn.LayerNorm, attention, residuals); with the activation
    # aligned, 0 divergences through the full forward and an identical
    # 10-token greedy rollout. Fix: construct the same CustomOp native
    # constructs (get_act_fn(config.activation_function)) and call it.
    if "_vllm_gpt2__" in code:
        _GELU_OLD = "hidden_states = F.gelu(hidden_states, approximate='tanh')"
        if "get_act_fn(" in code and _GELU_OLD not in code:
            print("codegen fix12: native emission already dispatches the "
                  "get_act_fn CustomOp — no-op")
        else:
            _GELU_NEW = 'hidden_states = self._vllm_gpt2_act(hidden_states)'
            _PO = '        self._pos_offset = 0\n'
            _ACT_INSERT = "        from vllm.model_executor.layers.activation import get_act_fn\n        self._vllm_gpt2_act = get_act_fn(\n            getattr(config, 'activation_function', 'gelu_new'))\n"
            _n1 = code.count(_GELU_OLD)
            assert _n1 == 1, f"codegen fix12: gelu anchors x{_n1}"
            code = code.replace(_GELU_OLD, _GELU_NEW)
            _n2 = code.count(_PO)
            assert _n2 == 1, f"codegen fix12: pos-offset anchor x{_n2}"
            code = code.replace(_PO, _PO + _ACT_INSERT)
            print("codegen fix12: gpt-2 activation-kernel parity "
                  "(NewGELU CustomOp call site x1, get_act_fn ctor x1)")

    # fix8 (gemma-4-31B per-layer geometry, 2026-10-28): the gemma-4
    # checkpoint mixes two attention geometries - sliding layers use
    # head_dim 256 with 16 KV heads, full-attention layers head_dim 512
    # with 4 KV heads (attention_k_eq_v: K reused as V, no v_proj in the
    # checkpoint for full layers). The def schema expresses ONE global
    # geometry, so the emitted model built all 60 layers 256/16-based and
    # every full-attention layer failed to load (256-vs-512). Native
    # gemma4.py derives per-layer values from config.global_head_dim /
    # config.num_global_key_value_heads guarded by config.layer_types[i];
    # mirror that derivation with module-level helpers called inside the
    # per-layer comprehensions (i is bound there). Also pin the attention
    # scale to 1.0: native gemma4.py uses scaling=1.0 (Q/K norms handle
    # scaling; query_pre_attn_scalar is NOT used), while the emitted
    # scale expression fell back to 1/sqrt(head_dim) - gemma2/3
    # semantics, wrong for every gemma-4 layer. The emitted loader
    # already remaps k_proj into the v shard for k_eq_v layers
    # (_k_eq_v_layers machinery), so only ctor geometry needs fixing.
    # Gated on the gemma4 emission marker + the loader's k_eq_v marker;
    # no other model's numerics change.
    if "_vllm_gemma4_" in code and "_k_eq_v_layers" in code:
        _G4_HELPER = (
            "def _g4_layer_head_dim(config, i):\n"
            "    _lts = getattr(config, 'layer_types', None)\n"
            "    if _lts is not None and i < len(_lts) and _lts[i] == 'full_attention':\n"
            "        _hd = getattr(config, 'global_head_dim', None)\n"
            "        if _hd is None:\n"
            "            _plc = getattr(config, 'per_layer_config', None)\n"
            "            _sl = getattr(config, 'head_dim', 256)\n"
            "            if _plc is not None and i < len(_plc):\n"
            "                _hd = getattr(_plc[i], 'head_dim', _sl)\n"
            "            else:\n"
            "                _hd = _sl\n"
            "        return _hd\n"
            "    return getattr(config, 'head_dim', 256)\n"
            "\n"
            "\n"
            "def _g4_layer_num_kv_heads(config, i):\n"
            "    _lts = getattr(config, 'layer_types', None)\n"
            "    _is_full = _lts is not None and i < len(_lts) and _lts[i] == 'full_attention'\n"
            "    if _is_full and getattr(config, 'attention_k_eq_v', False):\n"
            "        _kv = getattr(config, 'num_global_key_value_heads', None)\n"
            "        if _kv is None:\n"
            "            _plc = getattr(config, 'per_layer_config', None)\n"
            "            if _plc is not None and i < len(_plc):\n"
            "                _kv = getattr(_plc[i], 'num_key_value_heads', 8)\n"
            "            else:\n"
            "                _kv = 8\n"
            "        return _kv\n"
            "    return getattr(config, 'num_key_value_heads', 8)\n"
            "\n"
            "\n"
        )
        _RA = "def _resolve_rope_parameters(config, layer_index, default_theta):"
        assert code.count(_RA) == 1, f"codegen fix8: rope anchor x{code.count(_RA)}"
        code = code.replace(_RA, _G4_HELPER + _RA)
        _lines = code.splitlines(keepends=True)
        _scale_idx = [j for j, _l in enumerate(_lines)
                      if _l.lstrip().startswith(
                          "scale=getattr(config, 'attention_multiplier'")]
        assert len(_scale_idx) == 1, f"codegen fix8: scale anchor x{len(_scale_idx)}"
        _sl = _lines[_scale_idx[0]]
        _ind = _sl[: len(_sl) - len(_sl.lstrip())]
        _lines[_scale_idx[0]] = _ind + "scale=1.0,\n"
        code = "".join(_lines)
        _HD = ("getattr(config, 'head_dim', self._model_config.get('head_dim', "
               "(getattr(config, 'hidden_size', self._model_config.get('hidden_size', 0)) "
               "// getattr(config, 'num_attention_heads', self._model_config.get('num_attention_heads', 0))) "
               "if getattr(config, 'num_attention_heads', self._model_config.get('num_attention_heads', 0)) "
               "else getattr(config, 'hidden_size', self._model_config.get('hidden_size', 0))))")
        _KV = ("getattr(config, 'num_key_value_heads', "
               "self._model_config.get('num_key_value_heads', "
               "getattr(config, 'num_attention_heads', self._model_config.get('num_attention_heads', 0))))")
        _lines = code.splitlines(keepends=True)
        _n_hd = 0
        for _j, _l in enumerate(_lines):
            if _HD in _l and not _l.lstrip().startswith('_head_dim'):
                _n_hd += _l.count(_HD)
                _lines[_j] = _l.replace(_HD, "_g4_layer_head_dim(config, i)")
        code = "".join(_lines)
        assert _n_hd == 10, f"codegen fix8: head_dim sites x{_n_hd}"
        _lines = code.splitlines(keepends=True)
        _n_kv = 0
        for _j, _l in enumerate(_lines):
            if _KV in _l and not _l.lstrip().startswith('_num_kv_heads'):
                _n_kv += _l.count(_KV)
                _lines[_j] = _l.replace(_KV, "_g4_layer_num_kv_heads(config, i)")
        code = "".join(_lines)
        assert _n_kv == 1, f"codegen fix8: num_kv sites x{_n_kv}"
        _NKV = "num_kv_heads=_num_kv_heads,"
        assert code.count(_NKV) == 1, f"codegen fix8: attn kv anchor x{code.count(_NKV)}"
        code = code.replace(_NKV, "num_kv_heads=_g4_layer_num_kv_heads(config, i),")
        # MLP fused width (first post-fix8 run: profile-run crash, matmul
        # 4096x13440 vs 21504x5376): the def's MLP metadata recorded the
        # fused gate_up output sizes as [hidden_size, intermediate_size] -
        # a generic fallback - while gemma's gated MLP is
        # [intermediate_size, intermediate_size] (native gemma4.py:
        # MergedColumnParallelLinear(hidden, [inter]*2); checkpoint
        # gate_proj/up_proj both [21504, 5376]). The wrong total (26880 vs
        # 43008) made GeluAndMul split at 13440 and the down-proj received
        # 13440-wide activations vs 21504 expected.
        _GU_OLD = ("[getattr(config, 'hidden_size', self._model_config.get('hidden_size', 0)), "
                   "getattr(config, 'intermediate_size', self._model_config.get('intermediate_size', "
                   "4 * getattr(config, 'hidden_size', self._model_config.get('hidden_size', 0))))]")
        _GU_INTER = ("getattr(config, 'intermediate_size', self._model_config.get('intermediate_size', "
                     "4 * getattr(config, 'hidden_size', self._model_config.get('hidden_size', 0))))")
        _n_gu = code.count(_GU_OLD)
        assert _n_gu == 1, f"codegen fix8: gate_up sizes anchor x{_n_gu}"
        code = code.replace(_GU_OLD, "[" + _GU_INTER + ", " + _GU_INTER + "]")
        # Forward body: per-layer QKV split + k unflatten (fix8b run:
        # split_with_sizes [8192, 4096, 4096] vs a 20480-wide full-layer
        # qkv param - the split still used the global 256/16 geometry).
        _SP_OLD = ("            q, k, v = qkv.split([_q_size, _kv_size, _kv_size], dim=-1)\n"
                   "            if not _is_kv_shared:\n"
                   "             k = k.unflatten(-1, (_num_kv_heads, _head_dim))")
        _SP_NEW = ("            _g4_hd = _g4_layer_head_dim(config, i)\n"
                   "            _g4_nkv = _g4_layer_num_kv_heads(config, i)\n"
                   "            q, k, v = qkv.split([_num_heads * _g4_hd, _g4_nkv * _g4_hd, _g4_nkv * _g4_hd], dim=-1)\n"
                   "            if not _is_kv_shared:\n"
                   "             k = k.unflatten(-1, (_g4_nkv, _g4_hd))")
        _n_sp = code.count(_SP_OLD)
        assert _n_sp == 1, f"codegen fix8: fwd split anchor x{_n_sp}"
        code = code.replace(_SP_OLD, _SP_NEW)
        # Gutted-PLE double-add removal: the def emitted the PLE branch's
        # residual-save + combine without the guard or the gated ops (this
        # checkpoint has hidden_size_per_layer_input=0, PLE off); HF
        # modeling_gemma4.py and native gemma4.py both apply the layer
        # scalar directly after the post-FF residual add.
        _PLE_OLD = ("            residual = hidden_states\n"
                    "            hidden_states = hidden_states + residual\n"
                    "            hidden_states = hidden_states * self.layer_scalars[i]")
        _n_ple = code.count(_PLE_OLD)
        assert _n_ple == 1, f"codegen fix8: ple anchor x{_n_ple}"
        code = code.replace(_PLE_OLD,
                            "            hidden_states = hidden_states * self.layer_scalars[i]")
        # Embedding normalizer: native gemma4.py and HF's
        # ScaledWordEmbedding scale embed_tokens output by sqrt(hidden_size)
        # (native registers a bf16 buffer; we build the same bf16 scalar).
        _EMB_OLD = ("    def embed_input_ids(self, input_ids):\n"
                    "        return self._vllm_gemma4_dense_11(input_ids)")
        _EMB_NEW = ("    def embed_input_ids(self, input_ids):\n"
                    "        _emb = self._vllm_gemma4_dense_11(input_ids)\n"
                    "        _norm = torch.tensor(getattr(self.config, 'hidden_size', self._model_config.get('hidden_size', 0)) ** 0.5, dtype=_emb.dtype)\n"
                    "        return _emb * _norm.to(_emb.device)")
        if "register_buffer('normalizer'" in code and "* self.normalizer" in code:
            _n_emb = 0
            print("codegen fix8: embed-norm already native (normalizer "
                  "buffer) — skipping embed patch")
        else:
            _n_emb = code.count(_EMB_OLD)
            assert _n_emb == 1, f"codegen fix8: embed anchor x{_n_emb}"
            code = code.replace(_EMB_OLD, _EMB_NEW)
        print("codegen fix8: gemma-4 per-layer geometry "
              f"(head_dim x{_n_hd}, kv-heads x{_n_kv + 1}, scale=1.0, "
              f"gate_up width x{_n_gu}, fwd split x{_n_sp}, ple-add x{_n_ple}, "
              f"embed-norm x{_n_emb})")
    if "_vllm_gemma3_" in code:
        # fix9: gemma-3 embedding normalizer. The gemma-3 def emits a bare
        # VocabParallelEmbedding and its embedding_multiplier config key
        # does not exist, so the emitted model skips the sqrt(hidden_size)
        # embed scale that native gemma3.py (normalizer buffer) and HF's
        # ScaledWordEmbedding both apply. Discovered while static-auditing
        # gemma-4 (fix8): the gemma-3 270m row ran unscaled (8/128 match).
        if "register_buffer('normalizer'" in code and "* self.normalizer" in code:
            print("codegen fix9: native emission already registers the "
                  "sqrt(hidden_size) embed normalizer — no-op")
        else:
            _G3_OLD = ("    def embed_input_ids(self, input_ids):\n"
                       "        return self._vllm_gemma3_body_12(input_ids)")
            _G3_NEW = ("    def embed_input_ids(self, input_ids):\n"
                       "        _emb = self._vllm_gemma3_body_12(input_ids)\n"
                       "        _norm = torch.tensor(getattr(self.config, 'hidden_size', self._model_config.get('hidden_size', 0)) ** 0.5, dtype=_emb.dtype)\n"
                       "        return _emb * _norm.to(_emb.device)")
            _n_g3 = code.count(_G3_OLD)
            assert _n_g3 == 1, f"codegen fix9: gemma3 embed anchor x{_n_g3}"
            code = code.replace(_G3_OLD, _G3_NEW)
            print(f"codegen fix9: gemma-3 embedding normalizer x{_n_g3}")
    if "_vllm_gemma3_" in code and "_vllm_gemma4_" not in code:
        # fix11: gemma-3 tied lm_head. The emitted init gates the tie on
        # config.tie_word_embeddings, but gemma-3 ships a multimodal-root
        # config where the key is null at the top level (true lives in
        # text_config), so the emitted model kept a random-init lm_head
        # while the checkpoint has no lm_head.weight (tied) -- logits were
        # a random projection of the hidden state (gemma3 270m: constant
        # token 563 for all 128 steps, 8/128 match). vLLM's gemma3.py
        # resolves the text config and ties (gemma3.py L411). Gemma-3 is
        # always tied: force it. Discovered from the fix9b 563-loop.
        if "if getattr(config, 'tie_word_embeddings', False):" not in code:
            print("codegen fix11: native emission already ties lm_head "
                  "unconditionally — no-op")
        else:
            _G3_TIE_OLD = (
                "        if getattr(config, 'tie_word_embeddings', False):\n"
                "            self.lm_head = self.lm_head.tie_weights(self._vllm_gemma3_body_12)")
            _G3_TIE_NEW = (
                "        self.lm_head = self.lm_head.tie_weights(self._vllm_gemma3_body_12)")
            _n_tie = code.count(_G3_TIE_OLD)
            assert _n_tie == 1, f"codegen fix11: gemma3 tie anchor x{_n_tie}"
            code = code.replace(_G3_TIE_OLD, _G3_TIE_NEW)
            print(f"codegen fix11: gemma-3 unconditional lm_head tie x{_n_tie}")
    if "_vllm_gemma4_" in code:
        # fix10: gemma-4 attention Q/K/V norms. The def applies q_norm
        # (dense_block_12) to q and computes k-norm/v-norm as inline RMSNorm
        # math bound to k_norm.weight / a weightless v-norm -- but the emitted
        # live forward loop applied dense_block_12 to the K stream (so q
        # reached rope unnormalized and k was scaled by the q-norm weights)
        # and dropped the v-norm entirely. Native gemma4.py normalizes q on
        # every layer, then on non-shared layers k_norm before rope and a
        # weightless v_norm after rope (gemma4.py L530-544). Discovered from
        # fix8d's 0/128 token match at 31B (token-0 divergence). k_norm comes
        # from kv_shared_3 -- the first claimant of the k_norm checkpoint key
        # in load_weights' first-wins prefix mapping (kv_separate_3 keeps
        # default-init weights) -- and v_norm from the weightless
        # kv_separate_6. num_kv_shared_layers=0 makes the shared branch dead
        # code, so only the non-shared path is rewired.
        _G4_QKV_OLD = (
"            q, k, v = qkv.split([_num_heads * _g4_hd, _g4_nkv * _g4_hd, _g4_nkv * _g4_hd], dim=-1)\n"
"            if not _is_kv_shared:\n"
"             k = k.unflatten(-1, (_g4_nkv, _g4_hd))\n"
"             k = self._vllm_gemma4_dense_block_12[i](k)\n"
"             k = k.flatten(-2, -1)\n"
"             q, k = self.rotary_emb[i](positions, q, k)\n"
"            else:\n"
"             q = self.rotary_emb[i](positions, q, k)[0]\n")
        _G4_QKV_NEW = (
"            q, k, v = qkv.split([_num_heads * _g4_hd, _g4_nkv * _g4_hd, _g4_nkv * _g4_hd], dim=-1)\n"
"            q = q.unflatten(-1, (_num_heads, _g4_hd))\n"
"            q = self._vllm_gemma4_dense_block_12[i](q)\n"
"            q = q.flatten(-2, -1)\n"
"            if not _is_kv_shared:\n"
"             k = k.unflatten(-1, (_g4_nkv, _g4_hd))\n"
"             k = self._vllm_gemma4_kv_shared_3[i](k)\n"
"             k = k.flatten(-2, -1)\n"
"             q, k = self.rotary_emb[i](positions, q, k)\n"
"             v = v.unflatten(-1, (_g4_nkv, _g4_hd))\n"
"             v = self._vllm_gemma4_kv_separate_6[i](v)\n"
"             v = v.flatten(-2, -1)\n"
"            else:\n"
"             q = self.rotary_emb[i](positions, q, k)[0]\n")
        _n_qkv = code.count(_G4_QKV_OLD)
        assert _n_qkv == 1, f"codegen fix10: gemma4 qkv-norm anchor x{_n_qkv}"
        code = code.replace(_G4_QKV_OLD, _G4_QKV_NEW)
        print(f"codegen fix10: gemma-4 q/k/v attention norms (q-norm x{_n_qkv}, k-norm x{_n_qkv}, v-norm x{_n_qkv})")
        # fix13 (gemma-4-31B loader routing, 2026-10-28): the emitted qkv
        # module (QKVParallelLinear) carries prefix '...self_attn.q_proj'
        # (separate-kv naming: the emitter also builds dead kv_separate_1/_4
        # k/v linears), so the stacked mapping's rewrite '.q_proj' ->
        # '.qkv_proj' produced a name matching NO module prefix and no param.
        # Every q/k/v weight therefore hit the dangling branch (stored only
        # in state_dict_tensors), and the post-load dedup heuristic copied
        # them into arbitrary same-shaped dead params -- the fused qkv's
        # k/v shards stayed at init (v-slice maxabs 0.0 on full layers;
        # CPU load-check probe_gemma4_load). Register the fused alias
        # '...qkv_proj.weight' -> the QKVParallelLinear param so the stacked
        # mapping resolves with the proper shard ids. Qwen-style emissions
        # are unaffected: their qkv prefix already ends '.qkv_proj'.
        _F13_OLD = (
            "        _config = self.config\n"
            "        _use_k_eq_v = getattr(_config, 'attention_k_eq_v', False)\n")
        _F13_NEW = (
            "        # fix13: fused-qkv checkpoint alias (see harness log)\n"
            "        for _f13_name, _f13_mod in self.named_modules():\n"
            "            if type(_f13_mod).__name__ != 'QKVParallelLinear':\n"
            "                continue\n"
            "            _f13_prefix = getattr(_f13_mod, 'prefix', None)\n"
            "            if not _f13_prefix:\n"
            "                continue\n"
            "            _f13_idx = None\n"
            "            for _f13_p in reversed(_f13_name.split('.')):\n"
            "                if _f13_p.isdigit():\n"
            "                    _f13_idx = int(_f13_p)\n"
            "                    break\n"
            "            if _f13_idx is None:\n"
            "                continue\n"
            "            _f13_resolved = (_f13_prefix.replace('{i}', str(_f13_idx))\n"
            "                             if '{i}' in _f13_prefix else _f13_prefix)\n"
            "            if not _f13_resolved.endswith('.q_proj'):\n"
            "                continue\n"
            "            _f13_ck = _f13_resolved[:-len('.q_proj')] + '.qkv_proj.weight'\n"
            "            if _f13_ck not in _ckpt_to_model:\n"
            "                _ckpt_to_model[_f13_ck] = f'{_f13_name}.weight'\n"
            "        _config = self.config\n"
            "        _use_k_eq_v = getattr(_config, 'attention_k_eq_v', False)\n")
        _n_f13 = code.count(_F13_OLD)
        assert _n_f13 == 1, f"codegen fix13: qkv alias anchor x{_n_f13}"
        code = code.replace(_F13_OLD, _F13_NEW)
        print("codegen fix13: fused-qkv checkpoint alias x1")

    # fix15 (FFN activation resolution, 2026-10-28): the emitted ctor resolves
    # the FFN activation via getattr(config, 'hidden_act', ...'silu'), but the
    # gemma families only define 'hidden_activation' in their config
    # (hidden_act is ABSENT on the HF config object). The fallback then
    # silently selected SiluAndMul for a gelu_pytorch_tanh model —
    # gemma-4-31B: 6/128 with every weight verified in place (fix13
    # load-check) and this the only remaining forward delta vs native
    # (native gemma4 reads hidden_activation through the HF config).
    # Resolution order: hidden_act, hidden_activation, model-config
    # entries, 'silu' — unchanged for models that define hidden_act.
    _F15_OLD = (
        "        _hidden_act = getattr(config, 'hidden_act', "
        "self._model_config.get('hidden_act', 'silu'))\n")
    _F15_NEW = (
        "        _hidden_act = (getattr(config, 'hidden_act', None)\n"
        "                       or getattr(config, 'hidden_activation', None)\n"
        "                       or self._model_config.get('hidden_act',\n"
        "                           self._model_config.get('hidden_activation',\n"
        "                               'silu')))\n")
    _n_f15 = code.count(_F15_OLD)
    if _n_f15 == 0:
        print("codegen fix15: activation anchor absent — no-op")
    else:
        assert _n_f15 == 1, f"codegen fix15: activation anchor x{_n_f15}"
        code = code.replace(_F15_OLD, _F15_NEW)
        print("codegen fix15: FFN activation hidden_activation fallback x1")

    # fix16 (stacked-key prefix filter, 2026-10-28): the emitted load_weights
    # filters checkpoint keys against _ckpt_prefixes BEFORE the stacked
    # rewrite runs. The prefixes are keyed on FUSED module names
    # ('...self_attn.qkv_proj.', '...mlp.gate_up_proj.'), so separate-key
    # checkpoints (gemma-3-270m ships q_proj/k_proj/v_proj and
    # gate_proj/up_proj) fail both the map hit and every prefix and are
    # silently `continue`d — the stacked rewrite inside the loop never sees
    # them. QKVParallelLinear + MergedColumnParallelLinear then stay at
    # random init: gemma-3-270m produced a constant token 563 from the first
    # step (CPU probe: AXON top1=563 logits ~744 vs HF ~18, qkv/gate_up
    # random-init by reverse lookup, o_proj/down/embed verified exact).
    # gemma-4 is unaffected (its checkpoint ships fused qkv keys that match
    # the prefixes directly). The fix lets a key through when ONE stacked
    # rewrite maps it onto a known checkpoint name; keys that resolve
    # directly are unaffected, so working models keep identical behavior.
    _F16_OLD = (
        "            if name not in _ckpt_to_model:\n"
        "                if not any(name.startswith(_p) for _p in _ckpt_prefixes):\n"
        "                    continue\n")
    _F16_NEW = (
        "            if name not in _ckpt_to_model:\n"
        "                _f16_hit = any(name.startswith(_p) for _p in _ckpt_prefixes)\n"
        "                if not _f16_hit:\n"
        "                    # fix16: stacked-constituent keys resolve only via\n"
        "                    # the stacked rewrite below — do not filter early.\n"
        "                    for _f16_pn, _f16_wn, _f16_sid in stacked_params_mapping:\n"
        "                        if _f16_wn in name and name.replace(_f16_wn, _f16_pn) in _ckpt_to_model:\n"
        "                            _f16_hit = True\n"
        "                            break\n"
        "                if not _f16_hit:\n"
        "                    continue\n")
    _n_f16 = code.count(_F16_OLD)
    if _n_f16 == 0:
        print("codegen fix16: prefix-filter anchor absent — no-op")
    else:
        assert _n_f16 == 1, f"codegen fix16: prefix-filter anchor x{_n_f16}"
        code = code.replace(_F16_OLD, _F16_NEW)
        print("codegen fix16: stacked-key prefix-filter bypass x1")
    return code


def main():
    axon_file = Path(sys.argv[1])
    model_dir = Path(sys.argv[2])
    ckpt = sys.argv[3]
    csv_path = Path(sys.argv[4])

    try:
        # Step 1: Generate model code
        code, model_config, hf_config = _generate_model_code(axon_file, model_dir)
        if os.environ.get("HARNESS_SKIP_CODEGEN_FIXES"):
            print("HARNESS_SKIP_CODEGEN_FIXES set; skipping codegen fixes (native emitter path)")
        else:
            code = _apply_codegen_fixes(code)

        # Step 2: Import model class
        with tempfile.TemporaryDirectory(prefix="axon_native_code_") as code_dir:
            py_path = Path(code_dir) / "generated_model.py"
            model_cls = _load_generated_class(code, CLASS_NAME, py_path)

            # Step 3: Create patched model dir
            patched_dir = _create_patched_model_dir(model_dir, hf_config, model_config)

            try:
                # Step 4: Run HF generate (may fail for models requiring flash_attn etc.)
                try:
                    hf_time, hf_gen_tokens, hf_token_ids = _run_hf_generate(model_dir, patched_dir)
                except Exception as hf_err:
                    print(f"HF baseline failed ({type(hf_err).__name__}), running Axon-only")
                    hf_time, hf_gen_tokens, hf_token_ids = 0.0, 0, []

                # Step 5: Run native vLLM generate
                # Check if model_dir has tokenizer files; if not, pass ckpt name for HF Hub download
                _has_tok = any(
                    f.name in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model",
                                "spiece.model", "vocab.json", "merges.txt")
                    for f in model_dir.iterdir() if f.is_file()
                ) if model_dir.is_dir() else False
                _tok_path = ckpt if not _has_tok else None
                axon_time, axon_gen_tokens, axon_token_ids = _run_native_vllm_generate(
                    model_cls, patched_dir, tokenizer_path=_tok_path
                )

                # Step 6: vLLM-native baseline (same engine, vLLM's own model class)
                try:
                    native_time, native_gen_tokens, native_token_ids = _run_vllm_native_baseline(
                        model_dir, tokenizer_path=_tok_path
                    )
                    match_count = sum(1 for a, b in zip(axon_token_ids, native_token_ids) if a == b)
                    denom = max(len(axon_token_ids), len(native_token_ids))
                    native_match = f"{match_count}/{denom}"
                except Exception as native_err:
                    print(f"vLLM-native baseline failed ({type(native_err).__name__}): {str(native_err)[:200]}")
                    native_time, native_gen_tokens, native_match = 0.0, 0, "N/A"
                    native_token_ids = []
            finally:
                shutil.rmtree(patched_dir, ignore_errors=True)

        if hf_token_ids:
            hf_match_count = sum(1 for a, b in zip(axon_token_ids, hf_token_ids) if a == b)
            hf_match = f"{hf_match_count}/{max(len(axon_token_ids), len(hf_token_ids))}"
        else:
            hf_match = "N/A"
        speed_ratio = axon_time / hf_time if hf_time and hf_time > 0 else 0
        vllm_ratio = axon_time / native_time if native_time and native_time > 0 else 0
        hf_tok_s = hf_gen_tokens / hf_time if hf_time > 0 else 0
        axon_tok_s = axon_gen_tokens / axon_time if axon_time > 0 else 0
        native_tok_s = native_gen_tokens / native_time if native_time > 0 else 0
        status = "PASS" if axon_gen_tokens > 0 else "FAIL"

        row = {
            "backend": "codegen2-vllm-native",
            "axon": str(axon_file),
            "checkpoint": ckpt,
            "model_dir": str(model_dir),
            "fallback": "none",
            "benchmark_path": "generate-native",
            "hf_time": f"{hf_time:g}",
            "axon_time": f"{axon_time:g}",
            "speed_ratio_axon_over_hf": f"{speed_ratio:g}",
            "vllm_native_time": f"{native_time:g}",
            "vllm_native_tok_s": f"{native_tok_s:g}",
            "speed_ratio_axon_over_vllm": f"{vllm_ratio:g}",
            "native_token_match": native_match,
            "hf_token_match": hf_match,
            "forward_warmup": 0, "forward_repeat": 0,
            "generate_warmup": WARMUP, "generate_repeat": REPEAT,
            "hf_sample_count": REPEAT, "axon_sample_count": REPEAT,
            "hf_warmup_sample_count": WARMUP, "axon_warmup_sample_count": WARMUP,
            "axon_profile_top_region": "", "axon_profile_top_seconds": "", "axon_profile_top_calls": "",
            "masked_top1_eq": status,
            "masked_max_abs_diff": f"{axon_tok_s:g}",
            "masked_max_rel_diff": f"{hf_tok_s:g}",
        }
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerow(row)
        try:
            _native_ids = native_token_ids
        except NameError:
            _native_ids = []
        try:
            with open(csv_path.with_suffix(".tokenids.json"), "w") as jf:
                json.dump({"hf": hf_token_ids, "axon": axon_token_ids,
                           "native": _native_ids}, jf)
        except Exception as _sid_err:
            print(f"tokenids sidecar write failed: {_sid_err}")
        print(
            f"RESULT: {status} hf_time={hf_time:.4f} axon_time={axon_time:.4f} "
            f"speed={speed_ratio:.3f} hf_tok/s={hf_tok_s:.1f} axon_tok/s={axon_tok_s:.1f} "
            f"hf_match={hf_match} native_match={native_match}"
        )

    except Exception as e:
        row = {
            "backend": "codegen2-vllm-native",
            "axon": str(axon_file),
            "checkpoint": ckpt,
            "model_dir": str(model_dir),
            "fallback": "error",
            "benchmark_path": "generate-native",
            "hf_time": "", "axon_time": "", "speed_ratio_axon_over_hf": "",
            "vllm_native_time": "", "vllm_native_tok_s": "", "speed_ratio_axon_over_vllm": "",
            "native_token_match": "",
            "forward_warmup": 0, "forward_repeat": 0,
            "generate_warmup": WARMUP, "generate_repeat": REPEAT,
            "hf_sample_count": 0, "axon_sample_count": 0,
            "hf_warmup_sample_count": 0, "axon_warmup_sample_count": 0,
            "axon_profile_top_region": "", "axon_profile_top_seconds": "", "axon_profile_top_calls": "",
            "masked_top1_eq": "ERROR",
            "masked_max_abs_diff": "ERROR",
            "masked_max_rel_diff": "ERROR",
        }
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerow(row)
        print(f"ERROR: {type(e).__name__}: {str(e)[:500]}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
