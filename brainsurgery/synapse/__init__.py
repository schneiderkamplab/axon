from .axon import (
    AxonBind,
    AxonModule,
    AxonParam,
    AxonRepeat,
    AxonReturn,
    AxonScope,
    lower_axon_module_to_synapse_block,
    lower_axon_module_to_synapse_spec,
    lower_axon_program_to_synapse_spec,
    parse_axon_module,
    parse_axon_program,
    parse_axon_program_from_path,
    synapse_spec_to_axon_module_text,
)
from .codegen import emit_model_code_from_synapse_spec, load_synapse_torch_op_map
from .runtime import SynapseProgramModel


def run_axon_test(*args, **kwargs):
    # Lazy import keeps benchmarking deps (e.g., transformers) out of core package import paths.
    from .axon_test import run_axon_test as _run_axon_test

    return _run_axon_test(*args, **kwargs)


def run_axon_test_matrix(*args, **kwargs):
    # Lazy import keeps benchmarking deps (e.g., transformers) out of core package import paths.
    from .axon_test_matrix import run_axon_test_matrix as _run_axon_test_matrix

    return _run_axon_test_matrix(*args, **kwargs)


__all__ = [
    "AxonBind",
    "AxonModule",
    "AxonParam",
    "AxonRepeat",
    "AxonReturn",
    "AxonScope",
    "SynapseProgramModel",
    "emit_model_code_from_synapse_spec",
    "run_axon_test",
    "run_axon_test_matrix",
    "lower_axon_module_to_synapse_block",
    "lower_axon_module_to_synapse_spec",
    "lower_axon_program_to_synapse_spec",
    "load_synapse_torch_op_map",
    "parse_axon_module",
    "parse_axon_program",
    "parse_axon_program_from_path",
    "synapse_spec_to_axon_module_text",
]
