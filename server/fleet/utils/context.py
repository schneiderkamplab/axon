# Vendored from nano-vllm (https://github.com/GeeeekExplorer/nano-vllm)
# Copyright (c) 2025 Xingkai Yu — MIT License.
# The canonical singleton lives in synapse (the serving-profile artifact
# contract): generated models import it from
# ``synapse.axon.codegen2_torch.serving_context``, and the fleet engine must
# observe the SAME process-wide context, so this module re-exports it.
# Keeping this file preserves the upstream module layout for diffability.
from synapse.axon.codegen2_torch.serving_context import (  # noqa: F401
    Context,
    get_context,
    reset_context,
    set_context,
)

__all__ = ["Context", "get_context", "reset_context", "set_context"]
