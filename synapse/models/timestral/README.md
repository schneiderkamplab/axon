# Timestral neural core

`generic-timestral.axon` maps prepared patch features to normalized, ordered
quantiles. It uses the released Timestral parameter names and packed QKV layout.
Raw-series scaling, mask construction, RoPE tables, denormalization, and forecast
refinement are caller responsibilities. No forecasting package or task adapter
for `synapse axon-test` is included.

Generate code with the standard Axon CLI:

```bash
synapse axon-codegen-dump synapse/models/timestral/generic-timestral.axon \
  log/timestral_mlx.py --backend codegen2-mlx
```

Pass checkpoint configuration to the generated model, adding
`median_index = quantile_knots.index(0.5)`. The knots must be increasing and
contain `0.5`. Keep released weights and floating inputs in FP32.

The forward accepts named tensors. `R = G * V`, `S = L + NPRED`, and rows are
contiguous groups of `V` variates from each of `G` independent series.

| Input | Shape / meaning |
|---|---|
| `patches` | `[R,L,2*patch_size]`: normalized values followed by missing indicators |
| `forecast_shape` | `[R,NPRED,1]`: only its shape is used |
| `time_keep` | Boolean `[R,1,S,S]`; true means attend |
| `variate_keep` | Boolean `[G,S,1,V,V]`; groups attend independently |
| `cos`, `sin` | `[R,1,S,head_dim]`: prepared rotary tables |
| `output_shape` | `[R,NOUT,1]`: request the final `NOUT` positions |

Masks must permit self-attention, and encoder queries must not attend decoder
keys. The output is `[R,NOUT*patch_size,num_quantiles]` in normalized space.

`prune_final_block` defaults to `false`. Enabling it computes only requested
positions in the final block and can change floating-point results. Full
original-reference forecast parity is not established. Core tests cover both
settings, every layer schedule, gates, biases, and the monotone quantile head.

Architecture source: `mistralai/timestral@a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff`.
See `NOTICE` and `LICENSE` for attribution and licensing.
