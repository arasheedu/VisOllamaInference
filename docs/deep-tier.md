# Deep detail tier — contract

The `deep` level adds three event types on top of `full`. It's meant for step mode: a deep
forward pass syncs the GPU after every graph node and runs an extra vocabulary projection per
layer, so it is slow. Detail order: `basic < layers < full < deep`.

## Events (all `droppable`, detail `deep`)

Every event carries the usual envelope (`v`, `req`, `seq`, `t_ns`, `type`). `step` follows the
same rules as `decode.layer` (llama-server 1-based, Ollama subtracts 1).

### `decode.node`: one graph node of a decode layer

```jsonc
{"type":"decode.node","step":2,"layer":0,"i":3,"name":"Qcur","op":"ROPE",
 "shape":[128,24],"ms":0.012,"src":[2],
 "stats":{"min":-3.1,"max":2.9,"mean":0.01,"rms":0.84}}
```

| field | type | meaning |
|---|---|---|
| `i` | int ≥ 0 | index of the node within the layer, in graph (execution) order, starting at 0 |
| `name` | string | llama.cpp tensor name with the `-<il>` suffix removed (`attn_norm`, `Qcur`, `Kcur`, `Vcur`, `kq`, `kq_soft_max`, `kqv_out`, `attn_out`, `ffn_inp`, `ffn_norm`, `ffn_gate`, `ffn_up`, `ffn_swiglu`, `ffn_out`, `l_out`, …). Unnamed nodes use `ggml_op_desc` lowercased (`mul_mat`, `add`, …) |
| `op` | string | `ggml_op_desc(node)` (`MUL_MAT`, `ROPE`, `RMS_NORM`, `MUL`, `ADD`, `SOFT_MAX`, `FLASH_ATTN_EXT`, `GLU`, `SET_ROWS`, `CPY`, …) |
| `shape` | int[] | `ne[0..3]` with trailing 1s trimmed (decode: the token dimension is 1, so `[3072]` for a hidden vector) |
| `ms` | number ≥ 0 | time of the graph segment that ends at this node (same attribution rule as `decode.op`), collector overhead excluded |
| `src` | int[]? | `i` of each source that is a node of this same layer; sources outside the layer (layer input, weights, KV cache) are left out |
| `stats` | object? | `min`/`max`/`mean`/`rms` over the current token's values of the result. Omitted for non-float results and for results over 1M elements |

View-only nodes (`RESHAPE`, `VIEW`, `PERMUTE`, `TRANSPOSE`, `NONE`) are skipped and are never in
`src`; a source that is a view is resolved to the node it views. Layer membership: nodes from the
first node of layer *L* up to and including `l_out-<L>`.

### `decode.lens`: logit lens after each layer

```jsonc
{"type":"decode.lens","step":2,"layer":14,"top":[{"id":6437,"p":0.31,"text":" blue"}],"entropy":3.2}
```

`l_out` of the current token is passed through the model's final norm and output head (the real
`output_norm` + `output` weights), softmaxed over the full vocabulary. `top` holds the
`lens_topk` (default 5) most probable tokens, highest first; `text` is the token piece. `entropy`
is in nats over the full vocabulary. The last layer's lens equals the model's raw logits
distribution (before the sampler chain).

### `decode.attn.heads`: per-head attention rows

```jsonc
{"type":"decode.attn.heads","step":2,"layer":0,"from":0,"n_kv_head":8,
 "rows":[[0.81,0.02,0.05,0.12],[0.6,0.1,0.1,0.2]]}
```

`rows[h][j]` = softmaxed attention weight of query head `h` on KV position `from + j`. There is
one row per query head (`n_head`, e.g. 24), each with `n_kv - from` entries. `from = max(0,
n_kv - attn_window)` (default window 512). Weights are rounded to 4 decimals. `n_kv_head` = number
of KV heads (GQA: query head `h` uses KV head `h / (n_head / n_kv_head)`).

## llama-server request fields

```jsonc
"trace": { "detail": "full", "deep": true, "lens_topk": 5, "attn_window": 512, ... }
```

`deep` requires `detail: "full"`. At deep, `decode.node`, `decode.lens` and `decode.attn.heads`
are emitted for **every layer** of each traced step (still subject to `layer_stride`; step mode
forces stride 1). The existing `full` events keep being emitted. New capability values:
`no_lens` (model has no output head reachable), `no_nodes`.

## Ollama

- `X-Trace-Detail: deep` (and `OLLAMA_TRACE_DETAIL=deep`) → detail level `Deep` (> `Full`).
- The runner request gets `detail: "full", deep: true`.
- The three event types are forwarded at `Deep`; everything allowed at `Full` is allowed at `Deep`.
- `request.received.detail` reports `"deep"`.
