# LittleBit in llama.cpp

[![Buy Me A Coffee](https://img.buymeacoffee.com/button-api/?text=Buy%20me%20a%20coffee&emoji=%E2%98%95&slug=cybertiron&button_colour=FFDD00&font_colour=000000&font_family=Cookie&outline_colour=000000&coffee_colour=ffffff)](https://buymeacoffee.com/cybertiron)

Runs models compressed with [LittleBit](https://github.com/SamsungLabs/LittleBit) (sub-1-bit, NeurIPS 2025 /
LittleBit-2, ICML 2026) without expanding them back to dense weights.

A LittleBit linear replaces `W` (out x in) with binary low-rank factors plus scale vectors:

```
y = ((((x * v2) @ Vb^T) * (v1*u2)) @ Ub^T) * u1   (+ the same residual path)
```

`Ub` (out x r) and `Vb` (r x in) contain only +1/-1.

## How it is stored

Every converted linear `<name>` becomes:

| tensor | type | shape (ggml) |
|---|---|---|
| `<name>.lb_v`  | Q1_0 | [in, r] |
| `<name>.lb_u`  | Q1_0 | [r, out] |
| `<name>.lb_s_v2` / `lb_s_mid` / `lb_s_u1` | F32 | [in] / [r] / [out] |
| `lb_vr`, `lb_ur`, `lb_s_v2r`, `lb_s_midr`, `lb_s_u1r` | | residual path (optional) |

* The +-1 matrices use the existing `Q1_0` type (128-wide blocks, 1.125 bpw) with block scale 1.0.
* `mid = v1*u2`. The rank is padded to a multiple of 128. Padded `Ub`/`Vb` entries are +1 and padded `mid`
  entries are 0, so the padding is exact.

## Code changes

There is no new architecture. The changes are in four files:

* `src/llama-model.cpp`: in `create_tensor`, a missing `<name>.weight` with a `<name>.lb_v` present is loaded as a
  LittleBit linear and registered in `llama_model::lb_linears`.
* `src/llama-graph.cpp`: `build_lora_mm` runs the factor chain for registered weights.
* `src/llama-model-loader.cpp`: `lb_s_*` scale vectors are placed as `GGML_OP_MUL` operands.
* `src/llama-model.h`: holds the `llama_lb_linear` registry.

Because of this, every architecture whose graph goes through `build_lora_mm` works, for example Qwen3 and
Qwen3.5 / Qwen3.8 (`qwen35`, hybrid Gated DeltaNet).

## Converting

1. Make a base GGUF of the original model with `convert_hf_to_gguf.py`. Its metadata, tokenizer, embeddings,
   output head and any untrained layers (e.g. MTP) are copied.
2. Run:

   ```
   python tools/littlebit/lb_to_gguf.py <littlebit_checkpoint_dir> <hf_model_dir> <base.gguf> <out.gguf>
   ```

The checkpoint directory may contain either a single `model.safetensors` with bit-packed `U/V`, or per-layer
`layers/layer_XX.safetensors` files from block-wise training.

`convert_hf_to_gguf.py` may permute or merge the rows and columns of a weight; Qwen3.5, for example, reorders the
DeltaNet V heads. The script does not re-implement those rules. Instead it pushes row-index and column-index probes
through the real converter class and applies the permutation it finds to `Ub`/`u1` (rows) and `Vb`/`v2` (columns).
Merges are switched off, so each linear stays one tensor. A transform that is not a pure permutation aborts the
conversion.

## Results (Qwen3-0.6B, wikitext-2 test, ctx 2048)

| model | PPL PyTorch | PPL llama.cpp | file | tok/s (RTX 3090) |
|---|---|---|---|---|
| bf16 original | 20.95 | 18.49 | 1.5 GB | 351 |
| LittleBit 1 bit (block-wise + 1 epoch KD) | 50.20 | 45.55 | 0.36 GiB | 149 |

The PyTorch and llama.cpp numbers use different window scoring, but both show the same LittleBit/FP ratio
(2.40 vs 2.46).

On a model this small, LittleBit is slower than bf16 because each linear launches about 11 small kernels. Larger,
bandwidth-bound models should benefit; this is not yet measured.

### Qwen3.8-27B (`qwen35`), early check

The first 4 decoder layers (3 Gated DeltaNet + 1 full attention) were LittleBit 1-bit, block-wise trained. The
other 60 layers were left in Q8_0.

| model | PPL (wikitext-2, 4 x 2048) |
|---|---|
| Q8_0 base | 6.547 |
| first 4 layers LittleBit | 6.808 |

The full 1-bit model is about 6.5 GB: 3.4 GB of 1-bit layers, plus Q8_0 embeddings, output head and MTP layer.

## Windows binaries

[Releases](../../releases) has a prebuilt Windows x64 CUDA 13 build:

* GPUs: RTX 20 / 30 / 40 / 50 series.
* CPU: all ggml CPU variants are included, chosen at runtime.
* CUDA runtime DLLs are included.

Build it yourself the usual llama.cpp way, for example:

```
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
```

## License

* llama.cpp and these changes: MIT.
* LittleBit checkpoints come from the LittleBit training code, which is CC BY-NC 4.0 and is not included here.

## Support

If this saved you time, consider [**buying me a coffee**](https://buymeacoffee.com/cybertiron). It funds more open-source tooling. Thank you!

[![Buy Me A Coffee](https://img.buymeacoffee.com/button-api/?text=Buy%20me%20a%20coffee&emoji=%E2%98%95&slug=cybertiron&button_colour=FFDD00&font_colour=000000&font_family=Cookie&outline_colour=000000&coffee_colour=ffffff)](https://buymeacoffee.com/cybertiron)
