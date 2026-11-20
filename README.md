# llama.cpp-littlebit · sub-1-bit LittleBit models

**A llama.cpp fork that runs [LittleBit](https://github.com/SamsungLabs/LittleBit) sub-1-bit models natively**,
without expanding them back to dense weights. Qwen3.8-27B is included.

In LittleBit, each weight matrix is replaced by binary (±1) low-rank factors plus small scale vectors. Here those
factors are stored as ggml's `Q1_0` type (1.125 bits per value) and run on CUDA and CPU.

| model | file size | PPL wikitext-2 | note |
|---|---:|---:|---|
| Qwen3-0.6B bf16 | 1.5 GB | 18.49 | baseline |
| **Qwen3-0.6B LittleBit 1-bit** | **0.36 GiB** | 45.55 | layers are 61 MiB, the rest is embeddings |
| Qwen3.8-27B Q8_0 | 29 GB | 6.547 | baseline (4 × 2048 tokens) |
| Qwen3.8-27B, first 4 layers LittleBit | – | 6.808 | early check: DeltaNet + full attention both convert correctly |
| **Qwen3.8-27B LittleBit 1-bit** | **~6.5 GB** | *training* | full model in progress |

<sub>RTX 3090, CUDA 13. llama.cpp's perplexity tool scores the second half of each 2048-token window.</sub>

### How it works

* **Format.** A LittleBit linear is `y = ((((x·v2) @ Vbᵀ) · (v1·u2)) @ Ubᵀ) · u1`, plus the same residual path.
  `Ub` and `Vb` are stored as `Q1_0`, the scale vectors as F32.
* **No new architecture.** The loader turns `<name>.lb_*` tensors into a LittleBit linear, and `build_lora_mm`
  runs the factor chain. Every architecture whose graph uses `build_lora_mm` works. Tested: `qwen3` and
  `qwen35` (Qwen3.5 / Qwen3.8, hybrid Gated DeltaNet).
* **Converter.** [`tools/littlebit/lb_to_gguf.py`](tools/littlebit/README.md) sends row and column index probes
  through llama.cpp's own `convert_hf_to_gguf.py`. Any permutation the converter applies (for example Qwen3.5's
  DeltaNet V-head reorder) is then applied to the binary factors exactly.

Details, the tensor layout and the code changes are in [tools/littlebit/README.md](tools/littlebit/README.md).

### Use it

```bash
# run a LittleBit GGUF like any other model
llama-server -m Qwen3.8-27B-LittleBit-1bit.gguf -ngl 99 -fa on -c 32768

# convert a LittleBit checkpoint
# base.gguf = the original model made with convert_hf_to_gguf.py
python tools/littlebit/lb_to_gguf.py <littlebit_checkpoint_dir> <hf_model_dir> base.gguf out.gguf
```

LittleBit GGUFs only load in this fork. Stock llama.cpp, LM Studio, Ollama and other builds do not know the
`lb_*` tensors.

### Windows binaries

**Prebuilt Windows x64 CUDA 13 binaries are on the [Releases page](https://github.com/Cybertiron/llama.cpp-littlebit/releases)**:
download, unzip and run.

* GPUs: RTX 20 / 30 / 40 / 50 series.
* CPU: all CPU variants are included and picked at runtime.
* The CUDA runtime DLLs are included.

To build from source, follow the standard llama.cpp CUDA build:

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
```

### Credits

* LittleBit method and training code: **[SamsungLabs/LittleBit](https://github.com/SamsungLabs/LittleBit)**
  (NeurIPS 2025, LittleBit-2 ICML 2026).
* `Q1_0` binary type, ggml and llama.cpp: **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)**.
* LittleBit runtime, converter and Qwen3.5 support: this fork.

### License

* Code in this repo: MIT, the same as llama.cpp.
* LittleBit's training code is CC BY-NC 4.0 and is **not** included here. Models trained with it should be treated
  as non-commercial.

### About / support

Built by Cybertiron. If it saved you some VRAM:

[![Buy Me A Coffee](https://img.buymeacoffee.com/button-api/?text=Buy%20me%20a%20coffee&emoji=%E2%98%95&slug=cybertiron&button_colour=FFDD00&font_colour=000000&font_family=Cookie&outline_colour=000000&coffee_colour=ffffff)](https://buymeacoffee.com/cybertiron)

---

## Built on llama.cpp

Apart from the LittleBit parts, this is standard llama.cpp and it tracks
[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp). For general build instructions, backends and usage,
see the [upstream README](https://github.com/ggml-org/llama.cpp#readme) and the
[build guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md).
