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

On the GPU, Qwen3.8-27B LittleBit takes **about 5.2 GB of weights**. The token embeddings (1.35 GB) stay in system
RAM. The rest of the VRAM goes to the KV cache and the compute buffer.

Only 16 of the 64 layers use full attention; the other 48 are Gated DeltaNet with a fixed ~150 MB state. Because of
that the KV cache is small:

| KV type | per token | 32K | 128K | 256K |
|---|---:|---:|---:|---:|
| `q8_0` | 34 KB | 1.1 GB | 4.6 GB | 9.1 GB |
| `q4_0` | 18 KB | 0.6 GB | 2.4 GB | 4.8 GB |
| `q3_K` *(this fork)* | 14 KB | 0.5 GB | 1.9 GB | 3.7 GB |
| `q2_K` *(this fork, experimental)* | 11 KB | 0.35 GB | 1.4 GB | 2.8 GB |

Every command below uses the same base flags, `--jinja --flash-attn on --no-warmup --parallel 1
--reasoning-preserve`, and Qwen3.8's recommended thinking-mode sampling: `--temp 1.0 --top-p 0.95 --top-k 20
--min-p 0.0`. Each command is one line, so it works in cmd, PowerShell and bash.

**24 GB** (RTX 3090 / 4090): full 256K context with q8_0 KV, about 16 GB used.

```
llama-server -m Qwen3.8-27B-LittleBit-1bit.gguf -ngl 99 -c 262144 -ctk q8_0 -ctv q8_0 --mlock --jinja --flash-attn on --no-warmup --parallel 1 --reasoning-preserve --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
```

**12 GB** (RTX 3060 12G / 4070): 128K context with q4_0 KV, about 9 GB used.

```
llama-server -m Qwen3.8-27B-LittleBit-1bit.gguf -ngl 99 -c 131072 -ctk q4_0 -ctv q4_0 --jinja --flash-attn on --no-warmup --parallel 1 --reasoning-preserve --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
```

**8 GB** (RTX 3060 Ti / 3070 / 4060): 32K context with q4_0 KV and a smaller ubatch, about 6.5 GB used.

```
llama-server -m Qwen3.8-27B-LittleBit-1bit.gguf -ngl 99 -c 32768 -ctk q4_0 -ctv q4_0 -ub 256 --jinja --flash-attn on --no-warmup --parallel 1 --reasoning-preserve --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
```

For more context on 8 GB, use `-c 65536 -ctk q3_K -ctv q3_K`.

**Agentic work and long chats.** These flags match Strata's "prefill 32K" and its conversation cache:

```
-b 32768 -ub 2048 --cache-ram 8192 --ctx-checkpoints 32 -np 4 -kvu
```

| flag | what it does |
|---|---|
| `-b 32768 -ub 2048` | reads the prompt in 32K chunks and runs 2048-token GPU micro-batches, for faster prefill. Costs about 1–2 GB extra VRAM; use `-ub 1024` on 12 GB and leave it out on 8 GB. |
| `--cache-ram 8192` | keeps up to 8 GB of earlier prompts in system RAM, so a returning conversation does not prefill again |
| `--ctx-checkpoints 32` | lets the cache be reused on Qwen3.5 / Qwen3.8; the DeltaNet state cannot be rolled back, so checkpoints are needed |
| `-np 4 -kvu` | 4 parallel agent slots sharing one KV pool; `-c` is the total for all slots |

### KV cache types in this fork

| `-ctk` / `-ctv` | bits | notes |
|---|---:|---|
| `q3_K` | 3.4 | K-quant super-block. Needs head_dim % 256 == 0 (Qwen3.5 / 3.8, Gemma) |
| `q2_K` | 2.6 | experimental |
| `kvarn3` (`q3_0`) | 3.5 | KVarN storage ([beellama](https://github.com/Anbeeld/beellama.cpp) format), plus the built-in Hadamard KV rotation |
| `kvarn2` (`q2_0s`) | 2.5 | KVarN 2-bit |

Perplexity measured with this build (wikitext-2, 2048-token windows):

| KV type | Gemma 4 E4B Q8_0 | Qwen3-0.6B bf16 |
|---|---:|---:|
| q8_0 | 41.25 | 18.14 |
| q4_0 | 40.69 | 58.49 |
| q3_K | 43.91 | n/a (head 128) |
| q2_K | 41.65 | n/a |
| kvarn3 | 42.27 | 2951 |
| kvarn2 | 45.28 | 34905 |

Gemma 4 E4B (head_dim 256/512) handles every KV type. Very small models such as Qwen3-0.6B break already at q4_0,
and they do the same on stock llama.cpp (b12427: 58.49). Use q8_0 KV for those. In the older kvarn-cuda branch of
this work, kvarn2 and kvarn3 generated coherently on Qwen3.8-27B and passed needle tests on beellama. That still
needs re-measuring on this build.

<sub>These VRAM numbers are calculated, not yet measured. They will be confirmed when the full 27B model is
published. If you run out of memory, lower `-c` first, then `-ub`. The Windows desktop itself takes 0.5–1 GB.</sub>

Converting a LittleBit checkpoint:

```bash
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

This is an independent fork. It is based on [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) at
commit `4783b11` (2026-11-20) and is not meant for upstream merging. Apart from the LittleBit parts it is standard
llama.cpp. For general build instructions, backends and usage, see the
[upstream README](https://github.com/ggml-org/llama.cpp#readme) and the
[build guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md).
