"""Architecture-generic LittleBit -> GGUF converter (Qwen3, Qwen3.5 / Qwen3.8, ...).

llama.cpp's convert_hf_to_gguf.py renames tensors and may permute, split or merge rows/columns of a weight
(e.g. Qwen3.5 reorders the DeltaNet V heads and merges in_proj_qkv + in_proj_z). Instead of re-implementing that per
architecture, every LittleBit linear is pushed through the real converter as two index probes:
    R[i, j] = i  (row probe)      C[i, j] = j  (column probe)
The converted outputs say which source row / column ended up where, and the same permutation is applied to the
factors: rows -> Ub rows and u1, columns -> Vb columns and v2.  Merges are switched off, so each linear stays one
tensor (llama.cpp loads the separate attn_qkv / attn_gate / ssm_beta / ssm_alpha just as well).
Non-linear trained tensors (norms, A_log, dt_bias, conv1d, small bf16 linears) go through the converter as-is.

Everything outside the trained decoder layers (token_embd, output, output_norm, MTP layer, ...) and all metadata
is copied from a base GGUF made from the original model by convert_hf_to_gguf.py.

usage: python tools/littlebit/lb_to_gguf.py <checkpoint_dir> <hf_model_dir> <base.gguf> <out.gguf>
  checkpoint_dir: model.safetensors (win_train / block_train) or layers/layer_XX.safetensors (block_train_big)
"""
import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

LLAMA = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(LLAMA / "gguf-py"))
sys.path.insert(0, str(LLAMA))
import gguf  # noqa: E402
from conversion import get_model_class  # noqa: E402
from conversion.base import ModelBase, ModelType, get_model_architecture  # noqa: E402

QK = 128


def pad_to(n, m=QK):
    return (n + m - 1) // m * m


def unpack_pos(packed, shape):
    rows, cols = (int(x) for x in shape)
    p = packed.numpy().view(np.uint32)
    bits = (p[:, :, None] >> np.arange(32, dtype=np.uint32)) & 1
    return bits.reshape(rows, -1)[:, :cols] == 0


def q1_0(pos):
    rows, n = pos.shape
    nb = n // QK
    bits = np.packbits(pos.reshape(rows, nb, QK).astype(np.uint8), axis=-1, bitorder="little")
    d = np.full((rows, nb, 1), 1.0, dtype=np.float16).view(np.uint8)
    return np.ascontiguousarray(np.concatenate([d, bits], axis=-1).reshape(rows, nb * (2 + QK // 8)))


def f32(t):
    return t.float().numpy().reshape(-1).astype(np.float32)


class Ckpt:
    def __init__(self, d):
        files = sorted(Path(d).glob("model*.safetensors")) + sorted((Path(d) / "layers").glob("layer_*.safetensors"))
        if not files:
            raise SystemExit(f"no safetensors in {d}")
        self.where = {}
        self.handles = [safe_open(str(f), "pt") for f in files]
        for h in self.handles:
            for k in h.keys():
                self.where[k] = h

    def get(self, k):
        return self.where[k].get_tensor(k)


def make_converter(hf_dir, tmpdir):
    hf_dir = Path(hf_dir)
    hparams = ModelBase.load_hparams(hf_dir, False)
    arch = get_model_architecture(hparams, ModelType.TEXT)
    cls = get_model_class(arch)
    conv = cls(hf_dir, gguf.LlamaFileType.ALL_F32, Path(tmpdir) / "probe.gguf", dry_run=True)
    conv._SHARED_INPUT_MERGES = ()  # one GGUF tensor per linear
    print(f"converter class {cls.__name__} (arch {conv.model_arch.name})")
    return conv, hparams


def convert(conv, name, data, bid):
    """Same steps the real converter takes: filter_tensors (renames, drops) then modify_tensors."""
    item = conv.filter_tensors((name, lambda: data))
    if item is None:
        return []
    name, gen = item
    return list(conv.modify_tensors(gen(), name, bid))


def probe_linear(conv, name, bid, out_f, in_f):
    """-> list of (gguf_name, row_index, col_index) describing where the converter puts rows/columns."""
    R = torch.arange(out_f, dtype=torch.float32)[:, None].expand(out_f, in_f).contiguous()
    rows = convert(conv, name, R, bid)
    del R
    C = torch.arange(in_f, dtype=torch.float32)[None, :].expand(out_f, in_f).contiguous()
    cols = convert(conv, name, C, bid)
    del C
    res = []
    for (gn, r), (gn2, c) in zip(rows, cols):
        assert gn == gn2 and r.shape == c.shape and r.ndim == 2, (name, gn, gn2, r.shape, c.shape)
        ri = r[:, 0].round().long()
        ci = c[0, :].round().long()
        if not (torch.equal(r, ri[:, None].float().expand_as(r)) and torch.equal(c, ci[None, :].float().expand_as(c))):
            raise SystemExit(f"{name}: converter transform is not a pure row/column permutation")
        res.append((gn, ri, ci))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("hf_dir", type=Path)
    ap.add_argument("base", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()

    ck = Ckpt(args.ckpt)
    keys = list(ck.where)
    layer_ids = sorted({int(m.group(1)) for k in keys if (m := re.match(r"model\.layers\.(\d+)\.", k))})
    print(f"checkpoint: {len(keys)} tensors, layers {layer_ids[0]}..{layer_ids[-1]} ({len(layer_ids)})")

    # HF prefix of the decoder layers in the original model (Qwen3.5: model.language_model.layers.)
    idx = args.hf_dir / "model.safetensors.index.json"
    hf_keys = json.loads(idx.read_text())["weight_map"].keys() if idx.exists() else \
        safe_open(str(args.hf_dir / "model.safetensors"), "pt").keys()
    lpre = next(k[: k.index("layers.0.")] for k in sorted(hf_keys)
                if "layers.0." in k and not k.startswith("mtp") and "visual" not in k)
    print(f"HF layer prefix '{lpre}'")

    reader = gguf.GGUFReader(str(args.base))
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    tmp = tempfile.TemporaryDirectory()
    conv, _ = make_converter(args.hf_dir, tmp.name)
    assert conv.model_arch == next(a for a, n in gguf.MODEL_ARCH_NAMES.items() if n == arch), "base GGUF arch mismatch"

    w = gguf.GGUFWriter(str(args.out), arch)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        vt = field.types[0]
        sub = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        val = field.contents()
        if field.name == gguf.Keys.General.NAME:
            val = f"{val} LittleBit"
        w.add_key_value(field.name, val, vt, sub_type=sub)

    # non-layer tensors the student trained (e.g. final norm in the e2e stage); embeddings/lm_head stay from base
    overrides = {}
    for k in keys:
        if k.startswith("model.layers.") or "embed_tokens" in k or k.startswith("lm_head"):
            continue
        for gn, data in convert(conv, lpre + k[len("model."):] if k.startswith("model.") else k, ck.get(k).float(), None):
            overrides[gn] = data.float().numpy().astype(np.float32)
    print(f"non-layer overrides from checkpoint: {sorted(overrides)}")

    hp = conv.hparams
    tied = hp.get("tie_word_embeddings", (hp.get("text_config") or {}).get("tie_word_embeddings", False))
    if tied and any(t.name == "output.weight" for t in reader.tensors):
        overrides["output.weight"] = None  # tied: llama.cpp falls back to token_embd
        print("tied embeddings: dropping output.weight")

    trained = set(layer_ids)
    n_copy = 0
    for t in reader.tensors:  # everything outside the trained layers comes from the base
        m = re.match(r"blk\.(\d+)\.", t.name)
        if (m and int(m.group(1)) in trained) or t.name in overrides:
            continue
        w.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)
        n_copy += 1
    for gn, arr in overrides.items():
        if arr is not None:
            w.add_tensor(gn, arr)
    print(f"copied {n_copy} tensors from base")

    probe_cache = {}
    n_bin = 0.0
    for i in layer_ids:
        pre = f"model.layers.{i}."
        lk = [k for k in keys if k.startswith(pre)]
        lin = sorted({k[len(pre):].rsplit(".", 1)[0] for k in lk if k.endswith(".U_packed")})
        done = set()
        for sub in lin:
            base = pre + sub + "."
            U_shape = ck.get(base + "U_shape").tolist()
            V_shape = ck.get(base + "V_shape").tolist()
            out_f, in_f = U_shape[0], V_shape[1]
            hf_name = f"{lpre}layers.{i}.{sub}.weight"
            ltype = (conv.hparams.get("layer_types") or [None] * (i + 1))[i]
            cache_key = (ltype, sub, out_f, in_f)
            if cache_key not in probe_cache:
                probe_cache[cache_key] = [(re.sub(r"^blk\.\d+\.", "blk.{}.", gn), ri, ci)
                                          for gn, ri, ci in probe_linear(conv, hf_name, i, out_f, in_f)]
            for path_in, path_out in (("", ""), ("_R", "r")):
                if base + f"U{path_in}_packed" not in ck.where:
                    continue
                V = unpack_pos(ck.get(base + f"V{path_in}_packed"), ck.get(base + f"V{path_in}_shape"))
                U = unpack_pos(ck.get(base + f"U{path_in}_packed"), ck.get(base + f"U{path_in}_shape"))
                v2 = f32(ck.get(base + f"v2{path_in}"))
                u1 = f32(ck.get(base + f"u1{path_in}"))
                r = V.shape[0]
                mid = np.zeros(pad_to(r), dtype=np.float32)
                mid[:r] = f32(ck.get(base + f"v1{path_in}")) * f32(ck.get(base + f"u2{path_in}"))
                rp = pad_to(r)
                for gn_t, ri, ci in probe_cache[cache_key]:
                    gn = gn_t.format(i).removesuffix(".weight")
                    ri, ci = ri.numpy(), ci.numpy()
                    assert len(ci) % QK == 0, (gn, len(ci))
                    Vp = np.ones((rp, len(ci)), dtype=bool)
                    Vp[:r] = V[:, ci]
                    Up = np.ones((len(ri), rp), dtype=bool)
                    Up[:, :r] = U[ri]
                    w.add_tensor(f"{gn}.lb_v{path_out}", q1_0(Vp), raw_dtype=gguf.GGMLQuantizationType.Q1_0)
                    w.add_tensor(f"{gn}.lb_u{path_out}", q1_0(Up), raw_dtype=gguf.GGMLQuantizationType.Q1_0)
                    w.add_tensor(f"{gn}.lb_s_v2{path_out}", np.ascontiguousarray(v2[ci]))
                    w.add_tensor(f"{gn}.lb_s_mid{path_out}", mid)
                    w.add_tensor(f"{gn}.lb_s_u1{path_out}", np.ascontiguousarray(u1[ri]))
                    n_bin += (rp * len(ci) + len(ri) * rp) * 18 / 128
            done.update(k for k in lk if k.startswith(base))

        # remaining tensors of the layer (norms, A_log, dt_bias, conv1d, small bf16 linears): plain conversion
        for k in sorted(set(lk) - done):
            hf_name = lpre + k[len("model."):]
            for gn, data in convert(conv, hf_name, ck.get(k).float(), i):
                arr = data.float().numpy().astype(np.float32)
                w.add_tensor(gn, arr)
        if i == layer_ids[0]:
            print(f"layer {i}: {len(lin)} LittleBit linears -> " +
                  ", ".join(sorted({g.format(i) for key in probe_cache for g, _, _ in probe_cache[key]})))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=True)
    w.close()
    print(f"wrote {args.out}: {args.out.stat().st_size / 2**30:.2f} GiB (1-bit matrices {n_bin / 2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
