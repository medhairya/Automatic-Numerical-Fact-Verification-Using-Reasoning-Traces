"""
Per-trace Qwen3 embeddings for the ENGLISH split (Arabic / Spanish already exist in this format).

Output (same layout as the existing AR/ES files, so rank_traces.py reads it unchanged):
    {split}_trace_emb_en.npz : key "id_{i}" -> float16 (1 + #non-empty traces, dim)
        row 0 = claim, rows 1.. = non-empty reasoning traces in original order, L2-normalised
        i = position for train/val, query_id for test

Runs inside the notebook kernel (one thread + one model copy per GPU), so every log line is
visible in the cell. Steps:
  1. preflight : embeds ~100 real texts, prints tokens/s, NaN count, GPU memory and the ETA for
                 the whole job. fp16 NaNs -> automatic fp32 retry if the model fits, else abort.
  2. shards    : 200 claims per shard, each saved to OUT_DIR/shards as soon as it is done;
                 finished shards (also from --resume-root) are skipped.
  3. merge     : once every shard exists, writes the three final npz files.

Memory: weights stream straight to the GPU (low_cpu_mem_usage), batches are capped by a
padded-token budget, and a CUDA OOM halves the batch instead of crashing.
Models that do not fit one GPU (8B fp16 = 15 GB on a 15 GB T4) are split over all GPUs
(single copy, slower) -- or pass quant="4bit" (bitsandbytes nf4): 8B then needs ~5.5 GB, so each
T4 holds its own copy. 4-bit saves memory, not compute: 8B-4bit is ~2x slower than 4B fp16.

Notebook:   import embed_traces; embed_traces.run(data_root=..., out_dir=..., model="Qwen/Qwen3-Embedding-0.6B")
CLI:        python embed_traces.py --data-root ... --out-dir ... --model ...
"""

import argparse
import glob
import json
import os
import queue
import re
import shutil
import threading
import time

import numpy as np

SPLITS = {"train": "train.json", "val": "validation.json", "test": "test.json"}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def find(root, name):
    hits = sorted(glob.glob(os.path.join(root, "**", name), recursive=True), key=len)
    hits = [h for h in hits if "arabic" not in h.lower() and "spanish" not in h.lower()]
    if not hits:
        raise FileNotFoundError(f"cannot find {name} under {root}")
    return hits[0]


def load_json(path):
    with open(path, encoding="utf-8") as f:
        s = f.read()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return json.loads(re.sub(r'\{\s*\nx?(\s+)"(?![A-Za-z_]+"\s*:)', r'{\n        "Reasoning_traces": [\n\1"', s))


def claim_texts(e):
    return [str(e["claim"])] + [str(t) for t in e["Reasoning_traces"] if str(t).strip()]


def shard_path(out_dir, split, start):
    return os.path.join(out_dir, "shards", f"{split}_{start:06d}.npz")


# ============================================================
# MODEL REPLICA
# ============================================================
class Encoder:
    def __init__(self, model_id, dtype, devices, max_len, gpu_mem, quant=None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.max_len = max_len
        self.tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        kw = dict(torch_dtype=dtype, low_cpu_mem_usage=True, attn_implementation="sdpa")
        if quant == "4bit":
            from transformers import BitsAndBytesConfig
            # nf4 weights; `dtype` is the compute precision (fp16, or fp32 after a NaN fallback)
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
        if len(devices) > 1:  # model parallel: one copy split over several GPUs
            kw["device_map"] = "auto"
            kw["max_memory"] = {d: gpu_mem for d in devices}
        else:
            kw["device_map"] = {"": devices[0]}
        self.model = AutoModel.from_pretrained(model_id, **kw).eval()
        self.dev = next(self.model.parameters()).device

    def encode(self, batch):
        torch = self.torch
        with torch.inference_mode():
            enc = self.tok(batch, padding=True, truncation=True, max_length=self.max_len,
                           return_tensors="pt").to(self.dev)
            h = self.model(**enc).last_hidden_state[:, -1].float()   # last-token pooling (left padding)
            h = torch.nn.functional.normalize(h, dim=-1)
            return h.cpu().numpy(), int(enc["input_ids"].numel())

    def encode_safe(self, batch):
        """Returns (embeddings, padded_tokens, n_bad). OOM -> split the batch; NaN rows -> zeros."""
        try:
            out, ntok = self.encode(batch)
        except self.torch.cuda.OutOfMemoryError:
            self.torch.cuda.empty_cache()
            if len(batch) == 1:
                raise
            mid = len(batch) // 2
            a, ta, ba = self.encode_safe(batch[:mid])
            b, tb, bb = self.encode_safe(batch[mid:])
            return np.vstack([a, b]), ta + tb, ba + bb
        bad = ~np.isfinite(out).all(1)
        out[bad] = 0.0
        return out, ntok, int(bad.sum())


def embed_texts(enc, texts, token_budget, max_batch):
    """Length-sorted, token-budget batching. Returns (float16 array, padded tokens, #NaN rows)."""
    lens = np.array([len(t) for t in texts])
    order = np.argsort(-lens)                 # longest first: an OOM shows up immediately
    embs = [None] * len(texts)
    ntok = nbad = 0
    i = 0
    while i < len(order):
        longest = min(enc.max_len, int(lens[order[i]] / 3.5) + 8)   # chars/3.5 over-estimates tokens
        bs = max(1, min(max_batch, token_budget // longest))
        idx = order[i : i + bs]
        out, t, b = enc.encode_safe([texts[j] for j in idx])
        for j, v in zip(idx, out):
            embs[j] = v.astype(np.float16)
        ntok += t
        nbad += b
        i += bs
    return np.stack(embs), ntok, nbad


# ============================================================
# MAIN
# ============================================================
def run(data_root, out_dir, model="Qwen/Qwen3-Embedding-0.6B", max_len=512, shard_size=200,
        token_budget=16384, max_batch=64, time_budget_h=9.0, resume_root=None, gpu_mem="13GiB",
        preflight_only=False, quant=None):
    import torch

    t_start = time.time()
    os.makedirs(os.path.join(out_dir, "shards"), exist_ok=True)
    if resume_root:
        n = 0
        for f in glob.glob(os.path.join(resume_root, "**", "shards", "*.npz"), recursive=True):
            dst = os.path.join(out_dir, "shards", os.path.basename(f))
            if ".tmp" not in f and not os.path.exists(dst):
                shutil.copy(f, dst)
                n += 1
        log(f"resumed {n} finished shards from {resume_root}")

    data = {s: load_json(find(data_root, f)) for s, f in SPLITS.items()}
    plan = [(s, a, min(len(data[s]), a + shard_size)) for s in SPLITS for a in range(0, len(data[s]), shard_size)]
    todo = [p for p in plan if not os.path.exists(shard_path(out_dir, p[0], p[1]))]
    total_chars = sum(min(len(t), max_len * 4) for s, a, b in todo for e in data[s][a:b] for t in claim_texts(e))
    log(f"{len(plan)} shards, {len(todo)} to do, ~{total_chars / 4 / 1e6:.1f}M tokens (after truncation to {max_len})")

    # ---- replicas: one per GPU, or one split over all GPUs when the model is too big ----
    n_gpu = torch.cuda.device_count()
    if n_gpu == 0:
        raise RuntimeError("no GPU visible: set Accelerator = GPU T4 x2")
    gpu_gib = min(torch.cuda.get_device_properties(i).total_memory for i in range(n_gpu)) / 2**30
    from huggingface_hub import HfApi
    try:
        size_gib = sum(s.size or 0 for s in HfApi().model_info(model, files_metadata=True).siblings
                       if s.rfilename.endswith(".safetensors")) / 2**30
    except Exception:
        size_gib = 16.0 if "8B" in model else (8.0 if "4B" in model else 1.2)
    if quant == "4bit":
        size_gib *= 0.37          # nf4 linear layers + fp16 embedding table (8B: ~15 -> ~5.5 GiB)
    fits_one = size_gib < gpu_gib - 2.5
    log(f"{n_gpu} GPU(s) x {gpu_gib:.1f} GiB | {model} {quant or ''} ~{size_gib:.1f} GiB weights | "
        f"{'one copy per GPU' if fits_one else 'single copy split over all GPUs'}")

    def build(dtype):
        if fits_one:
            return [Encoder(model, dtype, [i], max_len, gpu_mem, quant) for i in range(n_gpu)]
        return [Encoder(model, dtype, list(range(n_gpu)), max_len, gpu_mem, quant)]

    # ---- preflight on ~100 real texts ----
    sample = [t for e in data["test"][:12] for t in claim_texts(e)][:100]
    dtype = torch.float16
    encs = build(dtype)
    for attempt in range(2):
        t0 = time.time()
        _, ntok, nbad = embed_texts(encs[0], sample, token_budget, max_batch)
        torch.cuda.synchronize()
        tps = ntok / (time.time() - t0)
        log(f"preflight {str(dtype)}: {len(sample)} texts, {nbad} NaN rows, {tps:,.0f} padded tokens/s per replica, "
            f"GPU0 mem {torch.cuda.max_memory_allocated(0) / 2**30:.1f} GiB")
        if nbad == 0:
            break
        # fp32 compute: 4-bit weights stay 4-bit, so it always fits; fp16 weights must double
        if attempt == 0 and fits_one and (quant == "4bit" or 2 * size_gib < gpu_gib - 2.5):
            log("fp16 produced NaNs -> retrying in fp32")
            del encs
            torch.cuda.empty_cache()
            dtype = torch.float32
            encs = build(dtype)
        else:
            raise RuntimeError("model produces NaN embeddings in fp16 and fp32 does not fit: use "
                               "Qwen/Qwen3-Embedding-0.6B (fits in fp32) or a smaller max_len")
    eta_h = total_chars / 4 * 1.15 / (tps * len(encs)) / 3600   # +15% padding overhead
    log(f"ETA for the remaining shards: ~{eta_h:.1f} h with {len(encs)} replica(s) (time budget {time_budget_h} h)")
    if eta_h > time_budget_h:
        log("WARNING: will not finish in one session; finished shards are kept - re-run with resume_root next time")
    if preflight_only:
        return

    # ---- shards: each replica pulls from a shared queue ----
    q = queue.Queue()
    for p in todo:
        q.put(p)
    done = [0]
    errors = []
    lock = threading.Lock()

    def worker(r, enc):
        while True:
            if time.time() - t_start > time_budget_h * 3600:
                log(f"[replica {r}] time budget reached, stopping")
                return
            try:
                split, a, b = q.get_nowait()
            except queue.Empty:
                return
            try:
                t0 = time.time()
                items = data[split][a:b]
                texts, owner = [], []
                for i, e in enumerate(items):
                    ts = claim_texts(e)
                    texts += ts
                    owner += [i] * len(ts)
                E, ntok, nbad = embed_texts(enc, texts, token_budget, max_batch)
                owner = np.array(owner)
                out = {}
                for i, e in enumerate(items):
                    key = e.get("query_id", a + i) if split == "test" else a + i
                    out[f"id_{key}"] = E[owner == i]
                path = shard_path(out_dir, split, a)
                tmp = path[:-4] + ".tmp.npz"
                np.savez(tmp, **out)
                os.replace(tmp, path)
                with lock:
                    done[0] += 1
                    el = (time.time() - t_start) / 3600
                    log(f"[replica {r}] {split} {a}-{b}: {len(texts)} texts, {ntok / (time.time() - t0):,.0f} tok/s, "
                        f"NaN rows {nbad} | {done[0]}/{len(todo)} shards, elapsed {el:.2f} h")
            except Exception as ex:  # keep the other replica going
                errors.append(repr(ex))
                log(f"[replica {r}] ERROR on {split} {a}-{b}: {ex!r}")
                return

    threads = [threading.Thread(target=worker, args=(r, e), daemon=True) for r, e in enumerate(encs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    missing = [p for p in plan if not os.path.exists(shard_path(out_dir, p[0], p[1]))]
    if missing:
        log(f"{len(missing)}/{len(plan)} shards missing -> save this version's output as a dataset and "
            f"re-run with resume_root='/kaggle/input/<that dataset>'")
        return False
    for split in SPLITS:
        merged = {}
        for f in sorted(glob.glob(os.path.join(out_dir, "shards", f"{split}_*.npz"))):
            with np.load(f) as z:
                merged.update({k: z[k] for k in z.files})
        dst = os.path.join(out_dir, f"{split}_trace_emb_en.npz")
        np.savez(dst, **merged)
        log(f"merged {len(merged)} claims -> {dst}")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--data-root", default="/kaggle/input/datasets/d202511054/clef-2026-checkthat-task-2-data")
    ap.add_argument("--out-dir", default="/kaggle/working/en_trace_emb")
    ap.add_argument("--resume-root", default=None)
    ap.add_argument("--shard-size", type=int, default=200)
    ap.add_argument("--max-len", type=int, default=512, help="token truncation per text")
    ap.add_argument("--token-budget", type=int, default=16384, help="max padded tokens per batch")
    ap.add_argument("--max-batch", type=int, default=64)
    ap.add_argument("--gpu-mem", default="13GiB", help="per-GPU weight cap when one copy is split over GPUs")
    ap.add_argument("--time-budget-h", type=float, default=9.0)
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--quant", default=None, choices=[None, "4bit"], help="bitsandbytes nf4 weights")
    a = ap.parse_args()
    run(a.data_root, a.out_dir, a.model, a.max_len, a.shard_size, a.token_budget, a.max_batch,
        a.time_budget_h, a.resume_root, a.gpu_mem, a.preflight_only, a.quant)
