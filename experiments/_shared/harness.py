"""
Training harness: config -> run -> metrics. The notebook is a thin caller of this.

AGENT GUIDELINE
---------------
This is train.py's setup + loop + eval, wrapped so a notebook can call it repeatedly
with different configs and different seeds in one process.

Three things it changes from train.py, deliberately:
  1. Hyperparameters are an `ExpConfig` dataclass, not module globals, so a notebook can
     hold several configs at once and so the exact config lands in metrics.json.
  2. There is NO time limit. upstream stops at prepare.TIME_BUDGET seconds; a run here
     stops after `cfg.train_tokens`, which defaults to one pass over the round's data.
     The LR schedule is driven by token progress rather than elapsed seconds, so a run
     is reproducible and its schedule completes identically on any hardware - only the
     wall clock differs.
  3. A NaN/exploding loss raises `TrainingDiverged` instead of calling exit(1), which
     in a notebook would kill the kernel and lose the log.

Everything else - the model, the optimizer, the schedules, the eval - is upstream
behaviour. `prepare.evaluate_bpb` is called unmodified. Do not wrap it, do not
reimplement it, do not "fix" it.

Typical use:
    from experiments._shared.data import prepare_round
    from experiments._shared.harness import ExpConfig, run, run_seeds

    prepare_round(1)                       # FIRST. before any dataloader or eval.
    cfg = ExpConfig(round_idx=1, matrix_lr=0.02)
    m = run(cfg, outdir="experiments/exp01r1")
"""

import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import gc
import json
import math
import statistics
import time
from dataclasses import dataclass, asdict, field, replace

import torch

import prepare
from prepare import Tokenizer, make_dataloader, evaluate_bpb

from .data import round_shards, round_tokens
from .model import GPT, GPTConfig

# bf16 peak FLOPS, for the MFU number only. train.py hardcodes the H100 value; keeping it
# in the config means MFU stays meaningful on other hardware. It never affects val_bpb.
PEAK_FLOPS = {
    "H100": 989.5e12, "H200": 989.5e12, "A100": 312e12,
    "L40S": 362e12, "4090": 165.2e12, "default": 989.5e12,
}


class TrainingDiverged(RuntimeError):
    """Loss went NaN or above 100. Upstream calls exit(1); a notebook needs an exception."""


@dataclass
class ExpConfig:
    """
    Every knob, in one place. The printed dict IS what the report cites - if a number
    is not in here, it is not reproducible.

    Defaults are upstream train.py's. Change them per experiment, one axis at a time.
    """
    # --- round wiring (set these; they are not hyperparameters to tune) ---
    round_idx: int = 1
    seed: int = 0
    # Run length, in tokens. None -> one pass over this round's data (the default, and
    # what makes compute scale with data: R1 1x, R2 2x, R3 4x). There is no clock; a run
    # takes as long as one pass costs on this machine. Set it explicitly only to study
    # run length itself, and then say so in the report.
    train_tokens: float = None

    # --- model architecture ---
    depth: int = 8                   # number of transformer layers
    aspect_ratio: int = 64           # model_dim = depth * aspect_ratio, rounded to head_dim
    head_dim: int = 128              # target attention head dimension
    window_pattern: str = "SSSL"     # L = full context, S = half context

    # --- optimization ---
    device_batch_size: int = 128     # per-device micro-batch (lower this if OOM)
    total_batch_size: int = 2 ** 19  # tokens per optimizer step
    embedding_lr: float = 0.6
    unembedding_lr: float = 0.004
    matrix_lr: float = 0.04          # Muon
    scalar_lr: float = 0.5
    weight_decay: float = 0.2
    adam_betas: tuple = (0.8, 0.95)
    warmup_ratio: float = 0.0
    warmdown_ratio: float = 0.5
    final_lr_frac: float = 0.0

    # --- bookkeeping ---
    gpu: str = "default"             # key into PEAK_FLOPS, for the MFU figure only
    notes: str = ""

    def __post_init__(self):
        if self.train_tokens is None:
            self.train_tokens = float(round_tokens(self.round_idx))

    def model_dim(self):
        base = self.depth * self.aspect_ratio
        return ((base + self.head_dim - 1) // self.head_dim) * self.head_dim

    def grad_accum_steps(self):
        per_fwdbwd = self.device_batch_size * prepare.MAX_SEQ_LEN
        assert self.total_batch_size % per_fwdbwd == 0, (
            f"total_batch_size {self.total_batch_size} must be divisible by "
            f"device_batch_size * MAX_SEQ_LEN = {per_fwdbwd}")
        return self.total_batch_size // per_fwdbwd


def build_model_config(cfg, vocab_size):
    model_dim = cfg.model_dim()
    num_heads = model_dim // cfg.head_dim
    return GPTConfig(
        sequence_len=prepare.MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=cfg.depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=cfg.window_pattern,
    )


def run(cfg, outdir=None, log_every=1, quiet=False):
    """
    One training run + eval. Returns a metrics dict.

    Writes `metrics.json` and `loss.png` into `outdir` when given. Appends to
    metrics.json rather than overwriting, so run_seeds leaves one object per seed.

    This is where the harness assumes `prepare_round()` has already been called. If it
    has not, DATA_DIR still points at the full cache and the run silently uses the wrong
    data scale - so it asserts instead.
    """
    assert os.path.basename(prepare.DATA_DIR).startswith("data_r"), (
        f"prepare.DATA_DIR is {prepare.DATA_DIR!r} - call prepare_round(ROUND) first, "
        "or this run would use the wrong data scale")

    t_start = time.time()
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")
    torch.cuda.reset_peak_memory_stats()   # so run_seeds reports per-run VRAM, not cumulative
    device = torch.device("cuda")
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    peak_flops = PEAK_FLOPS.get(cfg.gpu, PEAK_FLOPS["default"])

    tokenizer = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()
    model_config = build_model_config(cfg, vocab_size)
    if not quiet:
        print(f"Vocab size: {vocab_size:,}")
        print(f"Model config: {asdict(model_config)}")

    with torch.device("meta"):
        model = GPT(model_config)
    model.to_empty(device=device)
    model.init_weights()

    param_counts = model.num_scaling_params()
    num_params = param_counts["total"]
    num_flops_per_token = model.estimate_flops()
    if not quiet:
        for k, v in param_counts.items():
            print(f"  {k:24s}: {v:,}")
        print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

    grad_accum_steps = cfg.grad_accum_steps()
    optimizer = model.setup_optimizer(
        unembedding_lr=cfg.unembedding_lr, embedding_lr=cfg.embedding_lr,
        scalar_lr=cfg.scalar_lr, adam_betas=cfg.adam_betas,
        matrix_lr=cfg.matrix_lr, weight_decay=cfg.weight_decay,
    )
    model = torch.compile(model, dynamic=False)

    train_loader = make_dataloader(tokenizer, cfg.device_batch_size, prepare.MAX_SEQ_LEN, "train")
    x, y, epoch = next(train_loader)   # prefetch
    planned_steps = max(1, int(cfg.train_tokens // cfg.total_batch_size))
    if not quiet:
        print(f"Train tokens: {cfg.train_tokens/1e6:.1f}M (~{planned_steps} steps, no time limit) "
              f"| grad accum steps: {grad_accum_steps}")

    # --- schedules: upstream's shapes, but progress is measured in tokens, not seconds.
    # Upstream divides elapsed time by TIME_BUDGET, which makes the LR schedule depend on
    # how fast the machine is. Token progress makes a run bit-reproducible across
    # hardware, and it is why there is no clock anywhere below. ---
    def get_lr_multiplier(progress):
        if progress < cfg.warmup_ratio:
            return progress / cfg.warmup_ratio if cfg.warmup_ratio > 0 else 1.0
        elif progress < 1.0 - cfg.warmdown_ratio:
            return 1.0
        cooldown = (1.0 - progress) / cfg.warmdown_ratio
        return cooldown * 1.0 + (1 - cooldown) * cfg.final_lr_frac

    def get_muon_momentum(step):
        frac = min(step / 300, 1)
        return (1 - frac) * 0.85 + frac * 0.95

    # --- training loop ---
    t_start_training = time.time()
    history = []        # (step, wall_clock, smoothed_loss, lrm, tok_per_s) - the loss curve
    smooth_train_loss = 0.0
    total_training_time = 0.0   # measured, for throughput/MFU only. Never a stop condition.
    tokens_seen = 0
    step = 0
    gc_was_enabled = gc.isenabled()

    try:
        while True:
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(grad_accum_steps):
                with autocast_ctx:
                    loss = model(x, y)
                train_loss = loss.detach()
                (loss / grad_accum_steps).backward()
                x, y, epoch = next(train_loader)

            tokens_seen += cfg.total_batch_size
            progress = min(tokens_seen / cfg.train_tokens, 1.0)
            lrm = get_lr_multiplier(progress)
            muon_momentum = get_muon_momentum(step)
            muon_weight_decay = cfg.weight_decay * (1 - progress)
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * lrm
                if group["kind"] == "muon":
                    group["momentum"] = muon_momentum
                    group["weight_decay"] = muon_weight_decay
            optimizer.step()
            model.zero_grad(set_to_none=True)

            train_loss_f = train_loss.item()
            if math.isnan(train_loss_f) or train_loss_f > 100:
                raise TrainingDiverged(
                    f"loss={train_loss_f} at step {step} (seed {cfg.seed}). "
                    "Record as a crash; do not retry with the same config.")

            torch.cuda.synchronize()
            dt = time.time() - t0
            if step > 10:
                total_training_time += dt   # upstream: skip compile/warmup steps

            ema_beta = 0.9
            smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
            debiased = smooth_train_loss / (1 - ema_beta ** (step + 1))
            tok_per_sec = int(cfg.total_batch_size / dt)
            mfu = 100 * num_flops_per_token * cfg.total_batch_size / dt / peak_flops
            if step % log_every == 0:
                history.append((step, total_training_time, debiased, lrm, tok_per_sec))
            if not quiet:
                left = (planned_steps - step) * dt
                print(f"\rstep {step:05d}/{planned_steps} ({100 * progress:.1f}%) | "
                      f"loss: {debiased:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.0f}ms | "
                      f"tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | "
                      f"eta: {left/60:.1f}min    ", end="", flush=True)

            if step == 0:
                gc.collect(); gc.freeze(); gc.disable()
            elif (step + 1) % 5000 == 0:
                gc.collect()
            step += 1
            if tokens_seen >= cfg.train_tokens:
                break
    finally:
        # run_seeds calls this repeatedly in one process; leaving GC disabled would
        # distort every later run's timings.
        gc.unfreeze()
        if gc_was_enabled:
            gc.enable()

    if not quiet:
        print()

    # --- eval: upstream metric, untouched ---
    model.eval()
    with autocast_ctx:
        val_bpb = evaluate_bpb(model, tokenizer, cfg.device_batch_size)

    t_end = time.time()
    total_tokens = tokens_seen
    steady_mfu = (100 * num_flops_per_token * cfg.total_batch_size * (step - 10)
                  / total_training_time / peak_flops) if total_training_time > 0 else 0.0
    metrics = {
        "val_bpb": float(val_bpb),
        "training_seconds": round(total_training_time, 1),
        "total_seconds": round(t_end - t_start, 1),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1),
        "mfu_percent": round(steady_mfu, 2),
        "total_tokens_M": round(total_tokens / 1e6, 1),
        "tok_per_s": int(total_tokens / total_training_time) if total_training_time else 0,
        "num_steps": step,
        "num_params_M": round(num_params / 1e6, 1),
        "depth": cfg.depth,
        "epoch": float(epoch),
        "train_tokens_M": round(cfg.train_tokens / 1e6, 1),
        "final_train_loss": round(history[-1][2], 6) if history else None,
        "config": asdict(cfg),
    }

    # Upstream summary block, same keys and order, so `grep "^val_bpb:" run.log` works.
    print("---")
    for k in ("val_bpb", "training_seconds", "total_seconds", "peak_vram_mb", "mfu_percent",
              "total_tokens_M", "num_steps", "num_params_M", "depth"):
        v = metrics[k]
        print(f"{k + ':':18s}{v:.6f}" if k == "val_bpb" else f"{k + ':':18s}{v}")
    print(f"{'epoch:':18s}{metrics['epoch']:.3f}   # one pass by design; ~1.0 expected")
    if metrics["epoch"] > 1.3 or metrics["epoch"] < 0.7:
        # train_tokens came from SHARD_TOKENS, an estimate. The dataloader's own epoch
        # counter is ground truth, so it tells us what the estimate should have been.
        n_shards = len(round_shards(cfg.round_idx))
        actual = cfg.train_tokens / metrics["epoch"] / n_shards
        print(f"WARNING: epoch {metrics['epoch']:.2f}, not ~1.0. SHARD_TOKENS in "
              f"_shared/data.py is wrong for this corpus; set it to ~{int(actual):,} "
              "so a run is one pass. Results so far are still internally comparable "
              "(every run in the round used the same train_tokens), but the round is "
              "not the data scale it claims to be.")

    if outdir:
        _write_artifacts(outdir, metrics, history, cfg)
    del model, optimizer, train_loader
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def _write_artifacts(outdir, metrics, history, cfg):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "metrics.json")
    runs = []
    if os.path.exists(path):
        with open(path) as f:
            runs = json.load(f)
        if isinstance(runs, dict):
            runs = [runs]
    runs.append(metrics)
    with open(path, "w") as f:
        json.dump(runs, f, indent=2)

    # loss.png is cited as evidence in the report (section 5), so it is not optional:
    # it is how a reviewer tells "the change helped" from "the change got more steps".
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        steps, wall, losses, lrms, _ = zip(*history)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(steps, losses)
        axes[0].set_xlabel("step"); axes[0].set_ylabel("smoothed train loss")
        axes[1].plot(wall, losses)
        axes[1].set_xlabel("training seconds"); axes[1].set_ylabel("smoothed train loss")
        for ax in axes:
            ax.grid(alpha=0.3)
        fig.suptitle(f"seed {cfg.seed} | val_bpb {metrics['val_bpb']:.6f} | "
                     f"{metrics['num_steps']} steps")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "loss.png"), dpi=110)
        plt.close(fig)
    except Exception as e:                      # a missing curve must not void a valid run
        print(f"(loss.png not written: {e})")


def run_seeds(cfg, seeds=(0, 1, 2), outdir=None, quiet=True):
    """
    Same config, N seeds. Returns (mean, stdev, [metrics...]).

    This is how sigma is measured for a round (3 seeds of the round baseline), and how a
    `retest` verdict is resolved. A single-seed result never crosses a round boundary -
    see research-program.md section 4.
    """
    runs = [run(replace(cfg, seed=s), outdir=outdir, quiet=quiet) for s in seeds]
    bpbs = [r["val_bpb"] for r in runs]
    mean = statistics.fmean(bpbs)
    sd = statistics.stdev(bpbs) if len(bpbs) > 1 else 0.0
    print(f"\n{len(seeds)} seeds: " + " / ".join(f"{b:.6f}" for b in bpbs))
    print(f"mean {mean:.6f}  stdev {sd:.6f}  spread {max(bpbs) - min(bpbs):.6f}")
    return mean, sd, runs
