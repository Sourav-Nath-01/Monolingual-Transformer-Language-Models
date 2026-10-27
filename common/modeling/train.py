"""Pretraining loop: AdamW, cosine schedule, resume-capable checkpointing.

Written for Kaggle, and that shapes almost every decision here. A Kaggle
notebook session is capped at roughly 12 hours and can be evicted earlier, so
the run is built to be **interrupted and resumed as a matter of course**, not as
error handling. Spec section 2.2 requires exactly this: every checkpoint carries
model weights, optimiser state, scheduler state, current step and the config,
and a run that cannot resume after interruption loses marks.

Key decisions:

**Steps, not epochs.** At 590M tokens a single pass would take far longer than
the available GPU budget, so the run is defined by a step count and the data
loader samples random windows (see :mod:`common.modeling.data`). "Epoch" is not a
meaningful unit at this ratio of corpus size to compute.

**Cosine decay with linear warmup.** Warmup exists because Adam's second-moment
estimate is unreliable in the first few hundred steps -- stepping at full rate
into that noise is a well-known way to destabilise a fresh Transformer. Cosine
decay afterwards spends most of the budget at a useful rate then anneals
smoothly, which reliably beats a constant rate for a fixed step budget.

**Gradient accumulation.** The effective batch (in tokens) that matters for
stable LM training is often larger than a 16 GB GPU can hold at once.
Accumulating gradients over several micro-batches decouples the *statistical*
batch size from what fits in memory, at no cost beyond wall-clock.

**Mixed precision, preferring bfloat16 where the hardware is built for it.**
bf16 has fp32's exponent range, so it needs no loss scaling and cannot silently
underflow gradients the way fp16 can -- the better choice whenever the GPU
accelerates it. That means Ampere (sm_80) and newer. On older cards the
selection inverts: Kaggle's Turing T4 reports bf16 as *supported* yet has only
fp16 tensor cores, so bf16 falls back to emulation and loses to fp16 by several
times. There the code takes fp16 *with* a ``GradScaler``, the safe pairing.

**Two checkpoints, deliberately.** ``last.pt`` is overwritten on a fixed step
interval and exists purely so a killed session resumes near where it stopped.
``best.pt`` is written only when validation loss improves, and is what
evaluation and Phase 3 finetuning start from. Keeping them separate means a
late-training divergence cannot destroy the best model.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from common.modeling.data import TokenArray, get_batch, iter_eval_batches
from common.modeling.model import ModelConfig, TransformerLM

__all__ = ["TrainConfig", "Trainer", "main"]


@dataclass
class TrainConfig:
    """Optimisation and checkpointing settings. Saved with every checkpoint.

    Attributes:
        language: Which language this run trains; picks data and output paths.
        data_dir: Directory holding ``<lang>_{train,val}.bin``.
        out_dir: Where checkpoints and logs are written.
        max_steps: Total optimiser steps (not micro-batches).
        batch_size: Micro-batch size, what actually sits on the GPU.
        grad_accum_steps: Micro-batches per optimiser step. Effective batch is
            ``batch_size * grad_accum_steps``.
        learning_rate: Peak LR, reached at the end of warmup.
        min_lr: Floor the cosine decays to, usually ``learning_rate / 10``.
        warmup_steps: Linear ramp length.
        weight_decay: AdamW decay, applied to matrices but not to biases or
            LayerNorm gains (see :meth:`Trainer._param_groups`).
        grad_clip: Global gradient-norm clip. Guards the occasional bad batch
            from destabilising the run.
        eval_interval: Steps between validation passes.
        eval_batches: Validation batches per pass. Capped during training for
            speed; final metrics use the whole split via ``common.modeling.evaluate``.
        checkpoint_interval: Steps between ``last.pt`` writes.
        seed: Seeds torch RNG for reproducibility.
        compile_model: ``torch.compile`` the model. Big speedup on modern GPUs,
            but adds a slow first step and is unavailable on older torch.
    """

    language: str
    data_dir: str = "data/tokenized"
    out_dir: str = ""

    max_steps: int = 6_000
    batch_size: int = 24
    grad_accum_steps: int = 8
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 500
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    eval_interval: int = 500
    eval_batches: int = 40
    checkpoint_interval: int = 500
    log_interval: int = 20

    seed: int = 1337
    compile_model: bool = False

    def __post_init__(self) -> None:
        self.out_dir = self.out_dir or f"{self.language}/model"

    @property
    def effective_batch_tokens(self) -> int:
        """Tokens per optimiser step, at a 512 block size."""
        return self.batch_size * self.grad_accum_steps * 512


def cosine_lr(step: int, cfg: TrainConfig) -> float:
    """Linear warmup then cosine decay to ``min_lr``.

    Returns the learning rate for ``step``. Kept a pure function of the step so
    resuming is exact: the schedule is recomputed from the restored step rather
    than depending on how many times a scheduler object was advanced.
    """
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / max(1, cfg.warmup_steps)
    if step >= cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


class Trainer:
    """Owns the model, optimiser, data and checkpoint lifecycle for one run."""

    def __init__(self, model_cfg: ModelConfig, train_cfg: TrainConfig,
                 device: str | None = None) -> None:
        self.mcfg = model_cfg
        self.tcfg = train_cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(train_cfg.seed)

        self.out_dir = Path(train_cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        d = Path(train_cfg.data_dir)
        self.train_data = TokenArray(d / f"{train_cfg.language}_train.bin")
        self.val_data = TokenArray(d / f"{train_cfg.language}_val.bin")

        # A vocabulary mismatch between tokeniser and model would train happily
        # and produce garbage, so it is checked rather than assumed.
        if self.train_data.vocab_size and self.train_data.vocab_size != model_cfg.vocab_size:
            raise ValueError(
                f"data was tokenised with vocab {self.train_data.vocab_size} but "
                f"model config says {model_cfg.vocab_size}"
            )

        self.model = TransformerLM(model_cfg).to(self.device)
        if train_cfg.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)

        self.optimizer = torch.optim.AdamW(
            self._param_groups(train_cfg.weight_decay),
            lr=train_cfg.learning_rate,
            betas=(train_cfg.beta1, train_cfg.beta2),
        )

        # bf16 needs no loss scaling; fp16 does. Anything else runs fp32.
        #
        # Choose on compute capability, not on is_bf16_supported(). That query
        # answers "can this run bf16 at all", and returns True on a Turing T4 --
        # but Turing has fp16 tensor cores and no bf16 ones, so bf16 there is
        # emulated and runs several times slower than the fp16 path it would
        # displace. Tensor-core bf16 arrives with Ampere (sm_80), so gate on
        # that and take fp16 + GradScaler on anything older. Measured on a T4:
        # bf16 gave ~11.5k tokens/s against fp16's tensor-core path.
        self.amp_dtype = None
        if self.device.startswith("cuda"):
            major, _ = torch.cuda.get_device_capability()
            self.amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=(self.amp_dtype == torch.float16))

        self.step = 0
        self.best_val = float("inf")
        self.log_path = self.out_dir / "train_log.jsonl"

    def _param_groups(self, weight_decay: float) -> list[dict]:
        """Decay matrices, never biases or LayerNorm gains.

        Weight decay on a bias or a normalisation gain has no regularising
        interpretation -- those parameters set scale and offset, and shrinking
        them toward zero just fights the layer's purpose. The 2D/1D split is the
        standard, cheap way to express that.
        """
        decay, no_decay = [], []
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    # ---------------------------------------------------------------- checkpoints

    def save_checkpoint(self, name: str, **extra) -> Path:
        """Write a fully resume-capable checkpoint.

        Contains everything spec section 2.2 requires: weights, optimiser state,
        scheduler state (as the step, since the schedule is a pure function of
        it), the step itself, and the model/train configs. Written to ``.tmp``
        and renamed, so an eviction mid-write cannot leave a corrupt file that
        the next session would try to resume from.
        """
        path = self.out_dir / name
        tmp = path.with_suffix(".pt.tmp")
        raw = getattr(self.model, "_orig_mod", self.model)  # unwrap torch.compile
        payload = {
            "model": raw.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "step": self.step,
            "best_val": self.best_val,
            "model_config": asdict(self.mcfg),
            "train_config": asdict(self.tcfg),
            "lr_at_step": cosine_lr(self.step, self.tcfg),
            **extra,
        }
        torch.save(payload, tmp)
        tmp.replace(path)
        return path

    def load_checkpoint(self, path: Path | str) -> None:
        """Restore a run exactly, so resuming continues rather than restarts."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        raw = getattr(self.model, "_orig_mod", self.model)
        raw.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler"])
        self.step = ckpt["step"]
        self.best_val = ckpt.get("best_val", float("inf"))
        print(f"resumed from {path} at step {self.step:,} "
              f"(best val {self.best_val:.4f})")

    def maybe_resume(self) -> bool:
        """Resume from ``last.pt`` if present. Returns whether it resumed."""
        last = self.out_dir / "last.pt"
        if last.exists():
            self.load_checkpoint(last)
            return True
        return False

    # ---------------------------------------------------------------- eval / log

    @torch.no_grad()
    def evaluate(self, limit_batches: int | None = None) -> float:
        """Mean validation cross-entropy over consecutive held-out windows."""
        self.model.eval()
        limit = limit_batches if limit_batches is not None else self.tcfg.eval_batches
        total, count = 0.0, 0
        for x, y in iter_eval_batches(self.val_data, self.tcfg.batch_size,
                                      self.mcfg.block_size, self.device,
                                      limit_batches=limit):
            if self.amp_dtype:
                with torch.autocast("cuda", dtype=self.amp_dtype):
                    _, loss = self.model(x, y)
            else:
                _, loss = self.model(x, y)
            total += loss.item()
            count += 1
        self.model.train()
        return total / max(1, count)

    def log(self, record: dict) -> None:
        """Append one JSONL record. The loss curves are plotted from this file."""
        record["wall_time"] = time.time()
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    # ---------------------------------------------------------------- train loop

    def train(self) -> dict:
        """Run until ``max_steps``, checkpointing and validating on schedule."""
        cfg = self.tcfg
        self.model.train()
        t0 = time.time()
        tokens_per_step = cfg.effective_batch_tokens

        print(f"device={self.device} amp={self.amp_dtype} "
              f"params={self.model.num_parameters() if hasattr(self.model,'num_parameters') else 'n/a'}")
        print(f"effective batch = {cfg.batch_size} x {cfg.grad_accum_steps} x "
              f"{self.mcfg.block_size} = {tokens_per_step:,} tokens/step")
        print(f"steps {self.step:,} -> {cfg.max_steps:,} "
              f"({(cfg.max_steps - self.step) * tokens_per_step / 1e9:.2f}B tokens remaining)")

        while self.step < cfg.max_steps:
            lr = cosine_lr(self.step, cfg)
            for g in self.optimizer.param_groups:
                g["lr"] = lr

            self.optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            for _ in range(cfg.grad_accum_steps):
                x, y = get_batch(self.train_data, cfg.batch_size,
                                 self.mcfg.block_size, self.device)
                if self.amp_dtype:
                    with torch.autocast("cuda", dtype=self.amp_dtype):
                        _, loss = self.model(x, y)
                else:
                    _, loss = self.model(x, y)
                # Divide so the accumulated gradient is the MEAN over
                # micro-batches, matching what one large batch would give.
                loss = loss / cfg.grad_accum_steps
                self.scaler.scale(loss).backward()
                accum_loss += loss.item()

            # Unscale before clipping: clipping scaled gradients would apply the
            # threshold to the wrong magnitudes.
            if self.scaler.is_enabled():
                self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.step += 1

            if self.step % cfg.log_interval == 0:
                elapsed = time.time() - t0
                self.log({"step": self.step, "train_loss": accum_loss, "lr": lr,
                          "grad_norm": float(grad_norm), "elapsed_s": round(elapsed, 1),
                          "tokens": self.step * tokens_per_step})
                print(f"step {self.step:>6,}/{cfg.max_steps:,} "
                      f"loss {accum_loss:.4f} lr {lr:.2e} "
                      f"gnorm {float(grad_norm):.2f} "
                      f"{elapsed/60:.1f}m")

            if self.step % cfg.eval_interval == 0 or self.step == cfg.max_steps:
                val = self.evaluate()
                improved = val < self.best_val
                if improved:
                    self.best_val = val
                    self.save_checkpoint("best.pt", val_loss=val)
                self.log({"step": self.step, "val_loss": val,
                          "val_ppl": math.exp(min(val, 20)), "best": improved})
                print(f"  eval  step {self.step:,}  val_loss {val:.4f}  "
                      f"ppl {math.exp(min(val, 20)):.2f}"
                      f"{'  <- best, saved' if improved else ''}")

            if self.step % cfg.checkpoint_interval == 0:
                self.save_checkpoint("last.pt")

        self.save_checkpoint("last.pt")
        summary = {"steps": self.step, "best_val_loss": self.best_val,
                   "best_val_ppl": math.exp(min(self.best_val, 20)),
                   "total_minutes": round((time.time() - t0) / 60, 1),
                   "tokens_seen": self.step * tokens_per_step}
        (self.out_dir / "train_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\ndone: {summary}")
        return summary


def main(argv: list[str] | None = None) -> int:
    """CLI. Example::

        python -m common.modeling.train --lang hindi
        python -m common.modeling.train --lang assamese --data-dir /kaggle/input/lma-tokens
    """
    import argparse

    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--lang", required=True, choices=("hindi", "assamese"))
    p.add_argument("--data-dir", default="data/tokenized")
    p.add_argument("--out-dir", default="")
    p.add_argument("--config", default="", help="model config JSON; defaults to <lang>/configs/model.json")
    p.add_argument("--max-steps", type=int, default=6_000)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--checkpoint-interval", type=int, default=500)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--no-resume", action="store_true",
                   help="ignore last.pt and start from scratch")
    args = p.parse_args(argv)

    mcfg = ModelConfig.load(args.config or f"{args.lang}/configs/model.json")
    tcfg = TrainConfig(
        language=args.lang, data_dir=args.data_dir, out_dir=args.out_dir,
        max_steps=args.max_steps, batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum, learning_rate=args.lr,
        min_lr=args.lr / 10, warmup_steps=args.warmup,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
        compile_model=args.compile,
    )

    trainer = Trainer(mcfg, tcfg)
    rep = trainer.model.parameter_report() if hasattr(trainer.model, "parameter_report") else {}
    if rep:
        print(f"parameters: {rep['total_trainable']:,} "
              f"(embeddings {rep['embedding_share_of_total']:.1%})")
        (Path(tcfg.out_dir) / "parameter_report.json").write_text(
            json.dumps(rep, indent=2), encoding="utf-8")

    if not args.no_resume:
        trainer.maybe_resume()
    trainer.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
