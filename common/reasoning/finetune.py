"""Finetune a pretrained model on its own language's reasoning data.

Spec section 3.1's protocol, point by point: start from that language's own
pretrained checkpoint, keep its tokenizer and vocabulary fixed, document the
hyperparameters, and save checkpoints in the same resume-capable format as
pretraining.

Three decisions worth stating:

**Model weights are restored; the optimizer is not.** ``best.pt`` carries
AdamW's moment estimates from the end of a 6,000-step cosine decay, where the
learning rate had fallen to 3e-5 and the moments describe *that* regime.
Carrying them into a new run with a new schedule would fight the fresh warmup.
Only ``model`` is loaded, which is also what "start from the pretrained
checkpoint" means -- the weights are the pretrained artifact, the optimizer
state is an artifact of how they got there.

**A much lower peak learning rate.** Pretraining used 3e-4 on randomly
initialised weights. Finetuning starts from a converged model on a narrow task,
where the same rate would overwrite general language knowledge before the task
is learned -- catastrophic forgetting in its most literal form. 5e-5 is a sixth
of it.

**Selection on exact-match accuracy, not validation loss.** Loss and accuracy do
not have to move together: the model can grow more confident about answers it
already gets right, lowering loss while accuracy is flat. Accuracy is what the
report actually claims, so it is what ``best.pt`` is chosen on.

Checkpoints carry the same payload Phase 2 wrote -- weights, optimizer state,
scheduler position, step, and both configs -- so an interrupted Kaggle session
resumes exactly where it stopped.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from common.modeling.model import ModelConfig, TransformerLM
from common.reasoning.data import (IGNORE, ReasoningBatches, encode_examples,
                                   load_jsonl)

__all__ = ["FinetuneConfig", "Finetuner", "ANSWER_PREFIX", "main"]

ANSWER_PREFIX = {"hindi": "उत्तर:", "assamese": "উত্তৰ:"}


@dataclass
class FinetuneConfig:
    """Every hyperparameter the finetuning run depends on.

    Written to ``config.json`` beside the checkpoints so the run is reproducible
    from the README without unpickling a 300 MB file.
    """

    language: str
    pretrained: str                      # path to Phase 2's best.pt
    data_dir: str = ""                   # defaults to <language>/train
    out_dir: str = ""                    # defaults to <language>/train/finetuned
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 5e-5
    min_lr: float = 5e-6
    warmup_frac: float = 0.05            # of total steps
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_every: int = 250
    eval_batches: int = 40
    checkpoint_every: int = 250
    log_every: int = 50
    seed: int = 3407

    def __post_init__(self) -> None:
        self.data_dir = self.data_dir or f"{self.language}/train"
        self.out_dir = self.out_dir or f"{self.language}/train/finetuned"


def cosine_lr(step: int, total: int, cfg: FinetuneConfig) -> float:
    """Linear warmup then cosine decay, as a pure function of the step.

    Being a function of ``step`` alone -- never of accumulated state -- is what
    makes resume exact: a run restarted at step 900 computes the same rate the
    uninterrupted run would have used there.
    """
    warmup = max(1, int(total * cfg.warmup_frac))
    if step < warmup:
        return cfg.learning_rate * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return cfg.min_lr + (cfg.learning_rate - cfg.min_lr) * cosine


class Finetuner:
    """Runs reasoning finetuning for one language, resumably."""

    def __init__(self, cfg: FinetuneConfig, device: str | None = None) -> None:
        import sentencepiece as spm

        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(cfg.seed)

        ckpt = torch.load(cfg.pretrained, map_location="cpu", weights_only=False)
        self.mcfg = ModelConfig(**ckpt["model_config"])
        self.model = TransformerLM(self.mcfg).to(self.device)
        self.model.load_state_dict(ckpt["model"], strict=True)
        self.pretrained_step = ckpt.get("step")

        self.sp = spm.SentencePieceProcessor(
            model_file=f"{cfg.language}/tokenizer/"
                       f"{cfg.language}_bpe_{self.mcfg.vocab_size}.model")
        self.prefix = ANSWER_PREFIX[cfg.language]

        self.out_dir = Path(cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.out_dir / "finetune_log.jsonl"

        self.train_ex = self._load("train")
        self.val_ex = self._load("val")
        self.steps_per_epoch = (len(self.train_ex) + cfg.batch_size - 1) // cfg.batch_size
        self.total_steps = self.steps_per_epoch * cfg.epochs

        decay = [p for p in self.model.parameters() if p.dim() >= 2]
        no_decay = [p for p in self.model.parameters() if p.dim() < 2]
        self.optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": cfg.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2))

        # Same rule as pretraining: bf16 only where the hardware accelerates it.
        self.amp_dtype = None
        if self.device.startswith("cuda"):
            major, _ = torch.cuda.get_device_capability()
            self.amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=(self.amp_dtype == torch.float16))

        self.step = 0
        self.best_acc = -1.0

    def _load(self, split: str):
        rows = load_jsonl(Path(self.cfg.data_dir) / f"reasoning_{split}.jsonl",
                          self.prefix)
        return encode_examples(rows, self.sp, self.mcfg.block_size)

    # -- checkpointing -------------------------------------------------------

    def _save(self, name: str, val_loss: float, acc: float) -> None:
        """Write a checkpoint atomically, in Phase 2's payload format."""
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "step": self.step,
            "best_acc": self.best_acc,
            "val_loss": val_loss,
            "exact_match": acc,
            "model_config": asdict(self.mcfg),
            "finetune_config": asdict(self.cfg),
            "lr_at_step": cosine_lr(self.step, self.total_steps, self.cfg),
            "pretrained_from": self.cfg.pretrained,
            "pretrained_step": self.pretrained_step,
        }
        path = self.out_dir / name
        tmp = path.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        tmp.replace(path)

    def maybe_resume(self) -> bool:
        """Continue from ``last.pt`` if a previous run left one behind."""
        path = self.out_dir / "last.pt"
        if not path.exists():
            return False
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model"])
        self.optimizer.load_state_dict(ck["optimizer"])
        self.scaler.load_state_dict(ck["scaler"])
        self.step = ck["step"]
        self.best_acc = ck.get("best_acc", -1.0)
        print(f"resumed from {path} at step {self.step} (best EM {self.best_acc:.4f})")
        return True

    # -- evaluation ----------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, limit_batches: int | None = None) -> tuple[float, float]:
        """Return (mean loss on answer tokens, exact-match accuracy)."""
        self.model.eval()
        batches = ReasoningBatches(self.val_ex, self.cfg.batch_size,
                                   self.sp.pad_id(), self.device, shuffle=False)
        total_loss, n_loss, correct, n = 0.0, 0, 0, 0
        for i, (x, y) in enumerate(batches):
            if limit_batches and i >= limit_batches:
                break
            if self.amp_dtype:
                with torch.autocast("cuda", dtype=self.amp_dtype):
                    logits, loss = self.model(x, y)
            else:
                logits, loss = self.model(x, y)
            total_loss += loss.item()
            n_loss += 1
            # Exact match on the supervised positions: the model's argmax must
            # reproduce every answer token. Teacher-forced, so this measures the
            # same quantity the loss does but on the scale the report quotes.
            pred = logits.argmax(-1)
            mask = y != IGNORE
            hit = ((pred == y) | ~mask).all(dim=1)
            correct += hit.sum().item()
            n += x.size(0)
        self.model.train()
        return total_loss / max(1, n_loss), correct / max(1, n)

    # -- training ------------------------------------------------------------

    def train(self) -> dict:
        """Run to ``epochs``, evaluating, logging and checkpointing as it goes."""
        cfg = self.cfg
        self.model.train()
        t0 = time.time()
        print(f"device={self.device} amp={self.amp_dtype} "
              f"params={self.model.num_parameters():,}")
        print(f"train {len(self.train_ex):,} examples | val {len(self.val_ex):,} | "
              f"{self.steps_per_epoch:,} steps/epoch x {cfg.epochs} = "
              f"{self.total_steps:,} steps")
        print(f"steps {self.step} -> {self.total_steps}")

        while self.step < self.total_steps:
            epoch = self.step // self.steps_per_epoch
            batches = ReasoningBatches(self.train_ex, cfg.batch_size,
                                       self.sp.pad_id(), self.device,
                                       shuffle=True, seed=cfg.seed + epoch)
            for x, y in batches:
                if self.step >= self.total_steps:
                    break
                lr = cosine_lr(self.step, self.total_steps, cfg)
                for g in self.optimizer.param_groups:
                    g["lr"] = lr

                if self.amp_dtype:
                    with torch.autocast("cuda", dtype=self.amp_dtype):
                        _, loss = self.model(x, y)
                else:
                    _, loss = self.model(x, y)

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                gnorm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.step += 1

                if self.step % cfg.log_every == 0:
                    rec = {"step": self.step, "train_loss": loss.item(), "lr": lr,
                           "grad_norm": float(gnorm), "epoch": epoch,
                           "elapsed_s": round(time.time() - t0, 1)}
                    self._append(rec)
                    print(f"step {self.step:6,}/{self.total_steps:,} "
                          f"loss {loss.item():.4f} lr {lr:.2e} "
                          f"gnorm {float(gnorm):.2f} "
                          f"{(time.time()-t0)/60:.1f}m")

                if self.step % cfg.eval_every == 0 or self.step == self.total_steps:
                    vloss, acc = self.evaluate(cfg.eval_batches)
                    best = acc > self.best_acc
                    if best:
                        self.best_acc = acc
                    self._append({"step": self.step, "val_loss": vloss,
                                  "exact_match": acc, "best": best})
                    print(f"  eval  step {self.step:,}  val_loss {vloss:.4f}  "
                          f"exact-match {acc:.4f}" + ("  <- best, saved" if best else ""))
                    if best:
                        self._save("best.pt", vloss, acc)

                if self.step % cfg.checkpoint_every == 0:
                    vloss, acc = self.evaluate(4)
                    self._save("last.pt", vloss, acc)

        vloss, acc = self.evaluate(cfg.eval_batches)
        self._save("last.pt", vloss, acc)
        summary = {"steps": self.step, "epochs": cfg.epochs,
                   "best_exact_match": self.best_acc,
                   "final_val_loss": vloss, "final_exact_match": acc,
                   "total_minutes": round((time.time() - t0) / 60, 1),
                   "train_examples": len(self.train_ex)}
        (self.out_dir / "finetune_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\ndone: {summary}")
        return summary

    def _append(self, record: dict) -> None:
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")


def main(argv: list[str] | None = None) -> int:
    """CLI. Example::

        python -m common.reasoning.finetune --lang hindi \
            --pretrained hindi/model/best.pt
    """
    import argparse
    from common.io import enable_utf8_console

    enable_utf8_console()
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--lang", required=True, choices=("hindi", "assamese"))
    p.add_argument("--pretrained", required=True)
    p.add_argument("--data-dir", default="")
    p.add_argument("--out-dir", default="")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-5)
    args = p.parse_args(argv)

    cfg = FinetuneConfig(language=args.lang, pretrained=args.pretrained,
                         data_dir=args.data_dir, out_dir=args.out_dir,
                         epochs=args.epochs, batch_size=args.batch_size,
                         learning_rate=args.lr)
    ft = Finetuner(cfg)
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.out_dir) / "config.json").write_text(
        json.dumps({"model": asdict(ft.mcfg), "finetune": asdict(cfg)}, indent=2),
        encoding="utf-8")
    ft.maybe_resume()
    ft.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
