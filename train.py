"""Training script for the PNAS splicing model."""

from __future__ import annotations

import argparse
import logging
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm.auto import tqdm

from model import PNASModel

logger = logging.getLogger(__name__)


# ── Dataset ───────────────────────────────────────────────────────────────────

class PSIDataset(Dataset):
    def __init__(self, x_seq, x_struct, x_wobble, y):
        self.x_seq    = torch.as_tensor(x_seq)
        self.x_struct = torch.as_tensor(x_struct)
        self.x_wobble = torch.as_tensor(x_wobble)
        self.y        = torch.as_tensor(y, dtype=torch.float32)

        n = len(self.y)
        if not (len(self.x_seq) == len(self.x_struct) == len(self.x_wobble) == n):
            raise ValueError("All inputs and labels must have the same number of samples.")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_seq[idx], self.x_struct[idx], self.x_wobble[idx], self.y[idx]


# ── Loss and metrics ──────────────────────────────────────────────────────────

def kl_divergence_from_logits(logits, targets):
    """KL divergence between two Bernoullis: one from logits, one from targets."""
    return (
        F.binary_cross_entropy_with_logits(logits, targets)
        - F.binary_cross_entropy(targets, targets)
    )


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


# ── Per-epoch loops ───────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, loss_fn, device) -> dict:
    """One forward+backward pass over ``loader``. Returns loss and RMSE."""
    model.train()
    total_loss = 0.0
    pred_list, target_list = [], []

    pbar = tqdm(loader, desc="  train", leave=False, unit="batch")
    for seq, struct, wobble, y in pbar:
        seq    = seq.to(device, non_blocking=True)
        struct = struct.to(device, non_blocking=True)
        wobble = wobble.to(device, non_blocking=True)
        y      = y.to(device, dtype=torch.float32, non_blocking=True)

        optimizer.zero_grad()
        logits     = model(seq, struct, wobble, return_logits=True)
        pred_probs = torch.sigmoid(logits)
        loss       = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        pred_list.append(pred_probs.detach())
        target_list.append(y.detach())

        pbar.set_postfix(batch_loss=f"{loss.item():.5f}")

    preds   = torch.cat(pred_list)
    targets = torch.cat(target_list)
    return {"loss": total_loss / len(loader.dataset), "rmse": rmse(preds, targets)}


def eval_epoch(model, loader, loss_fn, device) -> dict:
    """No-grad evaluation pass over ``loader``. Returns loss and RMSE."""
    model.eval()
    total_loss = 0.0
    pred_list, target_list = [], []

    pbar = tqdm(loader, desc="  eval ", leave=False, unit="batch")
    with torch.no_grad():
        for seq, struct, wobble, y in pbar:
            seq    = seq.to(device, non_blocking=True)
            struct = struct.to(device, non_blocking=True)
            wobble = wobble.to(device, non_blocking=True)
            y      = y.to(device, dtype=torch.float32, non_blocking=True)

            logits     = model(seq, struct, wobble, return_logits=True)
            pred_probs = torch.sigmoid(logits)
            loss       = loss_fn(logits, y)

            total_loss += loss.item() * y.size(0)
            pred_list.append(pred_probs)
            target_list.append(y)

            pbar.set_postfix(batch_loss=f"{loss.item():.5f}")

    preds   = torch.cat(pred_list)
    targets = torch.cat(target_list)
    return {"loss": total_loss / len(loader.dataset), "rmse": rmse(preds, targets)}


# ── Training loop ─────────────────────────────────────────────────────────────

def train(model, train_loader, val_loader, hparams: dict) -> dict:
    """Outer training loop with early stopping and checkpointing.

    Args:
        model: PNASModel instance.
        train_loader: DataLoader for training examples.
        val_loader: DataLoader for validation examples.
        hparams: Dict with keys: device, num_epochs, patience, checkpoint_dir,
                 lr, weight_decay, loss_fn.

    Returns:
        Dict with keys: history (list of per-epoch dicts), best_val_loss (float),
        checkpoint_path (str).
    """
    device        = hparams["device"]
    num_epochs    = hparams["num_epochs"]
    patience      = hparams["patience"]
    checkpoint_dir = hparams["checkpoint_dir"]
    lr            = hparams["lr"]
    weight_decay  = hparams.get("weight_decay", 0.0)
    loss_fn       = hparams.get("loss_fn", kl_divergence_from_logits)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    os.makedirs(checkpoint_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(checkpoint_dir, f"best_model_{timestamp}.pt")

    logger.info("=== Training configuration ===")
    logger.info(f"  Device:          {device}")
    logger.info(f"  Max epochs:      {num_epochs}")
    logger.info(f"  Patience:        {patience}")
    logger.info(f"  LR:              {lr}")
    logger.info(f"  Weight decay:    {weight_decay}")
    logger.info(f"  Train batches:   {len(train_loader)}")
    logger.info(f"  Val batches:     {len(val_loader)}")
    logger.info(f"  Checkpoint path: {checkpoint_path}")

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, num_epochs + 1):
        logger.info(f"--- Epoch {epoch}/{num_epochs} ---")

        train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics   = eval_epoch(model, val_loader, loss_fn, device)

        record = {
            "epoch":      epoch,
            "train_loss": train_metrics["loss"],
            "train_rmse": train_metrics["rmse"],
            "val_loss":   val_metrics["loss"],
            "val_rmse":   val_metrics["rmse"],
        }
        history.append(record)

        logger.info(
            f"  Train — loss: {train_metrics['loss']:.6f}  rmse: {train_metrics['rmse']:.6f}"
        )
        logger.info(
            f"  Val   — loss: {val_metrics['loss']:.6f}  rmse: {val_metrics['rmse']:.6f}"
        )

        if val_metrics["loss"] < best_val_loss:
            prev_best = best_val_loss
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch":                epoch,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss":        best_val_loss,
                    "history":              history,
                    "hparams":              {k: str(v) for k, v in hparams.items()},
                },
                checkpoint_path,
            )
            logger.info(
                f"  Val loss improved: {prev_best:.6f} -> {best_val_loss:.6f}"
                f" — checkpoint saved to {checkpoint_path}"
            )
        else:
            epochs_without_improvement += 1
            logger.info(
                f"  No improvement: {epochs_without_improvement}/{patience} "
                f"(best val loss so far: {best_val_loss:.6f})"
            )
            if epochs_without_improvement >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs.")
                break

    logger.info(f"=== Training complete — best val loss: {best_val_loss:.6f} ===")
    return {
        "history":         history,
        "best_val_loss":   best_val_loss,
        "checkpoint_path": checkpoint_path,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the PNAS splicing model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("data")
    data.add_argument(
        "--train-npz", required=True,
        help="Training .npz produced by prepare_dataset.py.",
    )
    data.add_argument(
        "--test-npz", default=None,
        help="Optional held-out test .npz; evaluated once after training using the best checkpoint.",
    )
    data.add_argument(
        "--val-split", type=float, default=0.1,
        help="Fraction of training data held out for validation.",
    )

    mdl = parser.add_argument_group("model")
    mdl.add_argument(
        "--input-length", type=int, default=90,
        help="Input sequence length passed to PNASModel.",
    )
    mdl.add_argument(
        "--no-batchnorm", action="store_true",
        help="Disable BatchNorm in ResidualTuner (replace with nn.Identity).",
    )
    mdl.add_argument(
        "--checkpoint", default=None, metavar="PATH",
        help=(
            "Warm-start from this checkpoint. Supports partial checkpoints "
            "(e.g. seq filters only, seq+struct, or full model). "
            "Missing parameters are left at random initialization."
        ),
    )

    opt = parser.add_argument_group("optimization")
    opt.add_argument("--batch-size",    type=int,   default=64)
    opt.add_argument("--epochs",        type=int,   default=100)
    opt.add_argument("--lr",            type=float, default=1e-3)
    opt.add_argument("--weight-decay",  type=float, default=0.0)
    opt.add_argument(
        "--patience", type=int, default=10,
        help="Early stopping: epochs without val loss improvement before halting.",
    )

    run = parser.add_argument_group("runtime")
    run.add_argument(
        "--checkpoint-dir", default="./checkpoints",
        help="Directory for saving best-model checkpoints.",
    )
    run.add_argument(
        "--device", default=None, metavar="DEV",
        help="Torch device string (e.g. 'cpu', 'cuda', 'cuda:1'). Auto-detects if omitted.",
    )
    run.add_argument("--seed", type=int, default=42)

    return parser


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Reproducibility ───────────────────────────────────────────────────────
    logger.info(f"Random seed: {args.seed}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Load training data ────────────────────────────────────────────────────
    logger.info(f"Loading training data: {args.train_npz}")
    train_npz = np.load(args.train_npz)
    x_seq     = train_npz["seq_oh"]
    x_struct  = train_npz["struct_oh"]
    x_wobble  = train_npz["wobbles"]
    y         = train_npz["metadata_PSI"].astype(np.float32)
    logger.info(
        f"  Loaded — examples: {len(y):,}, "
        f"seq: {x_seq.shape}, struct: {x_struct.shape}, wobble: {x_wobble.shape}"
    )

    # ── Dataset and split ─────────────────────────────────────────────────────
    dataset = PSIDataset(x_seq, x_struct, x_wobble, y)
    n_total = len(dataset)
    n_val   = int(args.val_split * n_total)
    n_train = n_total - n_val
    if n_train == 0 or n_val == 0:
        raise ValueError(f"Dataset too small for val_split={args.val_split}.")
    logger.info(
        f"Split (seed={args.seed}) — "
        f"train: {n_train:,} ({1 - args.val_split:.0%}), "
        f"val: {n_val:,} ({args.val_split:.0%})"
    )

    train_dataset, val_dataset = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,  pin_memory=pin,
    )
    val_loader = DataLoader(
        val_dataset,   batch_size=args.batch_size, shuffle=False, pin_memory=pin,
    )
    logger.info(
        f"DataLoaders — train: {len(train_loader)} batches, "
        f"val: {len(val_loader)} batches (batch_size={args.batch_size})"
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    use_batchnorm = not args.no_batchnorm
    logger.info(
        f"Instantiating PNASModel — input_length={args.input_length}, "
        f"use_batchnorm={use_batchnorm}"
    )
    model = PNASModel(input_length=args.input_length, use_batchnorm=use_batchnorm)

    # ── Warm-start from checkpoint ────────────────────────────────────────────
    if args.checkpoint is not None:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        raw        = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state_dict = raw.get("model_state_dict", raw)
        model.load_partial_state_dict(state_dict)
    else:
        logger.info("No checkpoint provided — training from random initialization.")

    # ── Train ─────────────────────────────────────────────────────────────────
    hparams = {
        "device":         device,
        "num_epochs":     args.epochs,
        "patience":       args.patience,
        "checkpoint_dir": args.checkpoint_dir,
        "lr":             args.lr,
        "weight_decay":   args.weight_decay,
        "loss_fn":        kl_divergence_from_logits,
    }
    results = train(model, train_loader, val_loader, hparams)

    # ── Optional test evaluation ──────────────────────────────────────────────
    if args.test_npz is not None:
        logger.info(f"Loading test data: {args.test_npz}")
        test_npz = np.load(args.test_npz)
        test_dataset = PSIDataset(
            test_npz["seq_oh"],
            test_npz["struct_oh"],
            test_npz["wobbles"],
            test_npz["metadata_PSI"].astype(np.float32),
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=pin,
        )
        logger.info(
            f"  Test — examples: {len(test_dataset):,}, batches: {len(test_loader)}"
        )

        logger.info(f"Reloading best checkpoint for test eval: {results['checkpoint_path']}")
        best_ckpt = torch.load(results["checkpoint_path"], map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])
        model = model.to(device)

        test_metrics = eval_epoch(model, test_loader, kl_divergence_from_logits, device)
        logger.info(
            f"Test — loss: {test_metrics['loss']:.6f}  rmse: {test_metrics['rmse']:.6f}"
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
