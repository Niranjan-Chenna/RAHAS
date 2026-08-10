from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from torch.utils.data import DataLoader, Dataset


def add_project_root(root: Path) -> None:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train neural character-only alpha segmentation.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--characters", type=Path, default=Path("characters"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--steps_per_epoch", type=int, default=320)
    parser.add_argument("--val_steps", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--base_channels", type=int, default=24)
    parser.add_argument("--activation", choices=["relu", "leaky_relu", "gelu"], default="gelu")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--run_name", type=str, default="character_alpha_v1")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview_every", type=int, default=1)
    return parser.parse_args()


def list_images(folder: Path) -> list[Path]:
    return sorted(
        [path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda path: path.as_posix().casefold(),
    )


def image_to_alpha(image: Image.Image) -> Image.Image:
    arr = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    alpha = np.clip((0.92 - arr) / 0.72, 0.0, 1.0)
    alpha = np.power(alpha, 0.75)
    alpha_img = Image.fromarray(np.clip(alpha * 255.0, 0, 255).astype(np.uint8), mode="L")
    return alpha_img.filter(ImageFilter.GaussianBlur(0.45))


def local_background_array(arr: np.ndarray) -> np.ndarray:
    img = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode="L")
    bg = img.filter(ImageFilter.MaxFilter(23)).filter(ImageFilter.GaussianBlur(4.0))
    return np.asarray(bg, dtype=np.float32) / 255.0


def local_contrast_tensor(gray: torch.Tensor) -> torch.Tensor:
    local_mean = F.avg_pool2d(gray, 9, stride=1, padding=4)
    local_sq = F.avg_pool2d(gray.square(), 9, stride=1, padding=4)
    return ((local_sq - local_mean.square()).clamp_min(0.0).sqrt() * 4.0).clamp(0.0, 1.0)


class SyntheticCharacterAlphaDataset(Dataset):
    def __init__(self, character_paths: list[Path], length: int, image_size: int, seed: int) -> None:
        self.character_paths = character_paths
        self.length = length
        self.image_size = image_size
        self.seed = seed
        self.cache: dict[Path, tuple[Image.Image, Image.Image]] = {}

    def __len__(self) -> int:
        return self.length

    def load_character(self, path: Path) -> tuple[Image.Image, Image.Image]:
        if path not in self.cache:
            image = Image.open(path).convert("L")
            alpha = image_to_alpha(image)
            self.cache[path] = (image, alpha)
        return self.cache[path]

    def background(self, rng: random.Random) -> np.ndarray:
        size = self.image_size
        base = rng.uniform(0.82, 0.98)
        arr = np.full((size, size), base, dtype=np.float32)
        gx = np.linspace(rng.uniform(-0.04, 0.04), rng.uniform(-0.04, 0.04), size, dtype=np.float32)
        gy = np.linspace(rng.uniform(-0.03, 0.03), rng.uniform(0.03, -0.03), size, dtype=np.float32)[:, None]
        arr = np.clip(arr + gx + gy, 0.0, 1.0)
        texture = rng.uniform(0.006, 0.020)
        arr += np.random.default_rng(rng.randrange(1_000_000_000)).normal(0.0, texture, arr.shape).astype(np.float32)
        return np.clip(arr, 0.0, 1.0)

    def paste_character(self, canvas: np.ndarray, target: np.ndarray, rng: random.Random, x: int, baseline: int) -> int:
        path = rng.choice(self.character_paths)
        image, alpha = self.load_character(path)
        scale = rng.uniform(0.24, 0.54) * self.image_size / max(image.size)
        new_size = (max(10, int(image.width * scale)), max(10, int(image.height * scale)))
        image = image.resize(new_size, Image.Resampling.BICUBIC)
        alpha = alpha.resize(new_size, Image.Resampling.BICUBIC)
        angle = rng.uniform(-7.0, 7.0)
        image = image.rotate(angle, expand=True, fillcolor=255, resample=Image.Resampling.BICUBIC)
        alpha = alpha.rotate(angle, expand=True, fillcolor=0, resample=Image.Resampling.BICUBIC)
        alpha_arr = np.asarray(alpha, dtype=np.float32) / 255.0
        if alpha_arr.max() < 0.05:
            return x + 8
        ink = np.asarray(image, dtype=np.float32) / 255.0
        ink = np.clip(ink * rng.uniform(0.42, 0.82), 0.0, 1.0)
        y = int(baseline - image.height + rng.randint(-8, 8))
        x = int(x + rng.randint(-3, 5))
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + image.width, self.image_size), min(y + image.height, self.image_size)
        if x1 <= x0 or y1 <= y0:
            return x + image.width + rng.randint(3, 12)
        sx0, sy0 = x0 - x, y0 - y
        sx1, sy1 = sx0 + (x1 - x0), sy0 + (y1 - y0)
        a = alpha_arr[sy0:sy1, sx0:sx1]
        canvas[y0:y1, x0:x1] = canvas[y0:y1, x0:x1] * (1.0 - a) + ink[sy0:sy1, sx0:sx1] * a
        target[y0:y1, x0:x1] = np.maximum(target[y0:y1, x0:x1], a)
        return x + image.width + rng.randint(2, 14)

    def add_noise(self, canvas: np.ndarray, target: np.ndarray, rng: random.Random) -> np.ndarray:
        size = self.image_size
        noise_mask = np.zeros_like(canvas)
        speckles = rng.randint(180, 900)
        for _ in range(speckles):
            radius = rng.choice([1, 1, 1, 2, 2, 3])
            x = rng.randrange(size)
            y = rng.randrange(size)
            yy, xx = np.ogrid[:size, :size]
            blob = (xx - x) ** 2 + (yy - y) ** 2 <= radius * radius
            darkness = rng.uniform(0.08, 0.65)
            canvas[blob] = np.minimum(canvas[blob], np.clip(canvas[blob] - darkness, 0.0, 1.0))
            noise_mask[blob] = 1.0
        image = Image.fromarray(np.clip(canvas * 255.0, 0, 255).astype(np.uint8), mode="L")
        draw = ImageDraw.Draw(image)
        for _ in range(rng.randint(0, 4)):
            y = rng.randint(0, size - 1)
            draw.line((0, y, size, y + rng.randint(-4, 4)), fill=rng.randint(45, 130), width=rng.randint(1, 4))
        for _ in range(rng.randint(0, 3)):
            x0 = rng.randint(-20, size // 2)
            y0 = rng.randint(0, size - 1)
            x1 = rng.randint(size // 2, size + 20)
            y1 = rng.randint(0, size - 1)
            draw.line((x0, y0, x1, y1), fill=rng.randint(80, 180), width=rng.randint(1, 3))
        canvas = np.asarray(image, dtype=np.float32) / 255.0
        canvas = np.clip(canvas + np.random.default_rng(rng.randrange(1_000_000_000)).normal(0, 0.012, canvas.shape), 0, 1)
        return canvas

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = random.Random(self.seed + index * 9973)
        canvas = self.background(rng)
        target = np.zeros_like(canvas, dtype=np.float32)
        x = rng.randint(2, 18)
        baseline = rng.randint(int(self.image_size * 0.55), int(self.image_size * 0.86))
        while x < self.image_size - 20:
            x = self.paste_character(canvas, target, rng, x, baseline)
            if rng.random() < 0.15:
                break
        canvas = self.add_noise(canvas, target, rng)
        background = local_background_array(canvas)
        darkness = np.clip((background - canvas) / np.maximum(background, 0.08), 0.0, 1.0)
        darkness = np.power(darkness, 0.85)
        image = torch.from_numpy(canvas[None, :, :].astype(np.float32))
        darkness_tensor = torch.from_numpy(darkness[None, :, :].astype(np.float32))
        contrast = local_contrast_tensor(image.unsqueeze(0)).squeeze(0)
        inputs = torch.cat([image, darkness_tensor, contrast], dim=0)
        return inputs, torch.from_numpy(target[None, :, :].astype(np.float32))


def dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    smooth = 1e-5
    inter = (pred * target).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * inter + smooth) / (denom + smooth)).mean()


def train_epoch(model, loader, optimizer, scaler, device, use_amp: bool) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "bce": 0.0, "dice": 0.0, "noise": 0.0}
    seen = 0
    iterator = tqdm(loader, desc="train", unit="batch", leave=True) if tqdm else loader
    for inputs, target in iterator:
        inputs = inputs.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            pred = model(inputs)
        pred_loss = pred.float()
        target_loss = target.float()
        weights = 1.0 + 5.0 * target_loss
        bce = F.binary_cross_entropy(pred_loss, target_loss, weight=weights)
        dice = dice_loss(pred_loss, target_loss)
        noise = (pred_loss * (1.0 - target_loss)).mean()
        loss = bce + 0.85 * dice + 0.55 * noise
        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        batch = inputs.shape[0]
        seen += batch
        for key, value in [("loss", loss), ("bce", bce), ("dice", dice), ("noise", noise)]:
            totals[key] += float(value.detach().cpu()) * batch
        if tqdm:
            iterator.set_postfix(loss=f"{float(loss):.4f}", dice=f"{float(dice):.4f}", noise=f"{float(noise):.4f}")
    return {key: value / max(seen, 1) for key, value in totals.items()}


def validate(model, loader, device, use_amp: bool) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "bce": 0.0, "dice": 0.0, "noise": 0.0}
    seen = 0
    with torch.no_grad():
        for inputs, target in loader:
            inputs = inputs.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(inputs)
            pred_loss = pred.float()
            target_loss = target.float()
            weights = 1.0 + 5.0 * target_loss
            bce = F.binary_cross_entropy(pred_loss, target_loss, weight=weights)
            dice = dice_loss(pred_loss, target_loss)
            noise = (pred_loss * (1.0 - target_loss)).mean()
            loss = bce + 0.85 * dice + 0.55 * noise
            batch = inputs.shape[0]
            seen += batch
            for key, value in [("loss", loss), ("bce", bce), ("dice", dice), ("noise", noise)]:
                totals[key] += float(value.detach().cpu()) * batch
    return {key: value / max(seen, 1) for key, value in totals.items()}


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.detach().cpu().clamp(0, 1).squeeze().numpy()
    return Image.fromarray((arr * 255).astype(np.uint8), mode="L")


def save_preview(model, dataset: Dataset, device: torch.device, path: Path) -> None:
    model.eval()
    samples = [dataset[i] for i in range(6)]
    inputs = torch.stack([sample[0] for sample in samples]).to(device)
    target = torch.stack([sample[1] for sample in samples]).to(device)
    with torch.no_grad():
        pred = model(inputs)
    labels = ["input", "darkness", "contrast", "target", "pred", "overlay"]
    tile = 112
    label_h = 24
    margin = 14
    gutter = 8
    sheet = Image.new("RGB", (margin * 2 + len(labels) * tile + (len(labels) - 1) * gutter, margin * 2 + len(samples) * (tile + label_h + gutter)), (248, 248, 244))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for row in range(len(samples)):
        imgs = [
            tensor_to_image(inputs[row, 0]),
            tensor_to_image(inputs[row, 1]),
            tensor_to_image(inputs[row, 2]),
            tensor_to_image(target[row]),
            tensor_to_image(pred[row]),
            tensor_to_image((1.0 - pred[row]) + inputs[row, 0:1] * pred[row]),
        ]
        y = margin + row * (tile + label_h + gutter)
        for col, (label, image) in enumerate(zip(labels, imgs)):
            image.thumbnail((tile, tile), Image.Resampling.LANCZOS)
            x = margin + col * (tile + gutter)
            sheet.paste(image.convert("RGB"), (x + (tile - image.width) // 2, y))
            draw.rectangle((x, y, x + tile, y + tile), outline=(210, 210, 205))
            draw.text((x, y + tile + 4), label, fill=(30, 30, 30), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def save_checkpoint(path: Path, model, optimizer, epoch: int, best: float, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_args = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    torch.save({"epoch": epoch, "best_val_loss": best, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "args": clean_args}, path)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    add_project_root(Path(__file__).resolve().parents[1])
    from src.models.character_alpha_net import CharacterAlphaNet

    character_paths = list_images(root / args.characters)
    if not character_paths:
        raise FileNotFoundError(f"No character images found under {root / args.characters}")
    random.Random(args.seed).shuffle(character_paths)
    split = max(1, int(len(character_paths) * 0.9))
    train_paths = character_paths[:split]
    val_paths = character_paths[split:] or character_paths[: min(64, len(character_paths))]
    train_dataset = SyntheticCharacterAlphaDataset(train_paths, args.steps_per_epoch * args.batch_size, args.image_size, args.seed)
    val_dataset = SyntheticCharacterAlphaDataset(val_paths, args.val_steps * args.batch_size, args.image_size, args.seed + 100_000)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
    model = CharacterAlphaNet(base_channels=args.base_channels, activation=args.activation).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    checkpoint_dir = root / "pipeline/checkpoints" / args.run_name
    preview_dir = root / "results/restoration/previews" / args.run_name
    log_path = root / "results/logs" / f"{args.run_name}_train_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"characters: {len(character_paths)} train={len(train_paths)} val={len(val_paths)}")
    print(f"device used: {device}")
    print(f"amp enabled: {use_amp}")
    best = float("inf")
    start_epoch = 1
    if args.resume is not None:
        resume_path = args.resume if args.resume.is_absolute() else root / args.resume
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        best = float(checkpoint.get("best_val_loss", best))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(f"resumed from: {resume_path} at epoch {start_epoch - 1}")
    log_mode = "a" if args.resume is not None and log_path.exists() else "w"
    with log_path.open(log_mode, newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "train_bce", "train_dice", "train_noise", "val_loss", "val_bce", "val_dice", "val_noise", "best_val_loss"])
        if log_mode == "w":
            writer.writeheader()
        for epoch in range(start_epoch, start_epoch + args.epochs):
            train_metrics = train_epoch(model, train_loader, optimizer, scaler, device, use_amp)
            val_metrics = validate(model, val_loader, device, use_amp)
            if val_metrics["loss"] < best:
                best = val_metrics["loss"]
                save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, epoch, best, args)
            save_checkpoint(checkpoint_dir / "latest.pt", model, optimizer, epoch, best, args)
            row = {"epoch": epoch, "best_val_loss": best}
            row.update({f"train_{key}": value for key, value in train_metrics.items()})
            row.update({f"val_{key}": value for key, value in val_metrics.items()})
            writer.writerow(row)
            file.flush()
            print(f"epoch={epoch} train_loss={train_metrics['loss']:.5f} val_loss={val_metrics['loss']:.5f} val_dice={val_metrics['dice']:.5f} val_noise={val_metrics['noise']:.5f} best={best:.5f}")
            if epoch == 1 or epoch % args.preview_every == 0:
                save_preview(model, val_dataset, device, preview_dir / f"val_preview_epoch_{epoch:03d}.png")
    print(f"best checkpoint path: {checkpoint_dir / 'best.pt'}")
    print(f"training log CSV: {log_path}")
    print(f"preview dir: {preview_dir}")


if __name__ == "__main__":
    main()
