#!/usr/bin/env python3
"""Train a per-sample quality-weighted LoRA on the SANA transformer."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from r3dloop.sana_lora.data_protocol import read_jsonl, validate_lora_samples  # noqa: E402


REQUIRED_CONFIG = {
    "base_model",
    "trainable_modules",
    "rank",
    "alpha",
    "precision",
    "batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
    "optimizer",
    "max_steps",
    "warmup_steps",
    "checkpointing_steps",
    "gradient_clip",
    "random_crop",
    "random_flip",
    "gradient_checkpointing",
    "freeze_text_encoder",
    "freeze_dc_ae",
    "seed",
}


def load_config(path: str | Path) -> dict:
    import yaml

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("training config must be a YAML object")
    missing = sorted(REQUIRED_CONFIG - set(payload))
    if missing:
        raise ValueError(f"training config missing fields: {missing}")
    if payload["trainable_modules"] != ["to_q", "to_k", "to_v"]:
        raise ValueError("trainable_modules must be exactly [to_q, to_k, to_v]")
    fixed = {
        "rank": 16,
        "alpha": 16,
        "precision": "bf16",
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "optimizer": "adamw_8bit",
        "max_steps": 1500,
        "warmup_steps": 100,
        "checkpointing_steps": 250,
        "random_crop": False,
        "random_flip": False,
        "gradient_checkpointing": True,
        "freeze_text_encoder": True,
        "freeze_dc_ae": True,
    }
    mismatches = {
        key: (payload[key], expected)
        for key, expected in fixed.items()
        if payload[key] != expected
    }
    if mismatches:
        raise ValueError(f"config differs from frozen experiment: {mismatches}")
    if float(payload["learning_rate"]) != 5.0e-5:
        raise ValueError("learning_rate must be 5.0e-5")
    if float(payload["gradient_clip"]) != 1.0:
        raise ValueError("gradient_clip must be 1.0")
    return payload


def resolve_model_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    model_root = Path(os.environ.get("R3DGUARD_MODELS", ROOT / "models"))
    return model_root / path


def validate_training_inputs(config: dict, rows: list[dict], expected: int = 720) -> dict:
    summary = validate_lora_samples(rows, training_only=True)
    if len(rows) != expected:
        raise ValueError(f"expected {expected} training samples, found {len(rows)}")
    missing_images = [str(row["image_path"]) for row in rows if not Path(row["image_path"]).is_file()]
    if missing_images:
        raise FileNotFoundError(f"missing training images: {missing_images[:5]}")
    weights = [float(row["sample_weight"]) for row in rows]
    if not all(0.25 <= weight <= 1.5 for weight in weights):
        raise ValueError("sample_weight must be in [0.25, 1.50]")
    model_path = resolve_model_path(str(config["base_model"]))
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    return {
        **summary,
        "model_path": str(model_path),
        "min_weight": min(weights),
        "max_weight": max(weights),
    }


def run_training(config: dict, rows: list[dict], output_dir: Path, resume: str | None) -> None:
    import torch
    from accelerate import Accelerator
    from accelerate.utils import ProjectConfiguration, set_seed
    from diffusers import (
        AutoencoderDC,
        FlowMatchEulerDiscreteScheduler,
        SanaPipeline,
        SanaTransformer2DModel,
    )
    from diffusers.optimization import get_scheduler
    from diffusers.training_utils import (
        compute_density_for_timestep_sampling,
        compute_loss_weighting_for_sd3,
    )
    from diffusers.utils import convert_unet_state_dict_to_peft
    from peft import LoraConfig, set_peft_model_state_dict
    from peft.utils import get_peft_model_state_dict
    from PIL import Image, ImageOps
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from transformers import AutoTokenizer, Gemma2Model

    class WeightedDataset(Dataset):
        def __init__(self, records: list[dict]) -> None:
            self.records = records
            self.transform = transforms.Compose(
                [
                    transforms.Resize(1024, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(1024),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int) -> dict:
            row = self.records[index]
            with Image.open(row["image_path"]) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                pixels = self.transform(image)
            return {
                "pixel_values": pixels,
                "caption": str(row["caption_en"]),
                "sample_weight": float(row["sample_weight"]),
                "sample_id": str(row["sample_id"]),
            }

    def collate(examples: list[dict]) -> dict:
        return {
            "pixel_values": torch.stack([row["pixel_values"] for row in examples]).float(),
            "captions": [row["caption"] for row in examples],
            "sample_weights": torch.tensor([row["sample_weight"] for row in examples], dtype=torch.float32),
            "sample_ids": [row["sample_id"] for row in examples],
        }

    project = ProjectConfiguration(project_dir=str(output_dir), logging_dir=str(output_dir / "logs"))
    accelerator = Accelerator(
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        mixed_precision="bf16",
        project_config=project,
    )
    set_seed(int(config["seed"]), device_specific=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = resolve_model_path(str(config["base_model"]))
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer", local_files_only=True)
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_path, subfolder="scheduler", local_files_only=True
    )
    text_encoder = Gemma2Model.from_pretrained(
        model_path, subfolder="text_encoder", local_files_only=True, torch_dtype=dtype
    ).requires_grad_(False)
    vae = AutoencoderDC.from_pretrained(
        model_path, subfolder="vae", local_files_only=True, torch_dtype=torch.float32
    ).requires_grad_(False)
    transformer = SanaTransformer2DModel.from_pretrained(
        model_path, subfolder="transformer", local_files_only=True, torch_dtype=dtype
    ).requires_grad_(False)
    transformer.add_adapter(
        LoraConfig(
            r=int(config["rank"]),
            lora_alpha=int(config["alpha"]),
            init_lora_weights="gaussian",
            target_modules=list(config["trainable_modules"]),
        )
    )
    if config["gradient_checkpointing"]:
        transformer.enable_gradient_checkpointing()

    try:
        import bitsandbytes as bnb
    except ImportError as error:
        raise ImportError("adamw_8bit requires bitsandbytes") from error
    parameters = [parameter for parameter in transformer.parameters() if parameter.requires_grad]
    optimizer = bnb.optim.AdamW8bit(parameters, lr=float(config["learning_rate"]))
    dataset = WeightedDataset(rows)
    generator = torch.Generator().manual_seed(int(config["seed"]))
    dataloader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        generator=generator,
        collate_fn=collate,
        num_workers=2,
        pin_memory=True,
    )
    lr_scheduler = get_scheduler(
        "constant_with_warmup",
        optimizer=optimizer,
        num_warmup_steps=int(config["warmup_steps"]),
        num_training_steps=int(config["max_steps"]),
    )
    transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, dataloader, lr_scheduler
    )

    text_encoder.to(accelerator.device, dtype=dtype)
    vae.to(accelerator.device, dtype=torch.float32)
    text_pipeline = SanaPipeline.from_pretrained(
        model_path,
        transformer=None,
        vae=None,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        local_files_only=True,
    ).to(accelerator.device)

    def unwrap():
        return accelerator.unwrap_model(transformer)

    def save_hook(models, weights, directory):
        if accelerator.is_main_process:
            state = get_peft_model_state_dict(unwrap())
            SanaPipeline.save_lora_weights(directory, transformer_lora_layers=state)
        while weights:
            weights.pop()

    def load_hook(models, directory):
        model = models.pop()
        state = SanaPipeline.lora_state_dict(directory)
        transformer_state = {
            key.removeprefix("transformer."): value
            for key, value in state.items()
            if key.startswith("transformer.")
        }
        transformer_state = convert_unet_state_dict_to_peft(transformer_state)
        set_peft_model_state_dict(model, transformer_state, adapter_name="default")

    accelerator.register_save_state_pre_hook(save_hook)
    accelerator.register_load_state_pre_hook(load_hook)

    global_step = 0
    if resume:
        if resume == "latest":
            checkpoints = sorted(
                output_dir.glob("checkpoint-*"), key=lambda path: int(path.name.split("-")[-1])
            )
            checkpoint = checkpoints[-1] if checkpoints else None
        else:
            checkpoint = Path(resume)
        if checkpoint is not None and checkpoint.is_dir():
            accelerator.load_state(str(checkpoint))
            global_step = int(checkpoint.name.split("-")[-1])

    sigmas = noise_scheduler.sigmas.to(accelerator.device)
    schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)

    def get_sigmas(timesteps, dimensions, target_dtype):
        indices = [(schedule_timesteps == timestep).nonzero().item() for timestep in timesteps]
        selected = sigmas[indices].flatten().to(dtype=target_dtype)
        while selected.ndim < dimensions:
            selected = selected.unsqueeze(-1)
        return selected

    log_path = output_dir / "train_metrics.jsonl"
    started = time.perf_counter()
    epochs = math.ceil(
        int(config["max_steps"])
        * int(config["gradient_accumulation_steps"])
        / max(1, len(dataloader))
    )
    for _epoch in range(epochs + 1):
        transformer.train()
        for batch in dataloader:
            with accelerator.accumulate(transformer):
                with torch.no_grad():
                    prompt_embeds, prompt_attention_mask, _, _ = text_pipeline.encode_prompt(
                        batch["captions"], max_sequence_length=300, complex_human_instruction=None
                    )
                    pixels = batch["pixel_values"].to(accelerator.device, dtype=vae.dtype)
                    model_input = vae.encode(pixels).latent * vae.config.scaling_factor
                    model_input = model_input.to(dtype=dtype)
                noise = torch.randn_like(model_input)
                density = compute_density_for_timestep_sampling(
                    weighting_scheme="none", batch_size=model_input.shape[0]
                )
                indices = (density * noise_scheduler.config.num_train_timesteps).long()
                timesteps = noise_scheduler.timesteps[indices].to(accelerator.device)
                step_sigmas = get_sigmas(timesteps, model_input.ndim, model_input.dtype)
                noisy = (1.0 - step_sigmas) * model_input + step_sigmas * noise
                prediction = transformer(
                    hidden_states=noisy,
                    encoder_hidden_states=prompt_embeds.to(dtype=dtype),
                    encoder_attention_mask=prompt_attention_mask,
                    timestep=timesteps,
                    return_dict=False,
                )[0]
                schedule_weight = compute_loss_weighting_for_sd3(
                    weighting_scheme="none", sigmas=step_sigmas
                )
                target = noise - model_input
                per_sample = torch.mean(
                    (schedule_weight.float() * (prediction.float() - target.float()) ** 2).reshape(
                        target.shape[0], -1
                    ),
                    dim=1,
                )
                sample_weights = batch["sample_weights"].to(accelerator.device)
                loss = (per_sample * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(parameters, float(config["gradient_clip"]))
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process:
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(
                                {
                                    "step": global_step,
                                    "loss": float(loss.detach()),
                                    "learning_rate": float(lr_scheduler.get_last_lr()[0]),
                                    "elapsed_seconds": time.perf_counter() - started,
                                    "peak_memory_mib": int(torch.cuda.max_memory_allocated() / 1024**2),
                                }
                            )
                            + "\n"
                        )
                if global_step % int(config["checkpointing_steps"]) == 0:
                    accelerator.save_state(str(output_dir / f"checkpoint-{global_step}"))
                if global_step >= int(config["max_steps"]):
                    break
        if global_step >= int(config["max_steps"]):
            break
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_state = get_peft_model_state_dict(unwrap())
        SanaPipeline.save_lora_weights(output_dir / "final", transformer_lora_layers=final_state)
        (output_dir / "training_summary.json").write_text(
            json.dumps(
                {
                    "steps": global_step,
                    "elapsed_seconds": time.perf_counter() - started,
                    "peak_memory_mib": int(torch.cuda.max_memory_allocated() / 1024**2),
                    "seed": int(config["seed"]),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    accelerator.end_training()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expect", type=int, default=720)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    rows = read_jsonl(args.manifest)
    summary = validate_training_inputs(config, rows, expected=args.expect)
    print(json.dumps({"event": "training_preflight", **summary}, ensure_ascii=False), flush=True)
    if not args.preflight_only:
        run_training(config, rows, args.output_dir, args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
