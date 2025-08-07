#!/usr/bin/env python
import argparse
import logging
import os
from typing import Any

import numpy as np
import torch
import transformers
from clip_benchmark.datasets.builder import build_dataset, get_dataset_collate_fn
from clip_benchmark.metrics.linear_probe import (
    FeatureDataset,
    Featurizer,
    find_peak,
    infer,
    train,
)
from open_clip.transform import PreprocessCfg, image_transform_v2
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPProcessor

import vec2text
from vec2text.models.model_utils import ClipTextEmbedder, OpenClipEmbedder


def get_cpu_count() -> int:
    """Returns the number of CPUs available to the process."""
    return len(getattr(os, "sched_getaffinity", lambda _: [])(0)) or os.cpu_count() or 1


def get_cpus_per_gpus() -> int:
    """Returns the number of CPUs available per GPU, if any (otherwise it returns them all)."""
    return get_cpu_count() // max(torch.cuda.device_count(), 1)


# Like `linear_probe.evaluate` but it returns the model instead.
def evaluate(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    dataloader: DataLoader,
    fewshot_k: int,
    batch_size: int,
    num_workers: int,
    lr: float,
    epochs: int,
    model_id: str,
    seed: int,
    feature_root: str,
    device: torch.device | str,
    val_dataloader: DataLoader | None = None,
    normalize: bool = True,
    amp: bool = True,
    verbose: bool = False,
) -> tuple[torch.nn.Linear, dict[str, Any]]:
    device = torch.device(device)

    # first we need to featurize the dataset, and store the result in feature_root
    if not os.path.exists(feature_root):
        os.mkdir(feature_root)
    feature_dir = os.path.join(feature_root, model_id)
    if not os.path.exists(feature_dir):
        os.mkdir(feature_dir)

    featurizer = Featurizer(model, normalize).to(device)
    if not os.path.exists(os.path.join(feature_dir, "targets_train.pt")):
        # now we have to cache the features
        devices = [x for x in range(torch.cuda.device_count())]
        featurizer = torch.nn.DataParallel(featurizer, device_ids=devices)

        splits = ["_train", "_val", "_test"]
        for save_str, loader in zip(
            splits, [train_dataloader, val_dataloader, dataloader]
        ):
            if loader is None:
                continue
            features = []
            targets = []
            num_batches_tracked = 0
            num_cached = 0
            with torch.no_grad():
                for images, target in tqdm(loader, desc="Precomputing features"):
                    images = images.to(device)

                    with torch.autocast(device.type, enabled=amp):
                        feature = featurizer(images)

                    features.append(feature.cpu())
                    targets.append(target)

                    num_batches_tracked += 1
                    if (num_batches_tracked % 100) == 0:
                        features = torch.cat(features)
                        targets = torch.cat(targets)

                        torch.save(
                            features,
                            os.path.join(
                                feature_dir, f"features{save_str}_cache_{num_cached}.pt"
                            ),
                        )
                        torch.save(
                            targets,
                            os.path.join(
                                feature_dir, f"targets{save_str}_cache_{num_cached}.pt"
                            ),
                        )
                        num_cached += 1
                        features = []
                        targets = []

            if len(features) > 0:
                features = torch.cat(features)
                targets = torch.cat(targets)
                torch.save(
                    features,
                    os.path.join(
                        feature_dir, f"features{save_str}_cache_{num_cached}.pt"
                    ),
                )
                torch.save(
                    targets,
                    os.path.join(
                        feature_dir, f"targets{save_str}_cache_{num_cached}.pt"
                    ),
                )
                num_cached += 1

            features = torch.load(
                os.path.join(feature_dir, f"features{save_str}_cache_0.pt")
            )
            targets = torch.load(
                os.path.join(feature_dir, f"targets{save_str}_cache_0.pt")
            )
            for k in range(1, num_cached):
                next_features = torch.load(
                    os.path.join(feature_dir, f"features{save_str}_cache_{k}.pt")
                )
                next_targets = torch.load(
                    os.path.join(feature_dir, f"targets{save_str}_cache_{k}.pt")
                )
                features = torch.cat((features, next_features))
                targets = torch.cat((targets, next_targets))

            for k in range(num_cached):
                os.remove(os.path.join(feature_dir, f"features{save_str}_cache_{k}.pt"))
                os.remove(os.path.join(feature_dir, f"targets{save_str}_cache_{k}.pt"))

            torch.save(features, os.path.join(feature_dir, f"features{save_str}.pt"))
            torch.save(targets, os.path.join(feature_dir, f"targets{save_str}.pt"))

    features = torch.load(os.path.join(feature_dir, "features_train.pt"))
    targets = torch.load(os.path.join(feature_dir, "targets_train.pt"))

    # second, make a dataloader with k features per class. if k = -1, use all features.
    length = len(features)
    perm = [p.item() for p in torch.randperm(length)]
    idxs = []
    counts = {}
    num_classes = 0

    for p in perm:
        target = targets[p].item()
        if target not in counts:
            counts[target] = 0
            num_classes += 1

        if fewshot_k < 0 or counts[target] < fewshot_k:
            counts[target] += 1
            idxs.append(p)

    for c in counts:
        if fewshot_k > 0 and counts[c] != fewshot_k:
            raise TypeError("insufficient data for this eval")

    train_features = features[idxs]
    train_labels = targets[idxs]
    feature_train_val_loader = None
    feature_val_loader = None
    if val_dataloader is not None:
        features_val = torch.load(os.path.join(feature_dir, "features_val.pt"))
        targets_val = torch.load(os.path.join(feature_dir, "targets_val.pt"))
        feature_val_dset = FeatureDataset(features_val, targets_val)
        feature_val_loader = DataLoader(
            feature_val_dset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        feature_train_val_dset = FeatureDataset(
            np.concatenate((train_features, features_val)),
            np.concatenate((train_labels, targets_val)),
        )
        feature_train_val_loader = DataLoader(
            feature_train_val_dset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
    feature_train_dset = FeatureDataset(train_features, train_labels)
    feature_train_loader = DataLoader(
        feature_train_dset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    features_test = torch.load(os.path.join(feature_dir, "features_test.pt"))
    targets_test = torch.load(os.path.join(feature_dir, "targets_test.pt"))
    feature_test_dset = FeatureDataset(features_test, targets_test)
    feature_test_loader = DataLoader(
        feature_test_dset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    input_shape, output_shape = features[0].shape[0], targets.max().item() + 1
    if val_dataloader is not None:
        # perform openAI-like hyperparameter sweep
        # https://arxiv.org/pdf/2103.00020.pdf A.3
        # instead of scikit-learn LBFGS use FCNNs with AdamW
        wd_list = np.logspace(-6, 2, num=97).tolist()
        wd_list_init = np.logspace(-6, 2, num=7).tolist()
        wd_init_idx = [i for i, val in enumerate(wd_list) if val in wd_list_init]
        peak_idx = find_peak(
            wd_list,
            wd_init_idx,
            feature_train_loader,
            feature_val_loader,
            input_shape,
            output_shape,
            lr,
            epochs,
            amp,
            device,
            verbose,
            seed,
        )
        step_span = 8
        while step_span > 0:
            left, right = (
                max(peak_idx - step_span, 0),
                min(peak_idx + step_span, len(wd_list) - 1),
            )
            peak_idx = find_peak(
                wd_list,
                [left, peak_idx, right],
                feature_train_loader,
                feature_val_loader,
                input_shape,
                output_shape,
                lr,
                epochs,
                amp,
                device,
                verbose,
                seed,
            )
            step_span //= 2
        best_wd = wd_list[peak_idx]
        if fewshot_k < 0:
            # if we are doing full training, we use the full training set (train+val)
            train_loader = feature_train_val_loader
        else:
            # if we are doing few-shot learning, we use the few-shot training set only
            # as adding the validation set will train on more data than intended
            train_loader = feature_train_loader
    else:
        best_wd = 0
        train_loader = feature_train_loader

    linear_model = train(
        train_loader,
        input_shape,
        output_shape,
        best_wd,
        lr,
        epochs,
        amp,
        str(device),
        seed,
    ).module

    logits, target = infer(linear_model, feature_test_loader, amp, str(device))
    pred = logits.argmax(axis=1)
    classification_results = classification_report(target, pred, output_dict=True)

    return linear_model, classification_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment-dir", default="saves/openclip_vit_b_32_quickgelu_openai_1"
    )
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=get_cpus_per_gpus())
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )

    args = parser.parse_args()

    args.device = torch.device(args.device)

    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    args = parse_args()

    last_checkpoint = transformers.trainer_utils.get_last_checkpoint(
        args.experiment_dir
    )
    logging.info(f"Latest checkpoint: {last_checkpoint}")

    inversion_model = vec2text.models.InversionModel.from_pretrained(
        last_checkpoint, keep_visual=True
    )
    inversion_model.load_embedder_visual()  # Necessary because it may not be in the saved checkpoint.
    inversion_model.eval()

    model = inversion_model.embedder

    if isinstance(model, ClipTextEmbedder):
        processor = CLIPProcessor.from_pretrained(
            inversion_model.config.embedder_model_name
        )
        transform = (
            lambda image: processor.image_processor(image, return_tensors="pt")
            .data["pixel_values"]
            .squeeze(0)
        )
    elif isinstance(model, OpenClipEmbedder):
        pp_cfg = PreprocessCfg(**model.get_visual().preprocess_cfg)
        transform = image_transform_v2(pp_cfg, is_train=False)
    else:
        raise TypeError(f"Unsupported model: {type(model)}")

    dataset = build_dataset(
        dataset_name=args.dataset,
        transform=transform,
        download=True,
        task="linear_probe",
    )

    collate_fn = get_dataset_collate_fn(args.dataset)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=args.device.type != "cpu",
        persistent_workers=args.num_workers > 0,
    )

    train_dataset = build_dataset(
        dataset_name=args.dataset,
        transform=transform,
        split="train",
        download=True,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=args.device.type != "cpu",
        persistent_workers=args.num_workers > 0,
    )

    linear_model, classification_results = evaluate(
        model,
        train_dataloader,
        dataloader,
        fewshot_k=-1,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=0.1,
        epochs=10,
        model_id=inversion_model.config.embedder_model_name.replace("/", "_"),
        seed=args.seed,
        feature_root="features",
        device=args.device,
        normalize=True,
        amp=True,
        verbose=True,
    )

    # This is incorrect, but it doesn't matter for the test:
    corrector_model = vec2text.models.CorrectorEncoderModel.from_pretrained(
        last_checkpoint
    )

    corrector = vec2text.load_corrector(inversion_model, corrector_model)

    logging.info("Inverting weights…")
    inverted_embeddings = vec2text.invert_embeddings(
        linear_model.weight, corrector=corrector
    )
    logging.info("✅ Weights inverted.")

    for i, (class_, inverted_embedding, bias) in enumerate(
        zip(dataset.classes, inverted_embeddings, linear_model.bias, strict=True)
    ):
        class_results = classification_results[str(i)]
        precision = round(class_results["precision"] * 100)
        recall = round(class_results["recall"] * 100)
        print(
            f"{i} - Class: {class_} - P: {precision:3d}% - R: {recall:3d}% - Bias: {bias}"
            f' - Weight to text: "{inverted_embedding}"'
        )


if __name__ == "__main__":
    main()
