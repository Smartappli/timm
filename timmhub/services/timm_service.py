from __future__ import annotations

import io
from functools import lru_cache
from typing import Any

import timm
import torch
from PIL import Image
from timm.data import create_transform, resolve_data_config


class TimmService:
    @staticmethod
    @lru_cache(maxsize=8)
    def _build_model(model_name: str, classification: bool = True):
        if classification:
            model = timm.create_model(model_name, pretrained=True)
        else:
            model = timm.create_model(model_name, pretrained=True, num_classes=0)
        model.eval()
        return model

    @staticmethod
    def _read_image(image_bytes: bytes) -> Image.Image:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    def list_models(self, limit: int = 50) -> list[str]:
        return timm.list_models(pretrained=True)[:limit]

    def get_model_metadata(self, model_name: str) -> dict[str, Any]:
        model = timm.create_model(model_name, pretrained=True)
        cfg = getattr(model, "pretrained_cfg", {}) or {}
        return {
            "model_name": model_name,
            "num_classes": getattr(model, "num_classes", None),
            "default_cfg": cfg,
            "parameter_count": sum(p.numel() for p in model.parameters()),
        }

    def classify(self, model_name: str, image_bytes: bytes, top_k: int = 5) -> dict[str, Any]:
        model = self._build_model(model_name, classification=True)
        cfg = resolve_data_config({}, model=model)
        transform = create_transform(**cfg, is_training=False)
        image = self._read_image(image_bytes)
        x = transform(image).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits[0], dim=0)
            values, indices = torch.topk(probs, k=min(top_k, probs.shape[0]))
        labels = []
        if hasattr(model, "pretrained_cfg"):
            labels = model.pretrained_cfg.get("label_names") or []
        predictions = []
        for score, idx in zip(values.tolist(), indices.tolist(), strict=False):
            predictions.append({
                "index": idx,
                "score": score,
                "label": labels[idx] if idx < len(labels) else str(idx),
            })
        return {"task": "classification", "model_name": model_name, "predictions": predictions}

    def embedding(self, model_name: str, image_bytes: bytes) -> dict[str, Any]:
        model = self._build_model(model_name, classification=False)
        cfg = resolve_data_config({}, model=model)
        transform = create_transform(**cfg, is_training=False)
        image = self._read_image(image_bytes)
        x = transform(image).unsqueeze(0)
        with torch.no_grad():
            vector = model(x)[0].flatten()
        return {
            "task": "embedding",
            "model_name": model_name,
            "dimension": int(vector.shape[0]),
            "preview": vector[:16].tolist(),
        }

    def feature_maps(self, model_name: str, image_bytes: bytes) -> dict[str, Any]:
        model = timm.create_model(model_name, pretrained=True, features_only=True)
        model.eval()
        cfg = resolve_data_config({}, model=model)
        transform = create_transform(**cfg, is_training=False)
        image = self._read_image(image_bytes)
        x = transform(image).unsqueeze(0)
        with torch.no_grad():
            features = model(x)
        return {
            "task": "feature_maps",
            "model_name": model_name,
            "feature_shapes": [list(feat.shape) for feat in features],
        }


timm_service = TimmService()
