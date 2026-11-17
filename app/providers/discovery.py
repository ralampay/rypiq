"""Deterministic model listing without guessing remote capabilities."""
from dataclasses import replace

from app.providers.contracts import ModelInfo


def merge_models(configured: list[ModelInfo], discovered: list[ModelInfo]) -> list[ModelInfo]:
    merged = {}
    for model in configured:
        key = (model.provider, model.model)
        if key in merged:
            previous = merged[key]
            merged[key] = replace(previous, capabilities=tuple(sorted(set(previous.capabilities + model.capabilities))))
        else:
            merged[key] = model
    for model in discovered:
        key = (model.provider, model.model)
        if key in merged:
            merged[key] = replace(merged[key], source='catalog+discovered', availability='available')
        else:
            merged[key] = model
    return sorted(merged.values(), key=lambda model: (model.provider, model.id))
