#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dependency-free attention-head sampling helpers for knockout controls."""

from collections import Counter
from typing import List, Tuple
import random


Head = Tuple[int, int]


def ensure_nonempty_head_set(ns_heads: List[Head], dataset_key: str) -> List[Head]:
    """Treat an empty intervention set as a failed cell, never a skipped success."""
    if not ns_heads:
        raise RuntimeError(f"empty per-dataset NS set for {dataset_key}")
    return ns_heads


def get_random_heads_excluding(
    k: int,
    n_layers: int,
    n_heads_per_layer: int,
    exclude: List[Head],
    rng: random.Random,
) -> List[Head]:
    """Sample k heads from the full model, excluding known target heads.

    Raises ValueError if the non-excluded pool has fewer than k heads —
    silently returning fewer heads than requested would quietly change the
    knockout set size and invalidate the NS-vs-random comparison.
    """
    exclude_set = set(exclude)
    pool = [
        (layer, head)
        for layer in range(n_layers)
        for head in range(n_heads_per_layer)
        if (layer, head) not in exclude_set
    ]
    if len(pool) < k:
        raise ValueError(
            f"Cannot sample {k} random heads: only {len(pool)} eligible heads "
            f"remain in a {n_layers}x{n_heads_per_layer} grid after excluding "
            f"{len(exclude_set)} heads."
        )
    return rng.sample(pool, k)


def get_layer_matched_random_heads_excluding(
    ns_heads: List[Head],
    n_layers: int,
    n_heads_per_layer: int,
    exclude: List[Head],
    rng: random.Random,
) -> Tuple[List[Head], int]:
    """Sample random controls with the same per-layer counts as ns_heads.

    If an NS layer does not have enough non-excluded heads left, the
    remainder is filled from the global non-excluded pool so the returned
    set always has exactly len(ns_heads) heads.

    Returns (selected_heads, n_fallback):
      - selected_heads: list of length len(ns_heads)
      - n_fallback: how many of those heads came from the global fallback
        pool rather than from a layer-count match. n_fallback == 0 means a
        clean layer-for-layer match with no fallback.

    Raises ValueError if, even after exhausting all layer pools, the global
    non-excluded pool cannot supply the remaining shortfall — silently
    returning fewer heads than len(ns_heads) would break the K-matched
    comparison this control is meant to provide.
    """
    exclude_set = set(exclude)
    selected: List[Head] = []
    shortfall = 0

    for layer, count in Counter(layer for layer, _ in ns_heads).items():
        layer_pool = [
            (layer, head)
            for head in range(n_heads_per_layer)
            if (layer, head) not in exclude_set
        ]
        take = min(count, len(layer_pool))
        selected.extend(rng.sample(layer_pool, take))
        shortfall += count - take

    n_fallback = 0
    if shortfall > 0:
        selected_set = set(selected)
        global_pool = [
            (layer, head)
            for layer in range(n_layers)
            for head in range(n_heads_per_layer)
            if (layer, head) not in exclude_set and (layer, head) not in selected_set
        ]
        if len(global_pool) < shortfall:
            raise ValueError(
                f"Cannot layer-match {len(ns_heads)} NS heads: after layer-count "
                f"matching, {shortfall} heads still need a fallback but the global "
                f"non-excluded pool only has {len(global_pool)} heads left "
                f"(excluding {len(exclude_set)} excluded + {len(selected)} already "
                f"layer-matched)."
            )
        fallback_heads = rng.sample(global_pool, shortfall)
        selected.extend(fallback_heads)
        n_fallback = shortfall

    return selected, n_fallback
