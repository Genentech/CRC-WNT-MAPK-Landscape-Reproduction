"""Engine 1 -- two-stage L1 context + L2 mutation-group tree (oncophenomap).

Stage 1 (L1 "context"): an arm tree on binarized chromosome-arm fractions, crossed
with a median-split tree on H&E composition for samples that have an H&E slide.
Stage 2 (L2) fits a pooled mutation/focal-CNV tree. Every cap, minimum leaf size and
random state comes from `TREE_SETTINGS`, the study's 8/2 L1 caps and 150/50 L2
settings.

Stage 2 (L2 "mutation group"): a single pooled tree on mutations + binarized focal CNV,
fit against the L1-context-residualized phenotype. Serves SFig7 / Fig4b / Fig4c.
Megaclusters are formed as in the study: Ward clustering (with optimal leaf ordering) of
the z-scored per-leaf mean factor scores over a set of representative factors, cut into
`N_MEGA` clusters and numbered MC1.. from left to right in the dendrogram. Every leaf
belongs to exactly one megacluster.

L1 and L2 are fit independently and are NOT nested: L2 groups span many L1 contexts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from ..methods.megaclusters import compute_megaclusters
from ..tree_settings import TREE_SETTINGS

try:
    from oncophenomap import GenotypePhenotypeTree, focal_cnv_features
    from oncophenomap.residualize import (
        _binary_block,
        _fit_context_tree,
        _median_block,
        _routed_context,
        residualize as remove_context,
    )
    from oncophenomap.tree import _leaf_rules
except ImportError as exc:  # pragma: no cover - exercised only when the dep is missing
    raise ImportError(
        f"`oncophenomap` could not be imported ({exc}); it is needed for the 15 "
        "taxonomy/clinical panels (Fig4b-d, SFig6, SFig7, SFig8a-b, SFig10a-h). Run "
        "setup_workspace.py and `pip install -e <workspace>/oncophenomap` (see the "
        "README). The other 14 panels do not need it."
    ) from exc

ARM_THRESHOLD = 0.5

_SPLIT_RE = re.compile(r"^(.*?) (<=|>) (.+)$")


def _rerender_arm_rule(rule: str) -> str:
    """`_leaf_rules` gives `"feat <= 0.5 / feat2 > 0.5"`; render it in the published
    SFig6 style (`"feat<=0.50 / feat2>0.50"`, no spaces, 2dp thresholds)."""
    if rule == "Root":
        return "Root"
    parts = []
    for piece in rule.split(" / "):
        m = _SPLIT_RE.match(piece)
        if not m:
            parts.append(piece)
            continue
        feat, op, thr = m.groups()
        parts.append(f"{feat}{op}{float(thr):.2f}")
    return " / ".join(parts)


def _render_group_full(rule: str) -> str:
    """Convert a genotype-tree leaf rule (binary 0/1 mutation/CNV features) into the
    published pipe-delimited gene-status style (`"APC wt | BRAF mut | ..."`)."""
    if rule == "Root":
        return "Root"
    out = []
    for piece in rule.split(" / "):
        m = _SPLIT_RE.match(piece)
        if not m:
            continue
        feat, op, _thr = m.groups()
        mutated = op == ">"
        if feat == "MSI_High":
            out.append("MSI-H" if mutated else "MSS")
        elif feat.endswith("_amp"):
            gene = feat[: -len("_amp")]
            out.append(f"{gene} amp" if mutated else f"{gene} not_amp")
        elif feat.endswith("_del"):
            gene = feat[: -len("_del")]
            out.append(f"{gene} del" if mutated else f"{gene} not_del")
        else:
            out.append(f"{feat} mut" if mutated else f"{feat} wt")
    return " | ".join(out)


@dataclass
class L1Fit:
    context: pd.Series  # sample_id -> "A{arm_leaf}H{he_leaf}" / "A{arm_leaf}"
    rule: dict  # context label -> rendered rule string
    counts: pd.Series  # context label -> n


@dataclass
class L2Fit:
    labels: pd.Series  # sample_id -> "L2#<leaf_id>"
    group_full: dict  # "L2#<leaf_id>" -> pipe-joined gene-status string
    mega_cluster: dict  # "L2#<leaf_id>" -> "MC<k>" from clustering the leaves
    mutation_group_id: dict  # "L2#<leaf_id>" -> stable sequential int id (Fig4c)
    counts: pd.Series
    variance_explained: float
    tree: GenotypePhenotypeTree = field(repr=False)


def fit_l1_context(
    factor_scores: pd.DataFrame,
    arm_fractions: pd.DataFrame,
    he_features: pd.DataFrame,
    *,
    random_state: int | None = None,
    min_samples_leaf: int | None = None,
) -> L1Fit:
    profile = TREE_SETTINGS
    random_state = profile.random_state if random_state is None else random_state
    min_samples_leaf = min_samples_leaf or profile.l1_min_samples_leaf
    y = factor_scores
    arm_binary = _binary_block(arm_fractions, ARM_THRESHOLD)
    arm_tree = _fit_context_tree(
        arm_binary,
        y,
        max_leaf_nodes=profile.l1_arm_leaf_cap,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    arm_rules = {
        k: _rerender_arm_rule(v)
        for k, v in _leaf_rules(arm_tree, list(arm_fractions.columns)).items()
    }

    he_features = he_features.reindex(y.index)
    he_binary = _median_block(he_features)
    present = he_features.notna().any(axis=1)
    he_tree = _fit_context_tree(
        he_binary.loc[present],
        y.loc[present],
        max_leaf_nodes=profile.l1_he_leaf_cap,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    he_rules = {
        k: _rerender_arm_rule(v)
        for k, v in _leaf_rules(he_tree, list(he_binary.columns)).items()
    }

    context = _routed_context(arm_tree, he_tree, arm_binary, he_binary, present)

    arm_leaf_of = arm_tree.apply(arm_binary.values)
    he_leaf_of = he_tree.apply(he_binary.values)
    present_arr = present.to_numpy()
    ctx_arr = context.to_numpy()

    rule_by_context: dict[str, str] = {}
    for ctx in pd.unique(ctx_arr):
        idx = np.flatnonzero(ctx_arr == ctx)[0]
        arm_rule = arm_rules[arm_leaf_of[idx]]
        if present_arr[idx]:
            rule_by_context[ctx] = f"{arm_rule} + {he_rules[he_leaf_of[idx]]}"
        else:
            rule_by_context[ctx] = f"{arm_rule} + (H&E n/a)"

    return L1Fit(context=context, rule=rule_by_context, counts=context.value_counts())


def fit_l2_tree(
    factor_scores: pd.DataFrame,
    mutations: pd.DataFrame,
    focal_cnv: pd.DataFrame,
    l1_context: pd.Series,
    *,
    max_leaf_nodes: int | None = None,
    min_samples_leaf: int | None = None,
    random_state: int | None = None,
) -> L2Fit:
    profile = TREE_SETTINGS
    max_leaf_nodes = (
        profile.l2_max_leaf_nodes if max_leaf_nodes is None else max_leaf_nodes
    )
    min_samples_leaf = (
        profile.l2_min_samples_leaf if min_samples_leaf is None else min_samples_leaf
    )
    random_state = profile.random_state if random_state is None else random_state
    residualized = remove_context(factor_scores, l1_context)
    cnv_binary = focal_cnv_features(focal_cnv)
    genotype = pd.concat([mutations, cnv_binary], axis=1)

    tree = GenotypePhenotypeTree(
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    tree.fit(residualized, genotype)

    leaf_ids = tree.labels_
    labels = leaf_ids.map(lambda i: f"L2#{i}")
    counts = labels.value_counts()

    group_full = {
        f"L2#{leaf_id}": _render_group_full(rule)
        for leaf_id, rule in tree.leaf_rules_.items()
    }

    mega_cluster = compute_megaclusters(labels, factor_scores.loc[labels.index])

    ordered_leaves = sorted(tree.leaf_rules_.keys())
    mutation_group_id = {f"L2#{leaf_id}": i for i, leaf_id in enumerate(ordered_leaves)}

    return L2Fit(
        labels=labels,
        group_full=group_full,
        mega_cluster=mega_cluster,
        mutation_group_id=mutation_group_id,
        counts=counts,
        variance_explained=float(tree.variance_explained_),
        tree=tree,
    )
