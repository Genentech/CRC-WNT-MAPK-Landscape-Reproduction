"""Tree caps, minimum leaf sizes and random state used by the genotype-phenotype trees.

Kept separate from `taxonomy_clinical.tree_engine` so the settings can be read and tested
without `oncophenomap` installed.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TreeSettings:
    """The study's settings: 8 arm and 2 H&E leaves (min. 500) in L1; 150 leaves (min. 50) in L2."""

    l1_arm_leaf_cap: int = 8
    l1_he_leaf_cap: int = 2
    l1_min_samples_leaf: int = 500
    l2_max_leaf_nodes: int = 150
    l2_min_samples_leaf: int = 50
    random_state: int = 42


TREE_SETTINGS = TreeSettings()
