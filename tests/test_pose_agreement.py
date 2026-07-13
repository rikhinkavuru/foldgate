"""Unit tests for the W1 pose-agreement features (synthetic poses; no structure tarball).

These exercise the ligand-selection, intra-model diversity, and cross-model matching logic
on hand-built pose dicts, so they need only numpy + spyrmsd (skipped if spyrmsd is absent).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("spyrmsd")

from foldgate.features import pose_agreement as pa  # noqa: E402


def _pose(lig_xyz, lig_el, names, ca_xyz, chain="L"):
    cak = [("A", i) for i in range(len(ca_xyz))]
    return {"lig": {chain: {"el": np.asarray(lig_el, int), "xyz": np.asarray(lig_xyz, float),
                            "names": list(names), "plddt": 90.0}},
            "ca": np.asarray(ca_xyz, float), "cak": cak}


def _receptor(n=12, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 8, (n, 3))


def _ligand(seed=1, n=6):
    rng = np.random.default_rng(seed)
    xyz = rng.normal(0, 1.5, (n, 3))
    el = np.array([6, 6, 7, 8, 6, 6][:n])
    names = [f"C{i}" for i in range(n)]
    return xyz, el, names


def test_select_ligand_by_heavy():
    ca = _receptor()
    lig_xyz, el, names = _ligand()
    pose = _pose(lig_xyz, el, names, ca, chain="L")
    # add a small ion chain (2 heavy atoms) to make it multi-ligand
    pose["lig"]["I"] = {"el": np.array([30, 30]), "xyz": np.zeros((2, 3)),
                        "names": ["ZN1", "ZN2"], "plddt": 80.0}
    sel = pa.select_ligand(pose, expected_heavy=6)
    assert sel is not None and len(sel[0]) == 6            # picks the 6-atom drug, not the ion
    assert pa.select_ligand(pose, expected_heavy=999) is None
    assert pa.select_ligand(pose, None) is None            # ambiguous without a heavy count


def test_intra_model_pose_diversity():
    ca = _receptor()
    lig_xyz, el, names = _ligand()
    ref = _pose(lig_xyz, el, names, ca)
    tight = _pose(lig_xyz + 0.1, el, names, ca)             # ~0.1 A move
    far = _pose(lig_xyz + np.array([5.0, 0, 0]), el, names, ca)  # a flipped mode
    scores = [0.9, 0.8, 0.7]                                # ref delivered
    feat = pa.intra_model_pose_features([ref, tight, far], scores, expected_heavy=6)
    assert feat["n_samples_pose"] == 3
    assert feat["intra_model_pose_std"] > 0.5              # the far sample creates spread
    assert 0.0 < feat["pose_consensus_frac"] < 1.0         # tight in-mode, far out-of-mode


def test_cross_model_matches_shuffled_atoms():
    ca = _receptor()
    lig_xyz, el, names = _ligand()
    a = _pose(lig_xyz, el, names, ca)
    perm = np.array([3, 1, 5, 0, 4, 2])                    # different model -> different atom order
    b = _pose(lig_xyz[perm] + np.array([1.2, 0, 0]), el[perm], [names[i] for i in perm], ca)
    feats = pa.cross_model_pose_features({"af3": a, "boltz": b}, expected_heavy=6)
    assert abs(feats["af3"]["xmodel_pose_rmsd_median"] - 1.2) < 1e-6   # reorder recovered exactly
    assert feats["af3"]["xmodel_n_pose"] == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
