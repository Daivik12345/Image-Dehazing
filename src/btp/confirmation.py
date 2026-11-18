"""Scene-clustered paired summaries, separate from unchanged systems eligibility."""

from collections import defaultdict
from collections.abc import Sequence

import numpy as np


def paired_summary(
    baseline: Sequence[dict],
    candidate: Sequence[dict],
    *,
    planned_runs: Sequence[tuple[str, int, int]],
    seed: int = 42,
    samples: int = 10000,
) -> dict:
    """Bootstrap scene means of candidate-minus-baseline, never independent repeats.

    Each row requires scene_id, seed, repeat and finite value. Caller supplies one
    predeclared metric and verifies matching protocol/data/model-run provenance.
    This is descriptive inference, not an automatic promotion decision; intervals
    from tiny development panels cannot establish population-level improvements.
    """
    if type(samples) is not int or samples < 100:
        raise ValueError("At least 100 bootstrap samples required")

    def index(rows):
        result = {}
        for row in rows:
            key = (row["scene_id"], row["seed"], row["repeat"])
            if (
                not isinstance(key[0], str)
                or not key[0]
                or any(type(v) is not int or v < 0 for v in key[1:])
            ):
                raise ValueError("Invalid scene/seed/repeat")
            if key in result or not np.isfinite(row["value"]):
                raise ValueError("Duplicate or nonfinite run")
            result[key] = float(row["value"])
        return result

    left, right = index(baseline), index(candidate)
    if not left or left.keys() != right.keys():
        raise ValueError("Complete matched scene/seed/repeat coverage required")
    planned = [tuple(key) for key in planned_runs]
    if len(set(planned)) != len(planned) or set(planned) != set(left):
        raise ValueError("Complete predeclared run denominator required")
    scenes = defaultdict(list)
    for key in sorted(left):
        scenes[key[0]].append(right[key] - left[key])
    means = np.array([np.mean(values) for values in scenes.values()])
    rng = np.random.default_rng(seed)
    # Bounded bootstrap workspace, including large declared sample counts.
    boot = np.empty(samples)
    for start in range(0, samples, 100):
        end = min(start + 100, samples)
        boot[start:end] = rng.choice(means, size=(end - start, len(means))).mean(axis=1)
    return {
        "definition": "candidate minus baseline; scene-cluster bootstrap",
        "mean_delta": float(means.mean()),
        "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
        "independent_scenes": len(means),
        "matched_runs": len(left),
        "scene_deltas": {
            scene: float(np.mean(values)) for scene, values in scenes.items()
        },
        "bootstrap_seed": seed,
        "bootstrap_samples": samples,
        "warning": "Descriptive only; no automatic promotion or multiplicity correction",
    }
