"""Compare consumer-aware dispatch with exhaustive search on short chains."""
import importlib.util
import itertools
from pathlib import Path
import random

import pytest


@pytest.mark.parametrize('legacy', [False, True])
def test_dispatch_matches_exhaustive_search(legacy):
    path = Path(__file__).resolve().parents[1] / 'script/tune_joint_dispatch.py'
    spec = importlib.util.spec_from_file_location('dispatch_tuner', path)
    tuner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tuner)
    rng = random.Random(915)
    for length in range(1, 8):
        for _ in range(12):
            chain = [str(i) for i in range(length)]
            rows = {}
            for i, name in enumerate(chain):
                timing = {}
                for kind in ('full', 'prefix'):
                    timing[kind] = {}
                    for mode in (0, 8):
                        next_keys = ('none',) if i == length - 1 else ('0', '8')
                        timing[kind][str(mode)] = (
                            {'median_ms': rng.random()} if legacy else
                            {key: {'median_ms': rng.random()} for key in next_keys}
                        )
                rows[name] = {'timing': timing}

            def score(modes):
                total = 0.
                for i, mode in enumerate(modes):
                    kind = 'prefix' if i and modes[i - 1] == 0 else 'full'
                    costs = rows[str(i)]['timing'][kind]

                    def get(which):
                        entry = costs[str(which)]
                        key = 'none' if i + 1 == length else str(modes[i + 1])
                        return (entry if legacy else entry[key])['median_ms']

                    total += get(mode) + (max(.001, .01 * get(0)) if mode else 0.)
                return total

            expected = min(score(modes) for modes in itertools.product((0, 8), repeat=length))
            chosen, actual = tuner.solve_chain(chain, rows, .01, .001)
            assert actual == pytest.approx(expected, abs=1e-12)
            assert score(tuple(chosen[name] for name in chain)) == pytest.approx(expected, abs=1e-12)
