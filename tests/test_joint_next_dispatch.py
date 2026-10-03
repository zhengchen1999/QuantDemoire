"""The producer-aware optimizer must agree with exhaustive short-chain search."""
import importlib.util
import itertools
from pathlib import Path
import random
import pytest


def test_joint_next_dispatch_matches_exhaustive_search():
    path=Path(__file__).resolve().parents[1]/'script/tune_joint_next.py'
    spec=importlib.util.spec_from_file_location('joint_next_tuner',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rng=random.Random(996)
    for length in range(1,7):
        for trial in range(5):
            chain=[str(i) for i in range(length)];rows={}
            for i,name in enumerate(chain):
                modes=list(module.MODES) if i+1<length else ['separate','joint']
                next_keys=('0','8') if i+1<length else ('none',)
                rows[name]={'modes':modes,'timing':{
                    kind:{mode:{key:{'median_ms':rng.random()} for key in next_keys} for mode in modes}
                    for kind in ('full','prefix')}}
            def score(path):
                total=0.
                for i,mode in enumerate(path):
                    kind='prefix' if i and path[i-1]!='joint' else 'full'
                    key='none' if i+1==length else ('0' if path[i+1]=='separate' else '8')
                    costs=rows[str(i)]['timing'][kind]
                    total+=costs[mode][key]['median_ms']
                    if mode!='separate':total+=max(.001,.01*costs['separate'][key]['median_ms'])
                    if mode in ('direct','transpose'):total+=.001
                return total
            expected=min(score(path) for path in itertools.product(*(rows[name]['modes'] for name in chain)))
            chosen,actual=module.solve_chain(chain,rows,.01,.001,.001)
            assert actual==pytest.approx(expected,abs=1e-12)
            assert score(tuple(chosen[name] for name in chain))==pytest.approx(expected,abs=1e-12)
