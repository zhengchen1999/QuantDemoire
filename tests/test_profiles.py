import hashlib
import json
import os
import pytest
from quantdemoire.profiles import apply_execution_profile,profile_environment


@pytest.fixture
def package(tmp_path):
    folder=tmp_path/'model';folder.mkdir()
    (folder/'model.safetensors').write_bytes(b'opaque weights for profile checksum tests')
    (folder/'config.json').write_text('{}')
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    weight=sha(folder/'model.safetensors')
    (folder/'joint_plan.json').write_text(json.dumps(dict(format='quantdemoire-joint-plan-v1',model_sha256=weight,layers={})))
    p=dict(format='quantdemoire-runtime-v1',model_sha256=weight,config_sha256=sha(folder/'config.json'),
           environment={'QUANTDEMOIRE_PREPARE_IMPL':'quad_swizzle'},joint_plan='joint_plan.json',
           joint_plan_sha256=sha(folder/'joint_plan.json'))
    (folder/'runtime.json').write_text(json.dumps(p))
    return folder


def test_portable_profile_and_explicit_overrides(package,monkeypatch):
    moved=package.with_name('downloaded');package.rename(moved)
    monkeypatch.setenv('QUANTDEMOIRE_PREPARE_IMPL','pair')
    monkeypatch.delenv('QUANTDEMOIRE_JOINT_PLAN',raising=False)
    apply_execution_profile(moved)
    assert os.environ['QUANTDEMOIRE_PREPARE_IMPL']=='pair'
    assert os.environ['QUANTDEMOIRE_JOINT_PLAN']==str(moved/'joint_plan.json')
    apply_execution_profile(moved,overwrite=True)
    assert os.environ['QUANTDEMOIRE_PREPARE_IMPL']=='quad_swizzle'


@pytest.mark.parametrize('name',['model.safetensors','config.json','joint_plan.json'])
def test_profile_rejects_modified_artifacts(package,name):
    with (package/name).open('ab') as f:f.write(b' ')
    with pytest.raises(ValueError,match='checksum mismatch'):profile_environment(package)


def test_profile_rejects_unrelated_environment_and_outside_plan(package):
    p=package/'runtime.json';original=json.loads(p.read_text())
    changed={**original,'environment':{'CUDA_HOME':'/arbitrary/toolchain'}};p.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='invalid runtime setting'):profile_environment(package)
    changed={**original,'joint_plan':'../outside.json'};p.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='inside the model folder'):profile_environment(package)
