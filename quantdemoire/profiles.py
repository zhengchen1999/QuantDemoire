"""Apply a portable, weight-bound execution profile before loading a model."""
import hashlib
import json
import os
from pathlib import Path


_VALUES = {
    'QUANTDEMOIRE_OMIT_CACHED_FEATURES': {'0','1','2'},
    'QUANTDEMOIRE_PREPARE_IMPL': {'scalar','cached','pair','quad','quad_loop','quad_swizzle'},
    'QUANTDEMOIRE_CSR_KERNEL': {'legacy','linear','lut','wide','linear32','divmod'},
    'QUANTDEMOIRE_SPARSE_IMPL': {'csr','tensorcore'},
    'QUANTDEMOIRE_JOINT_NEXT_QUANT': {'0','1','2'},
    'QUANTDEMOIRE_JOINT_WARPS': {'0','4','8'},
    'QUANTDEMOIRE_JOINT_VARIANT': {str(i) for i in range(10)},
    **{name:{'0','1'} for name in (
        'QUANTDEMOIRE_RELEASE_FEATURES','QUANTDEMOIRE_COMPACT_SPARSE','QUANTDEMOIRE_MAIN_HALF',
        'QUANTDEMOIRE_FUSE_NEXT_QUANT','QUANTDEMOIRE_FUSE_ATTENTION','QUANTDEMOIRE_FUSE_RESIDUAL',
        'QUANTDEMOIRE_SHARED_SKIP_PLANAR','QUANTDEMOIRE_SHARED_SKIP_PREFIX_PLANAR','QUANTDEMOIRE_BALANCE_CSR','QUANTDEMOIRE_SHARED_RETAIN_STAGES','QUANTDEMOIRE_FEW_CHANNELS','QUANTDEMOIRE_PACKED_CSR_META','QUANTDEMOIRE_FUSE_SECOND_QUANT')},
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def profile_environment(directory):
    """Validate the artifact and return its environment without changing it."""
    directory=Path(directory).resolve()
    profile=json.loads((directory/'runtime.json').read_text())
    if profile.get('format')!='quantdemoire-runtime-v1':raise ValueError('unsupported runtime profile')
    if profile.get('model_sha256')!=_sha(directory/'model.safetensors'):
        raise ValueError('runtime profile weight checksum mismatch')
    if profile.get('config_sha256')!=_sha(directory/'config.json'):
        raise ValueError('runtime profile configuration checksum mismatch')
    env=dict(profile['environment'])
    for key,value in env.items():
        if key not in _VALUES or not isinstance(value,str) or value not in _VALUES[key]:raise ValueError(f'invalid runtime setting: {key}')
    if profile.get('joint_plan'):
        plan=(directory/profile['joint_plan']).resolve()
        if not plan.is_relative_to(directory):raise ValueError('joint plan must be inside the model folder')
        if profile.get('joint_plan_sha256')!=_sha(plan):raise ValueError('joint plan checksum mismatch')
        data=json.loads(plan.read_text())
        if data.get('model_sha256')!=profile['model_sha256']:raise ValueError('joint plan belongs to different weights')
        env['QUANTDEMOIRE_JOINT_PLAN']=str(plan)
    return env


def apply_execution_profile(directory, *, overwrite=False):
    """Apply saved flags; preserve explicit existing environment overrides by default.

    Call before load_model(). The profile stores measurements for the recorded
    GPU, software and UHDM shapes; it does not guarantee speed on other systems.
    """
    env=profile_environment(directory)
    for key,value in env.items():
        if overwrite or key not in os.environ:os.environ[key]=value
    return {key:os.environ[key] for key in env}
