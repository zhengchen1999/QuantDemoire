"""Attach a validated benchmark configuration to a local model folder; never uploads."""
import argparse
import json
import shutil
from pathlib import Path
from quantdemoire.deploy import sha256
from quantdemoire.profiles import profile_environment


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True)
    p.add_argument('--benchmark',required=True);p.add_argument('--variant',required=True)
    p.add_argument('--validation',required=True)
    a=p.parse_args();folder=Path(a.model);benchmark=json.loads(Path(a.benchmark).read_text())
    # Check the model card before mutating runtime.json or the execution plan.
    # Exported calibration folders need package_model.py before this step.
    card_path=folder/'README.md'
    if not card_path.is_file():
        raise ValueError('model card is missing; create a local package with script/package_model.py first')
    card=card_path.read_text()
    start='<!-- quantdemoire-runtime-profile -->';end='<!-- /quantdemoire-runtime-profile -->'
    if (start in card)!=(end in card) or (start in card and card.index(end)<card.index(start)):
        raise ValueError('model card contains an incomplete runtime-profile block')
    validation=json.loads(Path(a.validation).read_text());config=json.loads((folder/'config.json').read_text())
    weight_hash=sha256(folder/'model.safetensors');config_hash=sha256(folder/'config.json')
    if not benchmark.get('complete') or not validation.get('complete'):raise ValueError('completed reports are required')
    if validation.get('count',0)<1 or validation['model_sha256']!=weight_hash or validation['config_sha256']!=config_hash:
        raise ValueError('validation does not match this model')
    if any(not all(row['bitwise_equal']) or not all(row.get('graph_bitwise_equal',[True])) for row in validation['rows']):
        raise ValueError('validation contains unequal outputs')
    if config['weights_sha256']!=weight_hash:raise ValueError('model checksum mismatch')
    variant=next(v for v in benchmark['config']['variants'] if v['name']==a.variant)
    if variant['backend']!='cuda':raise ValueError('a CUDA deployment benchmark is required')
    env={**benchmark['config'].get('environment',{}),**variant.get('environment',{})}
    measured=[json.loads(Path(row['raw']).read_text()) for row in benchmark['rows'] if row['variant']==a.variant]
    if len(measured)!=benchmark['rounds']:raise ValueError('benchmark rounds are incomplete')
    for result in measured:
        if result['model_sha256']!=weight_hash or result['config_sha256']!=config_hash:raise ValueError('benchmark model mismatch')
        if 'cudnn_autotune' in result and 'cudnn_autotune' in validation and result['cudnn_autotune']!=validation['cudnn_autotune']:
            raise ValueError('benchmark and validation use different cuDNN search policies')
        for key in ('deployment_source_sha256','kernel_source_sha256'):
            if result['environment'][key]!=validation['environment'][key]:raise ValueError('benchmark and validation used different code')
        for key in ('torch','cuda','cudnn','gpu','prepare_implementation','csr_kernel','joint_warps',
                    'joint_plan_sha256','fuse_attention','fuse_residual','fuse_next_quant','fuse_second_quant','compact_sparse','sparse_implementation','shared_skip_planar','joint_next_quant','balance_csr','shared_retain_stages','few_channels','packed_csr_meta','shared_skip_prefix_planar','omit_cached_features','release_features'):
            if result['environment'].get(key)!=validation['environment'].get(key):raise ValueError(f'benchmark/validation setting mismatch: {key}')
        if result['environment'].get('joint_variant',0)!=validation['environment'].get('joint_variant',0):
            raise ValueError('benchmark/validation joint variant mismatch')
        for report in (result,validation):
            override=report['environment'].get('main_half_override')
            if override is not None and any((spec.get('main_output')=='float16')!=(override=='1') for spec in config['layers'].values() if spec['kind']=='integer'):
                raise ValueError('main-output override differs from the saved model')
    env.setdefault('QUANTDEMOIRE_JOINT_VARIANT',str(measured[0]['environment'].get('joint_variant',0)))
    plan=env.pop('QUANTDEMOIRE_JOINT_PLAN',None)
    profile=dict(format='quantdemoire-runtime-v1',status='validated',model_sha256=weight_hash,config_sha256=config_hash,
        environment=env,measured=benchmark['summary'][a.variant],tested_environment=measured[0]['environment'],
        input=measured[0]['input'],benchmark_sha256=sha256(a.benchmark),validation_sha256=sha256(a.validation),
        verified_images=validation['count'],graph_verified=bool(validation.get('graph')),
        note='Execution tuning for the recorded GPU/software/input; source hashes are provenance, not a compatibility guarantee. No upload has been performed.')
    if plan:
        source=Path(plan);target=folder/'joint_plan.json'
        if source.resolve()!=target.resolve():shutil.copyfile(source,target)
        profile.update(joint_plan=target.name,joint_plan_sha256=sha256(target))
    (folder/'runtime.json').write_text(json.dumps(profile,indent=2)+'\n')
    profile_environment(folder)  # Validate the portable artifact before documenting it.
    if start in card:card=card[:card.index(start)]+card[card.index(end)+len(end):]
    block=f'''\n{start}
## Verified execution profile

`runtime.json` and `joint_plan.json` store the local CUDA execution configuration.
It was checked on {validation['count']} local UHDM images. The recorded median
latency is {profile['measured']['median_ms']:.3f} ms for the hardware and input
listed in `runtime.json`; peak allocated memory includes activations/workspaces.

```python
from quantdemoire.profiles import apply_execution_profile
from quantdemoire.precision import configure
from quantdemoire.deploy import load_model
apply_execution_profile("{folder.name}")  # preserves existing explicit environment overrides
configure()
model = load_model("{folder.name}", backend="cuda")
```

The packed weights and sparse structure are unchanged. Other hardware/input
shapes require fresh timing; the saved source hashes document the validation run.
{end}
'''
    (folder/'README.md').write_text(card.rstrip()+'\n'+block)
    manifest={p.name:sha256(p) for p in sorted(folder.iterdir()) if p.is_file() and p.name!='manifest.json'}
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(dict(model=str(folder),runtime_sha256=sha256(folder/'runtime.json'),weights_sha256=weight_hash))


if __name__=='__main__':main()
