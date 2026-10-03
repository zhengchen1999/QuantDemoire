"""Copy an architecture-compatible execution schedule as an UNMEASURED proposal.

This never transfers speed, memory, correctness, or branch-budget claims. Run
verify_graph_fusions.py and benchmark scripts for the new weights before using
attach_runtime_profile.py to create a validated release profile.
"""
import argparse
import json
from pathlib import Path
from quantdemoire.deploy import sha256
from quantdemoire.experiment import write_json
from quantdemoire.profiles import profile_environment


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--model',required=True)
    p.add_argument('--release-features',type=int,choices=(0,1),help='explicitly select early feature release for the proposal')
    a=p.parse_args();source=Path(a.source).resolve();target=Path(a.model).resolve()
    if (target/'runtime.json').exists() or (target/'joint_plan.json').exists():p.error('target already has a profile or plan')
    profile_environment(source)
    old=json.loads((source/'config.json').read_text());new=json.loads((target/'config.json').read_text())
    def geometry(config):
        common=('kind','shape','stride','padding','dilation')
        return {name:{k:spec.get(k) for k in (*common,*(('bits','padded_channels','main_output') if spec['kind']=='integer' else ()))} for name,spec in config['layers'].items()}
    if geometry(old)!=geometry(new):p.error('model architectures or numerical main-output conventions differ')
    original=json.loads((source/'runtime.json').read_text());weight_hash=sha256(target/'model.safetensors')
    environment=dict(original['environment'])
    # Geometry comparison includes the saved main-output convention. A redundant
    # override would obscure which numerical format quality evaluation uses.
    override=environment.pop('QUANTDEMOIRE_MAIN_HALF',None)
    if override is not None and any((s.get('main_output')=='float16')!=(override=='1') for s in new['layers'].values() if s['kind']=='integer'):
        p.error('source profile overrides the saved numerical format')
    if a.release_features is not None:environment['QUANTDEMOIRE_RELEASE_FEATURES']=str(a.release_features)
    profile=dict(format='quantdemoire-runtime-v1',model_sha256=weight_hash,config_sha256=sha256(target/'config.json'),
        environment=environment,status='unmeasured_proposal',
        source_profile_sha256=sha256(source/'runtime.json'),source_model_sha256=original['model_sha256'],
        note=__doc__)
    if original.get('joint_plan'):
        plan_path=source/original['joint_plan'];old_plan=json.loads(plan_path.read_text())
        plan=dict(format='quantdemoire-joint-plan-v1',model_sha256=weight_hash,layers=old_plan['layers'],
            status='unmeasured_proposal',source_plan_sha256=sha256(plan_path),source_model_sha256=original['model_sha256'],note=__doc__)
        if 'joint_next_modes' in old_plan:plan['joint_next_modes']=old_plan['joint_next_modes']
        write_json(target/'joint_plan.json',plan)
        profile.update(joint_plan='joint_plan.json',joint_plan_sha256=sha256(target/'joint_plan.json'))
    write_json(target/'runtime.json',profile);profile_environment(target)
    print(dict(model=str(target),status=profile['status'],model_sha256=weight_hash))


if __name__=='__main__':main()
