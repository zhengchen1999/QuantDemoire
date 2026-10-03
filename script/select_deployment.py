"""Freeze a deployment candidate using matching held-out-training quality reports.

Selection does not assert that branch budgets or published paper scores pass.
The report explicitly preserves each candidate's quality and branch cost.
"""
import argparse
import json
from pathlib import Path
from quantdemoire.deploy import sha256
from quantdemoire.experiment import write_json


def select(qualities,branches):
    candidates=[];reference=None
    for quality_file,branch_file in zip(qualities,branches,strict=True):
        q=json.loads(Path(quality_file).read_text());b=json.loads(Path(branch_file).read_text())
        if not q.get('complete') or q.get('selection_subset')!='heldout_training' or not q.get('selection_manifest_sha256'):
            raise ValueError('selection requires completed, manifest-bound heldout training evaluation')
        if not b.get('complete') or q['model_sha256']!=b['model_sha256'] or q['config_sha256']!=b['config_sha256']:
            raise ValueError('branch measurement and quality must cover the same model')
        signature=(q['selection_manifest_sha256'],q['crop'],q['protocol'],
                   [(r['image'],r['gt_sha256'],r['input_sha256']) for r in q['rows']])
        if reference is None:reference=signature
        elif signature!=reference:raise ValueError('candidates use different heldout data or quality protocols')
        candidates.append(dict(model=q['model'],model_sha256=q['model_sha256'],config_sha256=q['config_sha256'],
            quality_report=str(quality_file),quality_report_sha256=sha256(quality_file),mean=q['mean'],
            branch_report=str(branch_file),branch_report_sha256=sha256(branch_file),R_core=b['R_core'],
            aggregate_below_30pct=b['aggregate_below_30pct']))
    if not candidates:raise ValueError('no candidates')
    chosen=max(candidates,key=lambda r:r['mean']['psnr'])
    return dict(rule='highest PSNR on the same heldout training subset; no test evaluation used',
        candidates=candidates,selected=chosen,
        branch_target_passed=chosen['aggregate_below_30pct'],paper_result_reproduced=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qualities',nargs='+',required=True);p.add_argument('--branches',nargs='+',required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    if Path(a.output).exists():p.error('selection already frozen; use a new output to preserve it')
    write_json(a.output,select(a.qualities,a.branches))


if __name__=='__main__':main()
