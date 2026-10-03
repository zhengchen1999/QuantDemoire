"""Hash the actual calibration, tuning and test subsets and check pair overlap."""
import argparse
import json
import random
from pathlib import Path
from quantdemoire.deploy import sha256
from quantdemoire.experiment import write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--calibration',default='data/UHDM/train')
    p.add_argument('--test',default='data/UHDM/test')
    p.add_argument('--pool')
    p.add_argument('--search',help='existing range_search.json with the held-out training image list')
    p.add_argument('--heldout-count',type=int,default=0,help='select a new disjoint training holdout')
    p.add_argument('--heldout-seed',type=int,default=2026)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if (a.search or a.heldout_count) and not a.pool:p.error('held-out selection requires --pool')
    if a.search and a.heldout_count:p.error('choose an existing search or a new holdout')
    if a.heldout_count<0:p.error('--heldout-count must be nonnegative')
    def rows(root,paths):
        output=[]
        for gt in paths:
            inp=gt.with_name(gt.name.replace('_gt.jpg','_moire.jpg'))
            output.append(dict(image=str(gt.relative_to(root)),gt_sha256=sha256(gt),input_sha256=sha256(inp)))
        return output
    result={name:rows(root,sorted(Path(root).rglob('*_gt.jpg'))) for name,root in [('calibration',a.calibration),('test',a.test)]}
    result['heldout_training']=[]
    if a.search:
        search=json.loads(Path(a.search).read_text())
        names=[r['image'] for r in search['trials'][0]['rows']]
        result['heldout_training']=rows(a.pool,[Path(a.pool)/name for name in names])
    elif a.heldout_count:
        known={(r['gt_sha256'],r['input_sha256']) for r in result['calibration']}
        candidates=sorted(Path(a.pool).rglob('*_gt.jpg'))
        random.Random(a.heldout_seed).shuffle(candidates)
        for gt in candidates:
            row=rows(a.pool,[gt])[0]
            key=(row['gt_sha256'],row['input_sha256'])
            if key not in known:
                known.add(key);result['heldout_training'].append(row)
            if len(result['heldout_training'])==a.heldout_count:break
        if len(result['heldout_training'])!=a.heldout_count:raise ValueError('not enough disjoint training pairs')
    sets={k:{(r['gt_sha256'],r['input_sha256']) for r in v} for k,v in result.items()}
    for first,second in [('calibration','test'),('calibration','heldout_training'),('heldout_training','test')]:
        if sets[first]&sets[second]:raise ValueError(f'pair overlap between {first} and {second}')
    result['counts']={k:len(v) for k,v in sets.items()}
    result['pair_overlap']=False
    if a.heldout_count:result['heldout_selection']=dict(seed=a.heldout_seed,pool=str(Path(a.pool).resolve()))
    write_json(a.output,result)
    print(result['counts'])


if __name__=='__main__':main()
