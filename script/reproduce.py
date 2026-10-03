"""Historical deployment tuning protocol (includes optional weight QAT).

For the raw-checkpoint, activation-bound-only PTQ recipe, use reproduce_ptq.py.
This historical path changes the mask during export and is not paper PTQ.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['export','polish','configure','evaluate','benchmark'], required=True)
    p.add_argument('--checkpoint', default='checkpoints/uhdm_checkpoint.pth')
    p.add_argument('--state4', help='existing calibrated W4A4 quant_state.pth')
    p.add_argument('--state8', help='existing calibrated W8A8 quant_state.pth')
    p.add_argument('--train-pool', help='training images including held-out tuning pairs')
    p.add_argument('--calibration', default='data/UHDM/train')
    p.add_argument('--test', default='data/UHDM/test')
    p.add_argument('--models', default='checkpoints/reproduced')
    p.add_argument('--results', default='results/reproduced')
    p.add_argument('--limit', type=int)
    p.add_argument('--full-fake-fp32',action='store_true',help='also run the expensive FP32 fake oracle over the test split')
    p.add_argument('--epochs',type=int,default=4)
    p.add_argument('--main-output',choices=['float16','float32'],default='float16')
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    def run(script, *args):
        command = [sys.executable, str(root/'script'/script), *map(str,args)]
        print(' '.join(command), flush=True)
        subprocess.run(command, check=True, cwd=root)
    models, results = Path(a.models), Path(a.results)
    if a.stage == 'export':
        if not a.state4 or not a.state8 or not a.train_pool:
            p.error('export requires --state4, --state8 and --train-pool')
        for bits,state,method in [(4,a.state4,'mse'),(8,a.state8,'minmax')]:
            base, target = models/f'base-w{bits}a{bits}', models/f'w{bits}a{bits}'
            run('export.py', '--checkpoint',a.checkpoint,'--quant-state',state,'--bits',bits,
                '--tail-fraction',.0025,'--weight-range',method,'--output',base)
            factors = [.9,1.,1.15,1.3] if bits==4 else [1.,1.25,1.5,1.75]
            run('tune_ranges.py','--model',base,'--checkpoint',a.checkpoint,'--pool',a.train_pool,
                '--calibration',a.calibration,'--factors',*factors,'--backend','fake','--output',target)
            run('verify_release.py','--model',target,'--checkpoint',a.checkpoint,
                '--output',results/f'format_w{bits}.json')
    elif a.stage == 'polish':
        if not a.train_pool:p.error('polish requires --train-pool for held-out TRAINING selection')
        if a.epochs<1:p.error('--epochs must be positive')
        training=models/'qat-w4a4'
        run('polish.py','--model',models/'w4a4','--checkpoint',a.checkpoint,
            '--data',a.calibration,'--output',training,'--train-weights',
            '--epochs',a.epochs,'--count',200,'--crop',512,'--lr',.001,'--weight-lr',1e-5)
        # Evaluate every epoch on the same held-out training subset with CUDA.
        # The test split is never used to select a candidate.
        candidates=[]
        for epoch in range(1,a.epochs+1):
            candidate=training/f'epoch{epoch}'
            scored=training/f'scored-epoch{epoch}'
            run('tune_ranges.py','--model',candidate,'--checkpoint',a.checkpoint,
                '--pool',a.train_pool,'--calibration',a.calibration,'--backend','cuda',
                '--factors',1,'--output',scored)
            score=json.loads((scored/'range_search.json').read_text())['trials'][0]
            candidates.append(dict(epoch=epoch,model=str(candidate),psnr=score['psnr'],gap=score['gap']))
        selected=max(candidates,key=lambda row:row['psnr'])
        target=models/'w4a4'
        for name in ('model.safetensors','config.json','training.json'):
            shutil.copyfile(Path(selected['model'])/name,target/name)
        # Keep the original range search and the epoch selection as separate provenance.
        (target/'selection.json').write_text(json.dumps(dict(
            criterion='CUDA PSNR on 24 held-out training images, seed 2026, center crop 512',
            candidates=candidates,selected=selected),indent=2)+'\n')
        run('verify_release.py','--model',target,'--checkpoint',a.checkpoint,
            '--output',results/'format_w4.json')
    elif a.stage == 'configure':
        for bits in (4,8):
            path=models/f'w{bits}a{bits}'/'config.json'
            config=json.loads(path.read_text())
            for layer in config['layers'].values():
                if layer['kind']=='integer':layer['main_output']=a.main_output
            path.write_text(json.dumps(config,indent=2)+'\n')
    elif a.stage == 'evaluate':
        jobs=[('fp32',0),('fp16',0),('fake_fp16',4),('fake_fp16',8),('cuda',4),('cuda',8)]
        if a.full_fake_fp32:jobs += [('fake',4),('fake',8)]
        for backend,bits in jobs:
            model = models/f'w{bits}a{bits}' if bits else a.checkpoint
            extra = ['--limit',a.limit] if a.limit else []
            run('evaluate.py','--backend',backend,'--model',model,'--data',a.test,
                '--output',results/f'quality_{backend}{bits or ""}.json','--resume',*extra)
    else:
        for bits in (4,8):
            run('benchmark_kernel.py','--model',models/f'w{bits}a{bits}','--bits',bits,
                '--height',2160,'--width',3840,'--repeats',5,'--output',results/f'kernel_w{bits}.json')
        run('summarize_kernels.py','--input',results/'kernel_w4.json',results/'kernel_w8.json',
            '--output',results/'kernel_summary.json')
        for graph in (False,True):
            for backend,bits in [('fp16',0),('cuda',4),('cuda',8)]:
                run('benchmark_e2e.py','--backend',backend,'--model',models/f'w{bits}a{bits}' if bits else a.checkpoint,
                    '--output',results/f'e2e_{"graph" if graph else "eager"}_{bits or "fp16"}.json',
                    *(['--graph'] if graph else []))
        run('benchmark_memory.py','--checkpoint',a.checkpoint,'--w4',models/'w4a4','--w8',models/'w8a8',
            '--output',results/'memory.json')


if __name__ == '__main__':
    main()
