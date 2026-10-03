"""Reproduce ESDNet/UHDM PTQ from the raw FP32 checkpoint and paired images.

This path freezes weights, smoothing and sparse masks throughout FAC. It never
retunes an exported mask or selects parameters on the test split. Numerical
reproduction of the published quantized scores remains an experimental goal.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import yaml
from quantdemoire.deploy import sha256
from quantdemoire.experiment import write_json

ROOT=Path(__file__).resolve().parents[1]


def validate_calibration_source(run, checkpoint, expected):
    """Do not label or combine a saved PTQ state with a different source recipe."""
    run=Path(run)
    provenance=json.loads((run/'provenance.json').read_text())
    if provenance['config_sha256']!=sha256(run/'config.yaml'):
        raise ValueError('saved calibration config was modified')
    if provenance['checkpoint_sha256']!=sha256(checkpoint):
        raise ValueError('export checkpoint differs from calibration checkpoint')
    saved=yaml.safe_load((run/'config.yaml').read_text())
    keys=('N_BITS_W','N_BITS_A','ACT_SAMPLE_RATE','INPUT_CHANNEL_SAMPLE_RATE',
          'ACT_SMALL_TENSOR_POLICY','CALIBRATION_AGGREGATION','SMOOTH_ALPHA',
          'WEIGHT_OUTLIER_PERCENTILE','WEIGHT_OUTLIER_MODE')
    for key in keys:
        if saved['QUANTIZATION'].get(key)!=expected['QUANTIZATION'].get(key):
            raise ValueError(f'export recipe differs from calibration: {key}')


def configuration(recipe,bits,policy,checkpoint,calibration,test,run,aggregation='mean',sampling_rate=.001,smooth_alpha=.5):
    config=yaml.safe_load((ROOT/f'config/ptq/{recipe}_w{bits}a{bits}.yaml').read_text())
    config['DATA'].update(CALI_DATASET=str(Path(calibration).resolve()),
        TRAIN_DATASET=str(Path(calibration).resolve()),TEST_DATASET=str(Path(test).resolve()))
    config['TEST']['LOAD_PATH']=str(Path(checkpoint).resolve())
    config['GENERAL'].update(SAVE_PREFIX=str(run.parent),EXP_NAME=run.name)
    config['QUANTIZATION'].update(ACT_SMALL_TENSOR_POLICY=policy,CALIBRATION_AGGREGATION=aggregation,
                                 QUANT_STATE_PATH=str(run/'quant_state.pth'),
                                 ACT_SAMPLE_RATE=sampling_rate,INPUT_CHANNEL_SAMPLE_RATE=sampling_rate)
    config['QUANTIZATION']['SMOOTH_ALPHA']=smooth_alpha
    return config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('calibrate','export','evaluate','all'),default='all')
    p.add_argument('--recipe',choices=('paper','deploy'),default='paper')
    p.add_argument('--bits',type=int,choices=(4,8),nargs='+',default=[4,8])
    p.add_argument('--small-tensor-policy',choices=('legacy','full_if_single','full_if_pooled'),default='legacy')
    p.add_argument('--aggregation',choices=('mean','extrema'),default='mean',
                   help='cross-image sampled-statistic reduction; the paper does not specify this detail')
    p.add_argument('--sampling-rate',type=float,default=.001,help='both gamma values; values other than .001 are adjusted recipes')
    p.add_argument('--smooth-alpha',type=float,default=.5,help='deployment-only smoothing experiments; paper recipe fixes .5')
    p.add_argument('--checkpoint',default='checkpoints/uhdm_checkpoint.pth')
    p.add_argument('--calibration',default='data/UHDM-paper-calibration/train')
    p.add_argument('--test',default='data/UHDM-full/test')
    p.add_argument('--artifacts',default='outputs/reproduced-ptq')
    p.add_argument('--models',default='checkpoints/reproduced-ptq')
    p.add_argument('--results',default='results/reproduced-ptq')
    p.add_argument('--backends',choices=('fake','fake_fp16','cuda'),nargs='+',default=['fake','cuda'])
    p.add_argument('--limit',type=int,help='development subset only; omit for the full result')
    a=p.parse_args()
    if not 0<a.sampling_rate<=1:p.error('--sampling-rate must lie in (0, 1]')
    if not 0<=a.smooth_alpha<=1:p.error('--smooth-alpha must lie in [0, 1]')
    if a.recipe=='paper' and a.smooth_alpha!=.5:p.error('use --recipe deploy for a smoothing experiment')
    def run_script(path,*args):
        command=[sys.executable,str(ROOT/path),*map(str,args)]
        print('Running:',command,flush=True)
        subprocess.run(command,cwd=ROOT,check=True)
    for bits in dict.fromkeys(a.bits):
        label=f'{a.recipe}-w{bits}a{bits}-{a.small_tensor_policy}'
        if a.aggregation!='mean':label+='-'+a.aggregation
        if a.sampling_rate!=.001:label+=f'-g{a.sampling_rate:g}'
        if a.smooth_alpha!=.5:label+=f'-a{a.smooth_alpha:g}'
        run=(Path(a.artifacts)/label).resolve();model=(Path(a.models)/label).resolve()
        results=(Path(a.results)/label).resolve()
        config=configuration(a.recipe,bits,a.small_tensor_policy,a.checkpoint,a.calibration,a.test,run,a.aggregation,a.sampling_rate,a.smooth_alpha)
        if a.stage in ('calibrate','all'):
            if (run/'quant_state.pth').exists():
                p.error(f'calibration already exists: {run}; use --stage export/evaluate or a new --artifacts directory')
            run.mkdir(parents=True,exist_ok=True)
            config_path=run/'config.yaml';config_path.write_text(yaml.safe_dump(config,sort_keys=False))
            with tarfile.open(run/'source.tar.gz','w:gz') as archive:
                files=[ROOT/'calibrate.py',ROOT/'uv.lock',ROOT/'pyproject.toml',Path(__file__)]
                files+=sorted((ROOT/'quantdemoire').rglob('*.py'))
                for path in files:archive.add(path,arcname=str(path.relative_to(ROOT)))
            run_script('calibrate.py','--config',config_path,'--calibrate-only')
        if a.stage in ('export','all'):
            state=run/'quant_state.pth'
            if not state.is_file():p.error(f'missing calibrated state: {state}')
            validate_calibration_source(run,a.checkpoint,config)
            if (model/'runtime.json').exists():
                p.error('output already has an execution profile; use a new --models directory to preserve its provenance')
            # Deliberately omit --tail-fraction and --weight-range overrides:
            # packing must preserve the calibrated sparse structure and values.
            run_script('script/export.py','--checkpoint',a.checkpoint,'--quant-state',state,
                       '--bits',bits,'--main-output','float16','--output',model)
            for name in ('provenance.json','calibration_history.json','config.yaml'):
                shutil.copyfile(run/name,model/name)
            write_json(model/'reproduction.json',dict(recipe=a.recipe,bits=bits,
                small_tensor_policy=a.small_tensor_policy,quant_state_sha256=sha256(state),
                calibration_aggregation=a.aggregation,
                gamma1=a.sampling_rate,gamma2=a.sampling_rate,
                smooth_alpha=a.smooth_alpha,
                source_archive_sha256=sha256(run/'source.tar.gz'),mask_policy='preserved from calibration',
                published_quantized_result_reproduced=False))
            run_script('script/verify_release.py','--model',model,'--checkpoint',a.checkpoint,
                       '--quant-state',state,'--output',results/'format.json')
        if a.stage in ('evaluate','all'):
            for backend in a.backends:
                options=['--limit',a.limit] if a.limit else []
                if backend=='fake':options+=['--memory-format','contiguous']
                run_script('script/evaluate.py','--backend',backend,'--model',model,'--data',a.test,
                    '--output',results/f'quality_{backend}.json','--resume',*options)


if __name__=='__main__':main()
