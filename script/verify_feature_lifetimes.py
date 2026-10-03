"""Verify outer feature and six level-input references are released with unchanged outputs."""
import os
os.environ['TORCH_CUDNN_V8_API_DISABLED']='1'
import argparse,json,weakref
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from quantdemoire.profiles import apply_execution_profile
from quantdemoire.precision import configure,metadata
from quantdemoire import experiment
from quantdemoire.deploy import sha256
from quantdemoire.runtime import CudaGraphRunner
from quantdemoire.esdnet import _EncoderFeatures as FeatureQueue, _LevelInput as FeatureInput
from quantdemoire import esdnet

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=('cuda','fp16'),required=True)
    p.add_argument('--model',required=True);p.add_argument('--data',default='data/UHDM/test');p.add_argument('--limit',type=int,default=100);p.add_argument('--output',required=True)
    args=p.parse_args();configure()
    if args.backend=='cuda':apply_execution_profile(args.model)
    model=experiment.model_for(args.backend,args.model)
    state=dict(enabled=True,source_sha256=sha256(Path(esdnet.__file__)))
    def mode(enabled):
        state['enabled']=enabled
        model.release_features=enabled;model.encoder.release_features=enabled
        os.environ['QUANTDEMOIRE_RELEASE_FEATURES']=str(int(enabled))
    refs=[];observed=[];level_refs={};level_observed=[]
    def remember(m,inputs,outputs):refs[:]=[weakref.ref(t) for t in outputs]
    def observe(m,inputs):observed.append(dict(enabled=state['enabled'],alive=[r() is not None for r in refs]))
    def install_hooks():
        handles=[model.encoder.register_forward_hook(remember),model.decoder.decoder_1.register_forward_pre_hook(observe)]
        for name in ('encoder.encoder_1','encoder.encoder_2','encoder.encoder_3','decoder.decoder_3','decoder.decoder_2','decoder.decoder_1'):
            level=model.get_submodule(name)
            def before(m,args,name=name):
                value=args[0].tensor if isinstance(args[0],FeatureInput) else args[0]
                level_refs[name]=weakref.ref(value)
            def after_rdb(m,args,name=name):
                level_observed.append(dict(name=name,enabled=state['enabled'],alive=level_refs[name]() is not None))
            handles.append(level.register_forward_pre_hook(before))
            handles.append(level.sam_blocks[0].register_forward_pre_hook(after_rdb))
        return handles
    hooks=install_hooks()
    paths=sorted(Path(args.data).rglob('*_moire.jpg'))[:args.limit]
    if not paths:raise ValueError('empty local dataset')
    rows=[];runner=None
    report=dict(complete=False,note=__doc__,backend=args.backend,model=args.model,rows=rows,
        source_sha256=state['source_sha256'],expected_count=len(paths),
        model_sha256=sha256(Path(args.model)/'model.safetensors' if args.backend=='cuda' else args.model),
        config_sha256=sha256(Path(args.model)/'config.json') if args.backend=='cuda' else None)
    for path in paths:
        with Image.open(path) as im:x=torch.from_numpy(np.asarray(im.convert('RGB'),dtype=np.float32)/255).permute(2,0,1)[None].cuda()
        x,_=experiment.pad_image(x);x=x.half().contiguous(memory_format=torch.channels_last)
        mode(False);expected=model(x)
        assert observed[-1]['alive']==[True,True,True],observed[-1]
        assert len(level_observed)>=6 and all(r['alive'] for r in level_observed[-6:]),level_observed[-6:]
        mode(True);actual=model(x)
        assert observed[-1]['alive']==[False,False,False],observed[-1]
        assert all(not r['alive'] for r in level_observed[-6:]),level_observed[-6:]
        assert len(actual)==len(expected)==3
        equal=[torch.equal(a,b) for a,b in zip(actual,expected)];assert all(equal),path
        # Instrumentation is removed for Graph capture; CPU hooks would only run
        # during capture, not replay. Eager weak references establish last use.
        if hooks:
            for hook in hooks:hook.remove()
            hooks=[]
        if runner is None:runner=CudaGraphRunner(model,x)
        captured=runner(x);graph_equal=[torch.equal(a,b) for a,b in zip(captured,expected)]
        assert all(graph_equal),path
        # Reinstall lightweight eager observers for the next changing image.
        hooks=install_hooks()
        rows.append(dict(image=str(path.relative_to(args.data)),input_sha256=sha256(path),
            bitwise_equal=equal,graph_bitwise_equal=graph_equal,encoder_features_released=True,all_six_level_inputs_released=True))
        report.update(count=len(rows),environment=metadata())
        experiment.write_json(args.output,report)
        print(len(rows),'all outputs exact; all encoder feature references released',flush=True)
        del x,expected,actual,captured
    for hook in hooks:hook.remove()
    # The old three-tensor Decoder API remains supported, even when enabled.
    tiny=torch.rand(1,3,64,64,device='cuda',dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    features=model.encoder(tiny)
    mode(False);expected=model.decoder(*features)
    mode(True);actual=model.decoder(*features)
    assert all(torch.equal(a,b) for a,b in zip(actual,expected))
    box=FeatureInput(features[0]);assert box.pop() is features[0]
    try:box.pop()
    except ValueError:pass
    else:raise AssertionError('consumed level input accepted')
    empty=FeatureQueue(features);empty.parts.clear()
    try:model.decoder(empty)
    except ValueError:pass
    else:raise AssertionError('reused feature queue accepted')
    report.update(complete=True,direct_decoder_api=True,queue_reuse_guard=True,environment=metadata())
    experiment.write_json(args.output,report)

if __name__=='__main__':main()
