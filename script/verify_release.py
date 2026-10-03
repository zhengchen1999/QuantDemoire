"""Validate the portable format, integer packing and coverage without Hub upload."""
import argparse
import json
import math
from pathlib import Path
import torch
from quantdemoire.deploy import load_model,PackedConv,sha256,unpack_signed
from quantdemoire.experiment import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True)
    p.add_argument('--checkpoint',help='verify actual ranked tails against the source FP32 checkpoint')
    p.add_argument('--quant-state',help='also prove mask, FP16 values and integer codes match the calibrated state exactly')
    a=p.parse_args();torch.set_num_threads(4)
    model=load_model(a.model,'fake','cpu',fuse_relu=False)
    report=dict(model=a.model,weights_sha256=sha256(Path(a.model)/'model.safetensors'),layers=[])
    config=json.loads((Path(a.model)/'config.json').read_text())
    source=None
    states=None
    if a.quant_state:
        if sha256(a.quant_state)!=config['calibration_sha256']:raise ValueError('calibration state hash differs')
        if config.get('requested_tail_fraction') is not None or config.get('weight_range','stored')!='stored':
            raise ValueError('export retuned the mask or weights; cannot claim calibration identity')
        states=torch.load(a.quant_state,map_location='cpu',weights_only=True)
    if a.checkpoint:
        if sha256(a.checkpoint)!=config['checkpoint_sha256']:raise ValueError('source checkpoint hash differs')
        source=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    for name,m in model.named_modules():
        if isinstance(m,PackedConv):
            assert m.spec['lower_tail_count'] >= m.spec['min_tail_fraction']*torch.tensor(m.spec['shape']).prod().item()
            assert m.spec['upper_tail_count'] >= m.spec['min_tail_fraction']*torch.tensor(m.spec['shape']).prod().item()
            assert torch.isfinite(m.dense_weight()).all()
            k,c,r,_=m.spec['shape'];cp=m.spec['padded_channels']
            codes=unpack_signed(m.packed_weight,m.spec['bits']).reshape(k,r,r,cp)
            assert not codes[...,c:].any(), 'nonzero padded weight channels'
            mask=torch.zeros(k,c*r*r,dtype=torch.bool)
            for oc in range(k):mask[oc,m.indices[m.row[oc]:m.row[oc+1]].long()]=True
            assert not codes[...,:c].reshape(k,-1)[mask].any(), 'integer/outlier double counting'
            if states is not None:
                state=states[name]
                expected_mask=state['outlier_mask'].permute(0,2,3,1).contiguous().reshape(k,-1)
                assert torch.equal(mask,expected_mask), 'calibrated sparse mask changed during export'
                calibrated=state['quantized_weight'].permute(0,2,3,1).contiguous().reshape(k,-1)
                assert torch.equal(m.values,calibrated[mask].half()), 'calibrated FP16 tail values changed'
                assert torch.equal(m.weight_scale,state['weight_scale'].flatten()), 'calibrated weight scales changed'
                normal=(calibrated/state['weight_scale'].reshape(k,1)).round().masked_fill(mask,0).to(torch.int8)
                assert torch.equal(codes[...,:c].reshape(k,-1),normal), 'calibrated integer codes changed'
                assert torch.equal(m.inverse_smooth,state['smooth_scale'].float().reciprocal()), 'smoothing changed'
                from quantdemoire.quantization.observers import calculate_qparams_from_bounds
                act,z,_,_=calculate_qparams_from_bounds(state['lb_a'].float(),state['ub_a'].float(),m.spec['bits'],False)
                assert torch.equal(m.activation,torch.stack([act,z]).flatten()), 'activation quantizer changed'
            if source is not None:
                original=source[name+'.weight']/m.inverse_smooth[None,:,None,None]
                order=original.flatten().argsort(stable=True)
                original_mask=mask.reshape(k,r,r,c).permute(0,3,1,2).contiguous().flatten()
                count=math.ceil(original.numel()*.0025)
                assert original_mask[order[:count]].all(), 'missing lower-tail original weight'
                assert original_mask[order[-count:]].all(), 'missing upper-tail original weight'
            report['layers'].append(dict(name=name,shape=m.spec['shape'],bits=m.spec['bits'],outliers=m.values.numel(),
                                        tensor_bytes=sum(t.numel()*t.element_size() for t in m.buffers())))
    assert len(report['layers'])==157
    report['verified']=True
    report['calibration_identity_verified']=states is not None
    report['quant_state_sha256']=sha256(a.quant_state) if states is not None else None
    report['tail_verification']='ranked source weights' if source is not None else 'metadata counts only; supply --checkpoint for ranked coverage'
    write_json(a.output,report)


if __name__=='__main__':main()
