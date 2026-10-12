"""Export calibrated integer weights and verify their exact provenance."""
import json
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace
import torch
from quantdemoire.deploy import export_legacy, load_model, PackedConv, sha256, unpack_signed
from quantdemoire.experiment import read_json, write_json
from quantdemoire.profiles import profile_environment


def export_model(checkpoint, quant_state, bits, output):
    torch.set_num_threads(4)
    config = export_legacy(checkpoint, quant_state, bits, output)
    for layer in config['layers'].values():
        if layer['kind'] == 'integer':
            layer['main_output'] = 'float16'
    write_json(Path(output) / 'config.json', config)


def verify_export(model, checkpoint, quant_state, output):
    a = SimpleNamespace(model=model, checkpoint=checkpoint, quant_state=quant_state, output=output)
    torch.set_num_threads(4)
    model=load_model(a.model,'fake','cpu',fuse_relu=False)
    report=dict(model=str(a.model),weights_sha256=sha256(Path(a.model)/'model.safetensors'),layers=[])
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


def prepare_runtime(folder, bits):
    folder = Path(folder)
    if (folder / "runtime.json").exists():
        profile_environment(folder)
        return
    template_path = Path.cwd() / f"config/release/w{bits}a{bits}.json"
    template = read_json(template_path)
    config = read_json(folder / "config.json")
    geometry = {}
    for name, spec in config["layers"].items():
        keys = ("kind", "shape", "stride", "padding", "dilation")
        if spec["kind"] == "integer":
            keys += ("bits", "padded_channels", "main_output")
        geometry[name] = {key: spec.get(key) for key in keys}
    geometry_hash = hashlib.sha256(json.dumps(geometry, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if geometry_hash != template["geometry_sha256"]:
        raise ValueError("Model geometry differs from the ESDNet execution template")
    weight_hash = sha256(folder / "model.safetensors")
    plan = dict(template["joint_plan"], format="quantdemoire-joint-plan-v1",
                model_sha256=weight_hash, status="unmeasured_proposal")
    write_json(folder / "joint_plan.json", plan)
    write_json(folder / "runtime.json", dict(
        format="quantdemoire-runtime-v1", status="unmeasured_proposal",
        model_sha256=weight_hash, config_sha256=sha256(folder / "config.json"),
        environment=template["environment"], joint_plan="joint_plan.json",
        joint_plan_sha256=sha256(folder / "joint_plan.json"),
        template_sha256=sha256(template_path),
        note="Execution settings only. Use this model's new quality, validation, speed and memory reports."))
    profile_environment(folder)
