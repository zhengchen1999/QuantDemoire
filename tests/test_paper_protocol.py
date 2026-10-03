import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from quantdemoire.quantization.observers import rank_tail_mask, resolve_sample_size
from quantdemoire.quantization.frequency_loss import gaussian_lowpass, VGGFeatureLoss
from quantdemoire.quantization.pipeline import resolve_quant_config, resolve_frequency_config, collect_trainable_quant_params
from quantdemoire.quantization.quant_conv import QuantizedConv2d


def test_small_activation_sampling_guard():
    from quantdemoire.quantization.observers import estimate_range
    values=torch.linspace(-3,2,64).reshape(1,64,1,1)
    torch.manual_seed(19)
    lo,hi=estimate_range(values,'sample_minmax',sample_rate=.001)
    assert hi-lo < 2e-6  # Preserve the historical behavior unless requested.
    lo,hi=estimate_range(values,'sample_minmax',sample_rate=.001,small_tensor_policy='full_if_single')
    assert lo==-3 and hi==2
    large=torch.linspace(-5,5,10000)
    torch.manual_seed(13)
    expected=estimate_range(large,'sample_minmax',sample_rate=.001)
    torch.manual_seed(13)
    actual=estimate_range(large,'sample_minmax',sample_rate=.001,small_tensor_policy='full_if_single')
    assert all(torch.equal(x,y) for x,y in zip(expected,actual))
    q=QuantizedConv2d(64,8,1,quant_config=dict(act_observer='sample_minmax',act_sample_rate=.001,
                                           act_small_tensor_policy='full_if_single'))
    q._collect_pass2_stats(values);q.finalize_activation_params()
    assert q.lb_a==-3 and q.ub_a==2


@pytest.mark.parametrize('rate',[.001,.01,.05])
def test_pooled_sampling_is_stable_when_gamma_increases(rate):
    from quantdemoire.quantization.observers import estimate_range
    values=torch.linspace(-2,5,256).reshape(1,256,1,1)
    lo,hi=estimate_range(values,'sample_minmax',sample_rate=rate,small_tensor_policy='full_if_pooled')
    assert lo==-2 and hi==5
    spatial=values.expand(1,256,16,16)
    torch.manual_seed(31)
    expected=estimate_range(spatial,'sample_minmax',sample_rate=rate)
    torch.manual_seed(31)
    actual=estimate_range(spatial,'sample_minmax',sample_rate=rate,small_tensor_policy='full_if_pooled')
    assert all(torch.equal(x,y) for x,y in zip(actual,expected))


@pytest.mark.parametrize('aggregation', ['mean','extrema'])
def test_cross_image_sample_aggregation(aggregation):
    q=QuantizedConv2d(2,2,1,quant_config=dict(act_observer='minmax',calibration_aggregation=aggregation,
        use_smooth_scale=True,smooth_alpha=1.))
    x=torch.tensor([[[[1.,2.]],[[3.,4.]]]])
    y=torch.tensor([[[[-5.,1.]],[[1.,2.]]]])
    q._collect_pass1_stats(x);q._collect_pass1_stats(y)
    assert torch.equal(q.compute_smooth_scale(),torch.tensor([3.5,3.]) if aggregation=='mean' else torch.tensor([5.,4.]))
    q.smooth_scale.data.fill_(1)
    q._collect_pass2_stats(x);q._collect_pass2_stats(y);q.finalize_activation_params()
    expected=(-2.,3.) if aggregation=='mean' else (-5.,4.)
    assert q.lb_a==expected[0] and q.ub_a==expected[1]
    q.reset_calibration_buffers()
    q._collect_pass2_stats(x);q.finalize_activation_params()
    assert q.lb_a==1 and q.ub_a==4  # Reset discards previous extrema.


@pytest.mark.parametrize('bits', [4,8])
@pytest.mark.parametrize('recipe,tail', [('paper',.5),('deploy',.25)])
def test_paper_config(bits,recipe,tail):
    payload=yaml.safe_load(Path(f'config/ptq/{recipe}_w{bits}a{bits}.yaml').read_text())
    flat={k:v for group in payload.values() for k,v in group.items()}
    q=resolve_quant_config(flat);f=resolve_frequency_config(flat)
    assert q['nbits_w']==q['nbits_a']==bits
    assert q['act_observer']=='sample_minmax' and q['weight_observer']=='minmax'
    assert q['input_channel_sample_rate']==q['act_sample_rate']==.001
    assert q['weight_outlier_mode']=='rank_fp16' and q['weight_outlier_percentile']==tail
    assert not q['learn_weight_bounds'] and not f['train_smooth']
    assert f['gaussian_levels']==3 and f['gaussian_first_dilation']==2
    assert f['scheduler']=='cosine_restarts' and f['epochs']==4 and f['lr']==.001
    assert f['perceptual_feature_indices']==[15] and f['perceptual_preprocess']=='esdnet'
    assert flat['CALI_USE_TRAIN_CROPS'] and flat['CROP_SIZE']==512
    assert resolve_sample_size(100_000,sample_size=10_000,sample_rate=q['act_sample_rate'])==100


@pytest.mark.parametrize('fraction', [.0025,.005])
def test_rank_ties_and_fp16_weights(fraction):
    torch.manual_seed(13)
    q=QuantizedConv2d(16,8,3,quant_config=dict(enable_weight_outlier_fp16=True,
        weight_outlier_mode='rank_fp16',weight_outlier_percentile=fraction*100,
        weight_observer='minmax',learn_weight_bounds=False))
    weight=torch.linspace(-1,1,q.weight.numel()).reshape_as(q.weight)
    weight.flatten()[:16]=-.1234567;weight.flatten()[-16:]=.1234567
    dequant,scale,z,mask,outliers=q._quantize_weight_tensor(weight)
    order=weight.flatten().argsort(stable=True);count=math.ceil(weight.numel()*fraction)
    assert mask.sum()==2*count
    assert mask.flatten()[order[:count]].all() and mask.flatten()[order[-count:]].all()
    assert torch.equal(dequant[mask],weight[mask].half().float())
    tied=torch.zeros_like(weight)
    selected=rank_tail_mask(tied,fraction).flatten()
    assert selected[:count].all() and selected[-count:].all() and selected.sum()==2*count


def test_paper_frequency_impulse():
    image=torch.zeros(1,1,65,65);image[0,0,32,32]=1
    actual=gaussian_lowpass(image,levels=3,first_dilation=2)[0,0]
    # Analytic separable response: three binomial filters at offsets 2,4,8.
    coefficients={0:1.}
    for dilation in (2,4,8):
        updated={}
        for position,value in coefficients.items():
            for offset,factor in ((-dilation,.25),(0,.5),(dilation,.25)):
                updated[position+offset]=updated.get(position+offset,0)+value*factor
        coefficients=updated
    vector=torch.zeros(65)
    for position,value in coefficients.items():vector[32+position]=value
    assert torch.equal(actual,vector[:,None]*vector[None,:])
    assert actual.sum()==1 and actual[1::2].sum()==0
    assert not torch.equal(actual,gaussian_lowpass(image,3)[0,0])


def test_only_activation_bounds_train():
    q=QuantizedConv2d(4,8,3,quant_config=dict(learn_weight_bounds=False,use_smooth_scale=True))
    collect_trainable_quant_params(q,train_smooth=False)
    assert {name for name,value in q.named_parameters() if value.requires_grad}=={'lb_a','ub_a'}
    q.skip_quantization=True
    collect_trainable_quant_params(q,train_smooth=False)
    assert not any(value.requires_grad for value in q.parameters())


def test_esdnet_perceptual_preprocess(monkeypatch):
    import torchvision.models
    monkeypatch.setattr(torchvision.models,'vgg16',lambda **kwargs:SimpleNamespace(
        features=torch.nn.Sequential(*[torch.nn.Identity() for _ in range(16)])))
    metric=VGGFeatureLoss([15],preprocess='esdnet')
    pred=torch.full((1,3,8,8),2.,requires_grad=True);target=torch.zeros_like(pred)
    actual=metric(pred,target)
    torch.testing.assert_close(actual,(2/metric.std).mean())
    actual.backward()
    assert pred.grad.abs().sum()>0  # No out-of-range clamp in the cited loss.
