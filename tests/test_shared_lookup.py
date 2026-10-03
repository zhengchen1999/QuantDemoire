"""W4 lookup keeps exact half operands and the original CSR accumulation order."""
import os
os.environ['TORCH_CUDNN_V8_API_DISABLED']='1'
import pytest
import torch
from helpers import synthetic_layer
from quantdemoire.precision import configure
from test_feature_omission import stream_graph

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('shape',[(36,48,3,2,3,17,19),(32,80,3,1,2,17,19),
                                  (32,144,3,1,1,1,1),(36,576,1,1,1,9,11)])
@pytest.mark.parametrize('scale',[2**-23,.1,100.])
@pytest.mark.parametrize('zero_point',[0.,6.,11.,15.])
@torch.inference_mode()
def test_shared_lookup_matches_separate_csr(shape,scale,zero_point):
    configure();torch.manual_seed(7103)
    k,c,r,stride,dil,h,w=shape
    m=synthetic_layer(dict(weight=[k,c,r,r],stride=stride,padding=dil if r==3 else 0,dilation=dil),4,zero_point=zero_point).cuda()
    m.main_half=True;m.csr_kernel='divmod';m.prepare_impl='quad_swizzle'
    m.balance_csr=True;m.packed_csr_meta=True;m.few_channels=True;m.joint_variant=9
    ratio=scale/float(m.activation[0]);m.activation[0]=scale
    m.output_scale.mul_(ratio);m.offset_sums.mul_(ratio);m.refresh_runtime()
    x=torch.randn(2,c,h,w,device='cuda',dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    def call():
        results=[]
        for relu in (False,True):
            m.spec['relu']=relu;m.joint_warps=0;expected=m(x)
            m.joint_warps=8;actual=m(x)
            results.extend((expected,actual))
        return results
    def validate(result):
        assert all(torch.equal(result[i],result[i+1]) for i in (0,2))
    stream_graph(call,validate,x)
