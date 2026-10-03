import os
os.environ['TORCH_CUDNN_V8_API_DISABLED']='1'
import pytest
import torch
from helpers import synthetic_layer
from quantdemoire.precision import configure
from quantdemoire.runtime import CudaGraphRunner


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('bits',[4,8])
@torch.inference_mode()
def test_graph_accepts_changing_inputs(bits):
    configure();torch.manual_seed(301)
    model=synthetic_layer(dict(weight=[32,48,3,3],stride=1,padding=1,dilation=1),bits).cuda()
    x=torch.randn(1,48,17,19,device='cuda',dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    y=torch.randn_like(x)
    graph=CudaGraphRunner(model,x)
    first=graph(x).clone();second=graph(y).clone()
    torch.testing.assert_close(first,model(x),rtol=0,atol=0)
    torch.testing.assert_close(second,model(y),rtol=0,atol=0)
    assert not torch.equal(first,second)
    with pytest.raises(ValueError,match='shape'):
        graph(x[:,:,:-1,:])
