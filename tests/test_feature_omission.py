"""Metadata-only dense features require complete, consumer-owned quantized caches."""
import os
os.environ['TORCH_CUDNN_V8_API_DISABLED']='1'
import pytest
import torch
from helpers import synthetic_layer
from kernel.ops import extension
from quantdemoire.deploy import ChannelConcat
from quantdemoire.esdnet import DenseBlock
from quantdemoire.precision import configure

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def layer(bits,c,r=3,dil=1,k=32):
    m=synthetic_layer(dict(weight=[k,c,r,r],stride=1,padding=dil if r>1 else 0,dilation=dil),bits).cuda()
    m.main_half=True;m.prepare_impl='quad_swizzle';m.joint_warps=8;m.joint_variant=9
    m.joint_next_mode=2;m.balance_csr=True;m.packed_csr_meta=True;m.fuse_second_quant=True
    m.csr_kernel='divmod';m.refresh_runtime()
    return m


def placeholder(tensor,valid=True):
    value=torch.empty(1 if valid else 2,device=tensor.device,dtype=tensor.dtype).as_strided(tensor.shape,(0,0,0,0))
    value._quantdemoire_omitted=True
    return value


def stream_graph(call,validate,x):
    validate(call())
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):validate(call());call()
    torch.cuda.current_stream().wait_stream(stream)
    if os.environ.get('PYTORCH_NO_CUDA_MEMORY_CACHING')!='1':
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):captured=call()
        for scale in (.00001,100.):x.normal_().mul_(scale);graph.replay();validate(captured)


@pytest.mark.parametrize('bits,next_bits,future_bits',[(b,n,f) for b in (4,8) for n in (4,8) for f in (4,8)])
@pytest.mark.parametrize('c',[80,144])
@pytest.mark.parametrize('mode',[1,2])
@pytest.mark.parametrize('next_planar',[False,True])
@pytest.mark.parametrize('relu',[False,True])
@torch.inference_mode()
def test_joint_prefix_only(bits,next_bits,future_bits,c,mode,next_planar,relu):
    configure();torch.manual_seed(1294);op=extension()
    n,h,w,dil=(2,17,19,2) if c==80 else (1,1,1,1)
    m=layer(bits,c,dil=dil);nxt=layer(next_bits,c+32,r=3 if next_planar else 1);future=layer(future_bits,c+64)
    m.joint_next_mode=mode;m.spec['relu']=relu
    nxt.shared_skip_planar=not next_planar;nxt.shared_skip_prefix_planar=not next_planar;nxt.retain_shared=not next_planar
    x=torch.randn(n,c,h,w,device='cuda',dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    def call():
        expected=m(x,next_quantizer=nxt,future_quantizer=future)
        single=m(x,next_quantizer=nxt,omit_feature=True)
        double=m(x,next_quantizer=nxt,future_quantizer=future,omit_feature=True)
        def finish(result):
            return op.prepare_next_cat([expected[0],x],nxt.quant_multiplier,nxt.activation,next_bits,
                nxt.spec['padded_channels'],nxt.sparse_channel_map,nxt.sparse_channels,result[1],result[2],32,5,next_planar)
        return expected,single,double,finish(expected),finish(single),finish(double)
    def validate(result):
        expected,single,double,full,one,two=result
        for output in (single[0],double[0]):
            assert output._quantdemoire_omitted and output.shape==expected[0].shape
            assert output.stride()==(0,0,0,0) and output.untyped_storage().nbytes()==2
        for actual in (one,two):
            assert torch.equal(full[0],actual[0])
            if next_planar:assert torch.equal(full[1],actual[1])
            else:assert actual[1].untyped_storage().nbytes()==1
        assert torch.equal(expected[3],double[3])
    stream_graph(call,validate,x)


@pytest.mark.parametrize('bits,next_bits',[(b,n) for b in (4,8) for n in (4,8)])
@pytest.mark.parametrize('geometry',[(2,17,19,2),(1,1,1,1),(1,8,16,1)])
@pytest.mark.parametrize('next_planar',[False,True])
@pytest.mark.parametrize('relu',[False,True])
@pytest.mark.parametrize('tiny',[False,True])
@torch.inference_mode()
def test_separate_prefix_only(bits,next_bits,geometry,next_planar,relu,tiny):
    configure();torch.manual_seed(2364);op=extension();n,h,w,dil=geometry;c=144
    m=layer(bits,c,dil=dil);m.joint_warps=0;m.spec['relu']=relu
    if tiny:
        ratio=2**-23/float(m.activation[0]);m.activation[0]=2**-23
        m.output_scale.mul_(ratio);m.offset_sums.mul_(ratio);m.refresh_runtime()
    nxt=layer(next_bits,c+32,r=3 if next_planar else 1)
    nxt.shared_skip_planar=not next_planar;nxt.shared_skip_prefix_planar=not next_planar;nxt.retain_shared=not next_planar
    x=torch.randn(n,c,h,w,device='cuda',dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    def call():
        expected=m(x,next_quantizer=nxt);actual=m(x,next_quantizer=nxt,omit_feature=True)
        def finish(result):
            return op.prepare_next_cat([expected[0],x],nxt.quant_multiplier,nxt.activation,next_bits,
                nxt.spec['padded_channels'],nxt.sparse_channel_map,nxt.sparse_channels,result[1],result[2],32,5,next_planar)
        return expected,actual,finish(expected),finish(actual)
    def validate(result):
        expected,actual,full,got=result
        assert actual[0]._quantdemoire_omitted and actual[0].shape==expected[0].shape
        assert actual[0].stride()==(0,0,0,0) and actual[0].untyped_storage().nbytes()==2
        assert torch.equal(full[0],got[0])
        if next_planar:assert torch.equal(full[1],got[1])
        else:assert got[1].untyped_storage().nbytes()==1
    stream_graph(call,validate,x)


@pytest.mark.parametrize('bits',[4,8])
@pytest.mark.parametrize('geometry',[(2,17,19),(1,1,1)])
@pytest.mark.parametrize('last_separate',[False,True])
@pytest.mark.parametrize('post_shared',[False,True])
@torch.inference_mode()
def test_omitted_dense_pipeline(bits,geometry,last_separate,post_shared):
    configure();torch.manual_seed(733)
    block=DenseBlock(48,(1,2,1),32).cuda().half().eval();block.lazy_concat=True;block.fuse_next_quant=True
    sequence=[]
    for i,wrapper in enumerate(block.conv_layers):
        m=layer(bits,48+32*i,dil=(1,2,1)[i]);m.spec['relu']=True
        wrapper.conv[0]=m;wrapper.conv[1]=torch.nn.Identity();sequence.append(m)
    if last_separate:sequence[-1].joint_warps=0
    post=layer(bits,144,r=1,k=48);post.shared_skip_planar=post_shared;post.shared_skip_prefix_planar=post_shared
    post.retain_shared=True
    if not post_shared:post.joint_variant=8
    block.conv_post.conv=post;sequence.append(post)
    n,h,w=geometry;x=torch.randn(n,48,h,w,device='cuda',dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    residual=torch.randn_like(x);seen=[]
    def observe(m,args):
        if isinstance(args[0],ChannelConcat) and args[0].has_omitted_features:seen.append(args[0])
    hooks=[m.register_forward_pre_hook(observe) for m in sequence]
    def call():
        for m in sequence:m.omit_cached_features=0
        expected=block(x,residual=residual)
        for m in sequence:m.omit_cached_features=2
        actual=block(x,residual=residual)
        return expected,actual
    assert torch.equal(*call());assert len(seen)==2
    assert all(value.prepared is None and value.second_cached is None for value in seen)
    # Reusing a consumed concat must fail before any metadata feature is read.
    with pytest.raises(RuntimeError,match='no consumer-specific cache coverage'):post(seen[-1])
    for handle in hooks:handle.remove()
    seen.clear()
    stream_graph(call,lambda result:torch.testing.assert_close(*result,rtol=0,atol=0),x)


@pytest.mark.parametrize('bits',[4,8])
@torch.inference_mode()
def test_missing_coverage_and_invalid_metadata(bits):
    configure();m=layer(bits,96);real=[torch.zeros(1,32,3,5,device='cuda',dtype=torch.float16).contiguous(memory_format=torch.channels_last) for _ in range(3)]
    def omitted(valid=True):
        value=placeholder(real[0],valid)
        def fail(*args,**kwargs):raise AssertionError('placeholder materialization attempted')
        value.contiguous=fail
        return value
    dummy=torch.empty(0,device='cuda');prepared=(m,dummy,dummy,32)
    cases=[(omitted(),'no consumer-specific cache coverage'),
        (ChannelConcat([omitted(),*real[1:]]),'no consumer-specific cache coverage'),
        (ChannelConcat([real[0],omitted(),real[2]],prepared=prepared),'no consumer-specific cache coverage'),
        (ChannelConcat([*real[:2],omitted()],prepared=prepared,second_cached=dummy),'no consumer-specific cache coverage'),
        (ChannelConcat([real[0],omitted(),real[2]],second_cached=dummy),'no consumer-specific cache coverage'),
        (ChannelConcat([omitted(False),*real[1:]],prepared=prepared),'invalid omitted-feature metadata tensor'),
        (ChannelConcat([omitted(),*real[1:]],prepared=(object(),dummy,dummy,32)),'different consumer or execution mode')]
    for value,message in cases:
        with pytest.raises((RuntimeError,ValueError),match=message):m(value)
    with pytest.raises(ValueError,match='unsupported omitted-feature producer'):m(real[0],omit_feature=True)
    for backend in ('fake','fake_fp16'):
        m.backend=backend
        with pytest.raises(RuntimeError,match='omitted features require a CUDA consumer'):m(cases[1][0])
    nested=ChannelConcat([real[0],cases[1][0]])
    assert nested.has_omitted_features and nested.prepared is None and nested.second_cached is None
