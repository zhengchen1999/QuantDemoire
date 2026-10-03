"""Candidate gathers must retain CSR FP16 operands and accumulation order."""
import pytest
import torch
from helpers import synthetic_layer
from kernel.ops import extension
from quantdemoire.deploy import balance_csr_rows
from quantdemoire.precision import configure

pytestmark = [pytest.mark.experimental,pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')]


@pytest.mark.parametrize('bits', [4, 8])
@pytest.mark.parametrize('candidate', ['spatial_packed','sparse_prefetch','sparse_products','sparse_half2','sparse_subwarp'])
@pytest.mark.parametrize('warps', [0, 8])
@pytest.mark.parametrize('geometry', [(1, 19, 17, 1, 1), (2, 17, 21, 3, 1), (1, 9, 7, 2, 2)])
@pytest.mark.parametrize('mode', ['plain', 'residual', 'next', 'two', 'omitted_two'])
@torch.inference_mode()
def test_spatial_packed(bits, warps, geometry, mode, candidate):
    import importlib
    implementation=importlib.import_module('research.kernel.'+candidate)
    planar_input=candidate!='spatial_packed'
    configure(); torch.manual_seed(91)
    n,h,w,dilation,stride = geometry
    m = synthetic_layer(dict(weight=[32,96,3,3],stride=stride,padding=dilation,dilation=dilation),bits).cuda()
    m.main_half = True; m.joint_warps = warps; m.joint_variant = 9
    m.balance_csr = True; m.balanced_row = balance_csr_rows(m.row); m.packed_csr_meta = True
    m.prepare_impl = 'quad_swizzle'; m.csr_kernel = 'divmod'; m.spec['relu'] = True
    consumer = synthetic_layer(dict(weight=[32,128,3,3],stride=1,padding=1,dilation=1),bits).cuda()
    future = synthetic_layer(dict(weight=[32,160,3,3],stride=1,padding=1,dilation=1),bits).cuda()
    x = torch.randn(n,96,h,w,device='cuda',dtype=torch.float16).contiguous(memory_format=torch.channels_last)
    expected = m(x)
    residual = torch.randn_like(expected) if mode=='residual' else None
    op = extension()

    def run_candidate():
        packed, planar = op.prepare_cached_cat([x],m.quant_multiplier,m.activation,bits,
            m.spec['padded_channels'],m.sparse_channel_map,m.sparse_channels,5,planar_input)
        if not planar_input:assert planar.untyped_storage().nbytes()==1
        return implementation.convolution(m,packed,planar,residual=residual,
            next_quantizer=consumer if mode in ('next','two','omitted_two') else None,
            future_quantizer=future if mode in ('two','omitted_two') else None,
            omit_feature=mode=='omitted_two',_next_planar=planar_input,
            _strategy=1 if planar_input and warps==8 else 0)

    def check(actual):
        expected = implementation.reference(m,x) if candidate=='sparse_half2' else m(x)
        if residual is not None:
            expected = expected + residual
        if mode in ('plain','residual'):
            assert torch.equal(expected,actual), (expected-actual).abs().max()
            return
        output,packed,planar = actual[:3]
        if mode=='omitted_two':
            assert output.untyped_storage().nbytes()==2
        else:
            assert torch.equal(expected,output), (expected-output).abs().max()
        # Prefix words must match independent quantization of the CSR reference.
        hh,ww = expected.shape[-2:]
        a,z = consumer.activation
        codes=(expected.float()*consumer.quant_multiplier[None,:,None,None][:,:32]+z).round().clamp(0,2**bits-1).to(torch.int8)
        codes=codes-(1<<(bits-1))
        from quantdemoire.deploy import unpack_signed
        actual_codes=unpack_signed(packed,bits).reshape(n,hh,ww,consumer.spec['padded_channels'])[...,:32].permute(0,3,1,2)
        assert torch.equal(actual_codes,codes)
        if planar_input:
            for ic,target in enumerate(consumer.sparse_channel_map[:32].cpu().tolist()):
                if target>=0:assert torch.equal(planar[:,target],codes[:,ic])
        if mode in ('two','omitted_two'):
            a,z = future.activation
            codes=(expected.float()*future.quant_multiplier[None,32:64,None,None]+z).round().clamp(0,2**bits-1).to(torch.int8)
            codes=codes-(1<<(bits-1))
            actual_codes=unpack_signed(actual[3],bits).reshape(n,hh,ww,32).permute(0,3,1,2)
            assert torch.equal(actual_codes,codes)

    check(run_candidate())
    run_candidate()
    cached_offsets=m.linear_offsets
    run_candidate()
    assert m.linear_offsets is cached_offsets, 'steady-state inference must reuse the same derived CSR metadata'
    stream=torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run_candidate(); run_candidate()
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output=run_candidate()
    x.normal_().mul_(5); graph.replay(); check(output)
