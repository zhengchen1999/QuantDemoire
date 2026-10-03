"""Row scheduling retains CSR row order, sparse operands and all fused outputs."""
import pytest
import torch
from quantdemoire.deploy import balance_csr_rows
from test_joint import test_joint_output as _joint, test_shared_pointwise_without_planar as _shared
from test_joint_next import test_joint_next as _next


@pytest.mark.parametrize('lengths',[[0]*36,[50,1,0,7,2,0,0,19]*8+[2,0,1,3]])
def test_balanced_row_permutation(lengths):
    row=torch.tensor([0]+lengths,dtype=torch.int32).cumsum(0).int()
    extended=balance_csr_rows(row);k=len(lengths)
    assert torch.equal(extended[:k+1],row)
    assert extended.dtype==row.dtype and extended.numel()==k+1+((k+31)//32)*32
    assert torch.equal(extended,balance_csr_rows(row))
    for permutation in extended[k+1:].split(32):
        assert torch.equal(permutation.sort().values,torch.arange(32,dtype=row.dtype))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('bits',[4,8])
@pytest.mark.parametrize('variant',[8,9])
@pytest.mark.parametrize('shape',[(36,48,3,2,3,17,19),(32,80,3,1,1,17,19),(64,144,1,1,1,9,11),(12,64,3,1,1,1,1),(96,80,3,1,2,13,21),(96,80,1,1,1,9,11)])
@pytest.mark.parametrize('retained',[False,True])
def test_balanced_joint(monkeypatch,bits,variant,shape,retained):
    monkeypatch.setenv('QUANTDEMOIRE_BALANCE_CSR','1')
    monkeypatch.setenv('QUANTDEMOIRE_SHARED_RETAIN_STAGES',str(int(retained)))
    _joint(bits,8,variant,shape)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('bits,c',[(4,80),(8,80),(4,240),(8,144)])
@pytest.mark.parametrize('retained',[False,True])
def test_balanced_shared(monkeypatch,bits,c,retained):
    monkeypatch.setenv('QUANTDEMOIRE_BALANCE_CSR','1')
    monkeypatch.setenv('QUANTDEMOIRE_SHARED_RETAIN_STAGES',str(int(retained)))
    _shared(bits,c)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('bits,next_bits',[(4,4),(4,8),(8,4),(8,8)])
@pytest.mark.parametrize('r',[1,3])
@pytest.mark.parametrize('shared',[False,True])
@pytest.mark.parametrize('geometry',[(2,17,19,2),(1,1,1,1)])
@pytest.mark.parametrize('strategy',[0,1])
@pytest.mark.parametrize('producer_prefixed',[False,True])
@pytest.mark.parametrize('retained',[False,True])
def test_balanced_next(monkeypatch,bits,next_bits,r,shared,geometry,strategy,producer_prefixed,retained):
    monkeypatch.setenv('QUANTDEMOIRE_BALANCE_CSR','1')
    monkeypatch.setenv('QUANTDEMOIRE_SHARED_RETAIN_STAGES',str(int(retained)))
    _next(bits,next_bits,r,shared,geometry,strategy,producer_prefixed)
