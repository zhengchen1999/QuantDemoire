import pytest
import torch
from quantdemoire.deploy import pack_signed,unpack_signed


@pytest.mark.parametrize("bits",[4,8])
def test_all_signed_codes(bits):
    q=torch.arange(-(1<<(bits-1)),1<<(bits-1),dtype=torch.int16).to(torch.int8)
    packed=pack_signed(q,bits)
    assert packed.numel()==q.numel()*bits//8
    assert torch.equal(q,unpack_signed(packed,bits))


def test_int4_nibble_order():
    q=torch.tensor([-8,7,-1,0],dtype=torch.int8)
    assert pack_signed(q,4).tolist()==[0x78,0x0f]
