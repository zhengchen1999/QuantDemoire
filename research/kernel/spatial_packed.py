"""Experimental CSR gather directly from integer NHWC input; not a default.

This removes the spatial branch's separate planar input. It preserves each
row's original FP16 operand and FMA order. Benchmark before adoption: sparse
NHWC gathers may cost more transactions than the removed transpose saves.
"""
from functools import lru_cache
from pathlib import Path

from kernel.diagnostic import _replace_once, generated_sources, build_sources


def sources():
    result = generated_sources()
    header = result.pop('diagnostic_common.cuh')
    header = _replace_once(header, ' int next_bits=0,next_cp=0,next_sc=0;',
        ' int next_bits=0,next_cp=0,next_sc=0;\n const uint8_t* packed_x=nullptr;int packed_cp=0;')
    helper = '''CUTLASS_DEVICE int packed_code(JointParams const& p,int batch,int y,int x,int channel) {
 int64_t index=((int64_t(batch)*p.h+y)*p.w+x)*p.packed_cp+channel;
 if(p.center==8) {
   int code=(p.packed_x[index/2]>>((index&1)*4))&15;
   return (code^8)-8;
 }
 return int(reinterpret_cast<const int8_t*>(p.packed_x)[index]);
}
'''
    anchor = 'template<int Access,int Warps,bool Sparse,int Variant> CUTLASS_DEVICE cutlass::MatrixCoord joint_coord(int ordinal);'
    header = _replace_once(header, anchor, helper + anchor)
    header = _replace_once(header, 'int code=int(p_.x[base[t]+off]);float operand;',
        'int loc=p_.location[j];\n'
        '               int code=packed_code(p_,batch[t],iy0[t]+((loc>>16)&15)*p_.dil,ix0[t]+((loc>>12)&15)*p_.dil,loc&4095);float operand;')
    header = _replace_once(header,
        'float av=(float(p_.x[((Index(batch[t])*p_.c+ic)*p_.h+iy)*p_.w+ix])+delta)*scale;',
        'float av=(float(packed_code(p_,batch[t],iy,ix,ic))+delta)*scale;')
    header = _replace_once(header,
        ' p.shared_offset=sizeof(typename Kernel::Epilogue::SharedStorage);',
        ' p.packed_x=x.data_ptr<uint8_t>();p.packed_cp=cp;\n p.shared_offset=sizeof(typename Kernel::Epilogue::SharedStorage);')
    output = {'spatial_packed_common.cuh':header}
    declarations, bindings = [], []
    source = (Path(__file__).resolve().parents[2]/'kernel/csrc/joint.cu').read_text()
    start = source.index('torch::Tensor joint_conv(')
    end = source.index('\n}', start)+2
    source = source[start:end]
    result['dense_conv.cu'] = '#include "diagnostic_common.cuh"\n' + source.replace('joint_conv(', 'dense_conv(', 1)
    for filename, source in result.items():
        if not filename.endswith('.cu'):
            continue
        name = filename.removesuffix('.cu').replace('dense_', 'packed_')
        source = source.replace(filename.removesuffix('.cu')+'(', name+'(', 1)
        source = source.replace('diagnostic_common.cuh', 'spatial_packed_common.cuh')
        declaration_start = source.index('torch::Tensor packed_conv(') if name == 'packed_conv' else source.index('std::vector<torch::Tensor> ')
        declaration_end = source.index(' {', declaration_start)
        declarations.append(source[declaration_start:declaration_end]+';')
        bindings.append(f'  m.def("{name}", &{name});')
        source = source.replace('(!sparse||shared_input||planar.is_contiguous())', 'true')
        source = source.replace('TORCH_CHECK(!sparse&&!next_planar&&!add_residual,"dense diagnostic must disable sparse buffers");',
            'TORCH_CHECK(sparse&&!next_planar&&!add_residual,"packed prototype must disable planar output");')
        check = ('\n TORCH_CHECK(sparse&&r>1&&(warps==4||(warps==8&&balance&&packed_metadata)),'
                 '"packed prototype requires spatial CSR, balanced metadata for eight warps");')
        source = source[:declaration_end+2] + check + source[declaration_end+2:]
        start = source.index(' #define RUN(')
        end = source.index(' C10_CUDA_KERNEL_LAUNCH_CHECK()', start)
        variant4, variant8 = (8,36) if name=='packed_conv' else (11,38)
        dispatch = f''' #define RUN(B,A,W,V) run_joint_selected<B,A,W,true,V>(x,weight,scale,out,cp,params,stream)
 if(warps==4) {{if(bits==4){{RUN(4,8,4,{variant4});}}else{{RUN(8,8,4,{variant4});}}}}
 else {{if(bits==4){{RUN(4,4,8,{variant8});}}else{{RUN(8,4,8,{variant8});}}}}
 #undef RUN
'''
        source = source[:start] + dispatch + source[end:]
        output[name+'.cu'] = source
    output['bindings.cpp'] = ('#include <torch/extension.h>\n'+'\n'.join(declarations)
        +'\nPYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {\n'+'\n'.join(bindings)+'\n}\n')
    return output


@lru_cache(None)
def extension():
    return build_sources('spatial_packed', sources())


def convolution(m, packed, planar, *, next_quantizer=None, residual=None,
                future_quantizer=None, omit_feature=False, _backend=None, _locations=None,
                _next_planar=False, _strategy=0):
    """Experimental prepared-input operator; original CSR locations are required."""
    from quantdemoire.branch_diagnostic import _joint_args
    if m.spec['shape'][2] <= 1 or m.spec['shape'][0] % 8:
        raise ValueError('prototype supports spatial convolutions with K divisible by 8')
    if m.joint_warps not in (0, 8):
        raise ValueError('prototype supports the deployed separate/eight-warp plan')
    if next_quantizer is None and (omit_feature or future_quantizer is not None):
        raise ValueError('omission and future cache require a next quantizer')
    if m.joint_warps == 8 and (not m.balance_csr or not m.packed_csr_meta):
        raise ValueError('eight-warp prototype requires balanced packed CSR metadata')
    locations = m.locations if _locations is None else _locations
    metadata = m.joint_warps == 8
    args, balanced = _joint_args(m, packed, planar, locations, residual, packed_metadata=metadata)
    args = list(args)
    args[20] = True  # sparse
    op = extension() if _backend is None else _backend
    if next_quantizer is None:
        return op.packed_conv(*args, balanced, False, False, metadata)
    name = 'packed_next' + ('_two' if future_quantizer is not None else '')
    name += '_omitted' if omit_feature else ''
    nxt = next_quantizer
    extra = () if future_quantizer is None else (
        future_quantizer.quant_multiplier, future_quantizer.activation, future_quantizer.spec['bits'], 32)
    result = getattr(op, name)(*args, nxt.quant_multiplier, nxt.activation, nxt.sparse_channel_map,
        nxt.spec['bits'], nxt.spec['padded_channels'], nxt.sparse_channels,
        _next_planar, _strategy, balanced, False, False, metadata, *extra)
    if omit_feature:
        result[0]._quantdemoire_omitted = True
    return result
