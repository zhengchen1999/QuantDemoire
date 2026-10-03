"""Experimental two-entry CSR prefetch, preserving FP32 FMA order."""
from functools import lru_cache
from kernel.diagnostic import generated_sources, _replace_once, build_sources
from .spatial_packed import sources as host_sources, convolution as invoke


def sources():
    header=generated_sources()['diagnostic_common.cuh']
    start=header.index('           for(int j=p_.row[oc];j<p_.row[oc+1];++j) {',header.index('if(oc<p_.k) {'))
    end=header.index('\n         } else {',start)
    body='''           for(int j=p_.row[oc];j<p_.row[oc+1];j+=2) {
             float weights[2],operands[2][PixelsPerLane];
             #pragma unroll
             for(int u=0;u<2;++u)if(j+u<p_.row[oc+1]) {
               Index off;
               if constexpr(Variant>=35&&Variant<=46) {
                 uint64_t entry=static_cast<uint64_t>(p_.offsets[j+u]);
                 off=Index(int32_t(uint32_t(entry)));
                 weights[u]=__half2float(__ushort_as_half(uint16_t(entry>>32)));
               } else {off=Index(p_.offsets[j+u]);weights[u]=__half2float(p_.value[j+u]);}
               #pragma unroll
               for(int t=0;t<PixelsPerLane;++t) {
                 int code=int(p_.x[base[t]+off]);
                 if constexpr(SharedW4Lookup)operands[u][t]=operand_table[code+8];
                 else operands[u][t]=__half2float(__float2half_rn((float(code)+delta)*scale));
               }
             }
             #pragma unroll
             for(int u=0;u<2;++u)if(j+u<p_.row[oc+1]) {
               #pragma unroll
               for(int t=0;t<PixelsPerLane;++t)acc[t]=fmaf(operands[u][t],weights[u],acc[t]);
             }
           }'''
    header=header[:start]+body+header[end:]
    result=host_sources()
    result['spatial_packed_common.cuh']=header
    for name,source in list(result.items()):
        if not name.endswith('.cu'):continue
        source=source.replace('sparse&&!next_planar&&!add_residual', 'sparse&&!add_residual')
        source=source.replace('&&true,"invalid joint planar input"', '&&planar.is_contiguous(),"invalid joint planar input"')
        # Eight-warp output uses the same transpose strategy when sparse
        # consumer buffers remain enabled. Four-warp direct stores need none.
        if name!='packed_conv.cu':
            source=_replace_once(source,
                ' else {if(bits==4){RUN(4,4,8,38);}else{RUN(8,4,8,38);}}',
                ' else if(strategy==1) {if(bits==4){RUN(4,4,8,40);}else{RUN(8,4,8,40);}}\n'
                ' else {if(bits==4){RUN(4,4,8,38);}else{RUN(8,4,8,38);}}')
        result[name]=source
    return result


@lru_cache(None)
def extension():
    return build_sources('sparse_prefetch',sources())


def convolution(m,packed,planar,**kwargs):
    return invoke(m,packed,planar,_backend=extension(),_locations=m.compact_locations,**kwargs)
