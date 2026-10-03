"""Experimental exact FP32 products of FP16 operands, indexed by input code.

Two finite binary16 operands have an exactly representable binary32 product.
The table therefore replaces a FP32 FMA by a correctly rounded FP32 addition
without changing CSR order. The table is derived at load time; it is not a
second weight format and is invalidated if weights or activation scales change.
"""
from functools import lru_cache
import torch
from kernel.diagnostic import generated_sources, _replace_once, build_sources
from .sparse_prefetch import sources as host_sources
from .spatial_packed import convolution as invoke


def sources():
    header=generated_sources()['diagnostic_common.cuh']
    header=_replace_once(header,' int next_bits=0,next_cp=0,next_sc=0;',
        ' int next_bits=0,next_cp=0,next_sc=0;\n const float* products=nullptr;')
    start=header.index('           for(int j=p_.row[oc];j<p_.row[oc+1];++j) {',header.index('if(oc<p_.k) {'))
    end=header.index('\n         } else {',start)
    body='''           for(int j=p_.row[oc];j<p_.row[oc+1];++j) {
             Index off;
             if constexpr(Variant>=35&&Variant<=46)off=Index(int32_t(uint32_t(p_.offsets[j])));
             else off=Index(p_.offsets[j]);
             #pragma unroll
             for(int t=0;t<PixelsPerLane;++t) {
               int code=int(p_.x[base[t]+off])+p_.center;
               acc[t]=__fadd_rn(acc[t],__ldg(p_.products+j*(2*p_.center)+code));
             }
           }'''
    header=header[:start]+body+header[end:]
    result=host_sources()
    result['spatial_packed_common.cuh']=header
    for name,source in list(result.items()):
        if name=='bindings.cpp':
            result[name]='\n'.join(line.replace(');',',torch::Tensor products);')
                if line.endswith(');') and not line.lstrip().startswith('m.def(') else line
                for line in source.split('\n'))
        elif name.endswith('.cu'):
            index=source.index(') {')
            source=source[:index]+',torch::Tensor products'+source[index:]
            source=_replace_once(source,' params.q_div=cutlass::FastDivmod(q);',
                ' check(products,at::kFloat);\n'
                ' TORCH_CHECK(products.numel()==value.numel()*(1<<bits),"invalid product table");\n'
                ' params.products=products.data_ptr<float>();\n params.q_div=cutlass::FastDivmod(q);')
            result[name]=source
    return result


@lru_cache(None)
def extension():
    return build_sources('sparse_products',sources())


@torch.no_grad()
def product_table(m):
    def version(tensor):
        try:return tensor._version
        except RuntimeError:return None  # Inference tensors are fixed after loading.
    key=(m.values.data_ptr(),version(m.values),m.activation.data_ptr(),version(m.activation))
    if getattr(m,'_product_key',None)!=key:
        codes=torch.arange(1<<m.spec['bits'],device=m.values.device,dtype=torch.float32)
        scale,z=m.activation.unbind()
        # Match the production kernel's separate add and multiply before half rounding.
        center=1<<(m.spec['bits']-1)
        operand=((codes-center+(center-z))*scale).half().float()
        if not torch.isfinite(operand).all() or not torch.isfinite(m.values).all():
            raise ValueError('exact product lookup requires finite FP16 operands')
        m._product_table=(m.values.float()[:,None]*operand[None,:]).contiguous()
        m._product_key=key
    return m._product_table


def convolution(m,packed,planar,**kwargs):
    table=product_table(m)
    backend=extension()
    class Adapter:
        def __getattr__(self,name):
            fn=getattr(backend,name)
            return lambda *args:fn(*args,table)
    return invoke(m,packed,planar,_backend=Adapter(),_locations=m.compact_locations,**kwargs)
