"""Experimental packed INT4 planar staging for spatial CSR (even input width).

The scalar CSR arithmetic is unchanged. A zero-stride metadata view describes
logical NCHW dimensions; its owned backing storage contains two signed spatial
codes per byte. Only this isolated backend may consume that view.
"""
from functools import lru_cache
from pathlib import Path
from kernel.diagnostic import generated_sources, _replace_once, build_sources
from .sparse_prefetch import sources as host_sources
from .spatial_packed import convolution as invoke


def sources():
    header=generated_sources()['diagnostic_common.cuh']
    anchor='template<int Access,int Warps,bool Sparse,int Variant> CUTLASS_DEVICE cutlass::MatrixCoord joint_coord(int ordinal);'
    helper='''CUTLASS_DEVICE int planar_code(const int8_t* data,int64_t index) {
 int code=(uint8_t(data[index/2])>>((index&1)*4))&15;
 return (code^8)-8;
}
'''
    header=_replace_once(header,anchor,helper+anchor)
    header=_replace_once(header,'int code=int(p_.x[base[t]+off]);float operand;',
                         'int code=planar_code(p_.x,base[t]+off);float operand;')
    old='p_.x[((Index(batch[t])*p_.c+ic)*p_.h+iy)*p_.w+ix]'
    header=header.replace(old,'planar_code(p_.x,((Index(batch[t])*p_.c+ic)*p_.h+iy)*p_.w+ix)')
    source=host_sources()['packed_conv.cu']
    source=source.replace('&&planar.is_contiguous()',
                         '&&planar.size(3)%2==0&&planar.storage().nbytes()==planar.numel()/2')
    source=source.replace('TORCH_CHECK(bits==4||bits==8,"invalid joint bit width");',
                         'TORCH_CHECK(bits==4,"planar INT4 prototype requires four bits");')
    source=source.replace('if(bits==4){RUN(4,8,4,8);}else{RUN(8,8,4,8);}','RUN(4,8,4,8);')
    source=source.replace('if(bits==4){RUN(4,4,8,36);}else{RUN(8,4,8,36);}','RUN(4,4,8,36);')
    declaration=source[source.index('torch::Tensor packed_conv('):source.index(' {')]+';'
    original=(Path(__file__).resolve().parents[2]/'kernel/csrc/epilogue.cu').read_text()
    prefix=original[:original.index('template <int Bits, bool Sparse')]
    start=original.index('template<int Bits,bool Sparse,int Pixels,bool Partial=false,bool Unroll=true,int Channels=128>')
    quad=original[start:original.index('\nstd::vector<torch::Tensor> prepare_cat_impl',start)]
    store='planar[(int64_t(batch)*sc+target)*h*w+sp]=int8_t(tile[pp][(cc/4) ^ (Channels==64?((pp>>1)&15):0)]>>((cc&3)*8));'
    quad=_replace_once(quad,store,'''if((pp&1)==0) {
           int v0=(tile[pp][(cc/4)^((pp>>1)&15)]>>((cc&3)*8))&15;
           int v1=(tile[pp+1][(cc/4)^(((pp+1)>>1)&15)]>>((cc&3)*8))&15;
           planar[((int64_t(batch)*sc+target)*h*w+sp)/2]=int8_t(v0|(v1<<4));
         }''')
    host='''std::vector<torch::Tensor> prepare4(std::vector<torch::Tensor> inputs,torch::Tensor multiplier,
 torch::Tensor params,int64_t cp,torch::Tensor selection,int64_t sc) {
 TORCH_CHECK(!inputs.empty()&&inputs.size()<=8,"invalid input segments");
 auto x=inputs[0];
 TORCH_CHECK(x.is_cuda()&&x.dim()==4&&x.scalar_type()==at::kHalf,"invalid input");
 int n=x.size(0),h=x.size(2),w=x.size(3),c=0;
 TORCH_CHECK(w%2==0&&n>0&&h>0&&w>0,"planar INT4 requires even width");
 Segments segments{};segments.count=inputs.size();
 for(int i=0;i<inputs.size();++i) {
   auto part=inputs[i];
   TORCH_CHECK(part.device()==x.device()&&part.scalar_type()==at::kHalf&&part.dim()==4&&
     part.is_contiguous(at::MemoryFormat::ChannelsLast)&&part.size(0)==n&&part.size(2)==h&&part.size(3)==w&&
     part.size(1)%4==0,"invalid aligned input segment");
   segments.pointer[i]=reinterpret_cast<const half*>(part.data_ptr());
   segments.channels[i]=part.size(1);c+=part.size(1);segments.ends[i]=c;
 }
 check_tensor(multiplier,x,at::kFloat);check_tensor(params,x,at::kFloat);check_tensor(selection,x,at::kInt);
 TORCH_CHECK(cp>=c&&cp%32==0&&multiplier.numel()==c&&params.numel()==2&&selection.numel()==c&&sc>0&&sc<=c,"invalid quantizer");
 TORCH_CHECK(int64_t(n)*h*w<INT_MAX-128,"too many spatial positions");
 c10::cuda::CUDAGuard guard(x.device());
 auto packed=torch::empty({int64_t(n)*h*w*cp/2},x.options().dtype(at::kByte));
 auto planar=torch::empty({int64_t(n)*sc*h*w/2},x.options().dtype(at::kChar)).as_strided({n,sc,h,w},{0,0,0,0});
 int pixels=n*h*w>128?128:32,tiles=(n*h*w+pixels-1)/pixels;
 dim3 blocks((cp+63)/64,std::min(tiles,65535),(tiles+65534)/65535);
 auto stream=c10::cuda::getCurrentCUDAStream(x.get_device());
 #define RUN(P) prepare_quad_kernel<4,true,P,false,false,64><<<blocks,dim3(16,16),0,stream>>>(segments,multiplier.data_ptr<float>(),params.data_ptr<float>(),packed.data_ptr<uint8_t>(),planar.data_ptr<int8_t>(),n,h,w,c,cp,selection.data_ptr<int>(),sc)
 if(pixels==128){RUN(128);}else{RUN(32);}
 #undef RUN
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {packed,planar};
}
'''
    prepare_declaration=host[:host.index(' {')]+';'
    bindings=('#include <torch/extension.h>\n'+declaration+'\n'+prepare_declaration+
              '\nPYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("packed_conv",&packed_conv);m.def("prepare4",&prepare4);}\n')
    return {'spatial_packed_common.cuh':header,'packed_conv.cu':source,
            'prepare.cu':prefix+quad+'\n'+host,'bindings.cpp':bindings}


@lru_cache(None)
def extension():return build_sources('planar_int4',sources())


def prepare(m,inputs):
    if m.spec['bits']!=4:raise ValueError('planar INT4 requires four bits')
    return extension().prepare4(inputs,m.quant_multiplier,m.activation,m.spec['padded_channels'],m.sparse_channel_map,m.sparse_channels)


def convolution(m,packed,planar,**kwargs):
    if kwargs.get('next_quantizer') is not None:raise ValueError('consumer prefixes not yet implemented')
    return invoke(m,packed,planar,_backend=extension(),_locations=m.compact_locations,**kwargs)
