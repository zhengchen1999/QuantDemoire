"""Experimental sixteen-lane CSR rows with unchanged FP32 FMA ordering."""
from functools import lru_cache
from kernel.diagnostic import generated_sources, _replace_once, build_sources
from .sparse_prefetch import sources as host_sources
from .spatial_packed import convolution as invoke


def sources():
    header=generated_sources()['diagnostic_common.cuh']
    start=header.index('     constexpr int TileM=Warps==8?64:128,TileN=Variant==6?64:32,PixelsPerLane=TileM/32;')
    end=header.index('\n     __syncthreads();\n     }\n   }',start)
    body=header[start:end]
    body=_replace_once(body,'PixelsPerLane=TileM/32','PixelsPerLane=TileM/16')
    body=_replace_once(body,'int pos=pb+lane+t*32;', 'int pos=pb+(lane&15)+t*16;')
    body=_replace_once(body,'group<TileN/Warps','group<TileN/(Warps*2)')
    body=_replace_once(body,'int local=warp+group*Warps;', 'int local=(threadIdx.x>>4)+group*(Warps*2);')
    body=_replace_once(body,'(lane+t*32)*(TileN+1)+local', '((lane&15)+t*16)*(TileN+1)+local')
    header=header[:start]+body+header[end:]
    result=host_sources();result['spatial_packed_common.cuh']=header
    return result


@lru_cache(None)
def extension():return build_sources('sparse_subwarp',sources())


def convolution(m,packed,planar,**kwargs):
    return invoke(m,packed,planar,_backend=extension(),_locations=m.compact_locations,**kwargs)
