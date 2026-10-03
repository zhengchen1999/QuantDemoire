"""Versioned, portable INT4/INT8 + CSR FP16 deployment format.

Fake quantization and CUDA inference consume exactly the same packed tensors.
INT4 stores the first signed value in the low nibble, in padded KRSC order.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file, save_file

from .esdnet import ESDNet
from .quantization.observers import calculate_qparams_from_bounds


class ChannelConcat:
    """Deferred channel concatenation, consumed by the fused quantization kernel."""
    def __init__(self, parts, prepared=None, second_cached=None):
        self.parts = []
        self.has_omitted_features = False
        for part in parts:
            self.parts.extend(part.parts if isinstance(part, ChannelConcat) else [part])
            self.has_omitted_features |= (part.has_omitted_features if isinstance(part,ChannelConcat)
                                          else getattr(part,'_quantdemoire_omitted',False))
        n, _, h, w = self.parts[0].shape
        if len(self.parts)>8 or any(p.shape[0]!=n or p.shape[2:]!=(h,w) for p in self.parts):
            raise ValueError("invalid deferred concatenation")
        self.shape = (n, sum(p.shape[1] for p in self.parts), h, w)
        # A prefix belongs to one consumer and one invocation. Never inherit a
        # nested concat's prefix: later consumers have different quantizers.
        self.prepared = prepared
        self.second_cached = second_cached


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pack_signed(q, bits):
    q = q.flatten().to(torch.int8)
    if bits == 8:
        return q.view(torch.uint8)
    if bits != 4 or q.numel() % 2:
        raise ValueError("INT4 requires an even element count")
    return ((q[::2] & 15) | ((q[1::2] & 15) << 4)).to(torch.uint8)


def unpack_signed(packed, bits):
    if bits == 8:
        return packed.view(torch.int8)
    v = torch.stack([packed & 15, packed >> 4], -1).flatten().to(torch.int8)
    return (v ^ 8) - 8


def balance_csr_rows(row):
    """Append a deterministic 32-row permutation; retain the original CSR data."""
    pointers = row.detach().cpu().tolist()
    k = len(pointers) - 1
    lengths = [end - begin for begin, end in zip(pointers, pointers[1:])]
    permutation = []
    for first in range(0, k, 32):
        work = [lengths[first + i] if first + i < k else 0 for i in range(32)]
        groups, loads = [[] for _ in range(8)], [0] * 8
        for i in sorted(range(32), key=lambda i: (-work[i], i)):
            warp = min((w for w in range(8) if len(groups[w]) < 4),
                       key=lambda w: (loads[w], len(groups[w]), w))
            groups[warp].append(i)
            loads[warp] += work[i]
        permutation.extend(groups[w][group] for group in range(4) for w in range(8))
    return torch.tensor(pointers + permutation, dtype=row.dtype, device=row.device)


class PackedConv(nn.Module):
    def __init__(self, spec, tensors, backend="cuda"):
        super().__init__()
        self.spec = spec
        self.backend = backend
        self.sparse_enabled = True
        # Enabled explicitly by quantdemoire.branch_diagnostic, never by a model
        # package or environment variable. This counterfactual is timing-only.
        self.diagnostic_main_only = False
        self.fused = spec.get("fused_sparse", False)
        self.sparse_impl = os.environ.get("QUANTDEMOIRE_SPARSE_IMPL", "csr")
        self.main_half = os.environ.get('QUANTDEMOIRE_MAIN_HALF', str(int(spec.get('main_output')=='float16'))) == '1'
        self.compact_sparse = os.environ.get('QUANTDEMOIRE_COMPACT_SPARSE','1') == '1'
        self.prepare_impl=os.environ.get('QUANTDEMOIRE_PREPARE_IMPL','pair')
        self.csr_kernel=os.environ.get('QUANTDEMOIRE_CSR_KERNEL','legacy')
        self.joint_warps=int(os.environ.get('QUANTDEMOIRE_JOINT_WARPS','0'))
        self.joint_variant=int(os.environ.get('QUANTDEMOIRE_JOINT_VARIANT','0'))
        self.shared_skip_planar=os.environ.get('QUANTDEMOIRE_SHARED_SKIP_PLANAR','0')=='1'
        self.shared_skip_prefix_planar=os.environ.get('QUANTDEMOIRE_SHARED_SKIP_PREFIX_PLANAR','0')=='1'
        self.joint_next_mode=int(os.environ.get('QUANTDEMOIRE_JOINT_NEXT_QUANT','0'))
        self.balance_csr=os.environ.get('QUANTDEMOIRE_BALANCE_CSR','0')=='1'
        self.retain_shared=os.environ.get('QUANTDEMOIRE_SHARED_RETAIN_STAGES','0')=='1'
        self.few_channels=os.environ.get('QUANTDEMOIRE_FEW_CHANNELS','0')=='1'
        self.packed_csr_meta=os.environ.get('QUANTDEMOIRE_PACKED_CSR_META','0')=='1'
        self.fuse_second_quant=os.environ.get('QUANTDEMOIRE_FUSE_SECOND_QUANT','0')=='1'
        # 0: retain features; 1: omit eligible joint outputs; 2: also separate CSR.
        self.omit_cached_features=int(os.environ.get('QUANTDEMOIRE_OMIT_CACHED_FEATURES','0'))
        if self.omit_cached_features not in (0,1,2):raise ValueError('invalid cached feature omission mode')
        self.register_buffer('balanced_row',None,persistent=False)
        if self.joint_next_mode not in (0,1,2):raise ValueError('joint next-quant mode must be 0, 1 or 2')
        if self.joint_variant not in range(10):raise ValueError('joint variant must be between 0 and 9')
        if self.joint_warps not in (0,4,8):raise ValueError('joint warps must be 0, 4 or 8')
        if self.joint_warps and (not self.main_half or self.sparse_impl!='csr'):
            raise ValueError('joint output requires a half main output and CSR branch')
        if self.csr_kernel not in ('legacy','linear','lut','wide','linear32','divmod'):raise ValueError('unknown CSR kernel')
        if self.csr_kernel!='legacy' and (not self.main_half or self.sparse_impl!='csr'):
            raise ValueError('tuned CSR kernels require a half main output and CSR branch')
        self.register_buffer('linear_offsets',None,persistent=False)
        self._linear_key=None
        if self.prepare_impl not in ('scalar','cached','pair','quad','quad_loop','quad_swizzle'):raise ValueError('unknown prepare implementation')
        if self.sparse_impl not in ('csr','tensorcore') or (self.main_half and self.sparse_impl!='csr'):
            raise ValueError('unsupported sparse implementation / main output combination')
        self._dense_cache = None
        self._sparse_cache = None
        self._half_cache = None
        for name, value in tensors.items():
            self.register_buffer(name, value, persistent=name not in ('epilogue_sums','sparse_channel_map','compact_locations','quant_multiplier'))
        if "locations" not in tensors:
            _, c, r, _ = spec["shape"]
            if c >= 4096 or r >= 16:
                raise ValueError("packed sparse coordinates exceed format limits")
            self.register_buffer("locations", (self.indices % c) | (((self.indices // c) % r) << 12) | ((self.indices // (c*r)) << 16))
        self.validate()
        self.refresh_runtime()
        if self.compact_sparse and self.sparse_impl=='csr':
            self.make_compact_sparse()
        if self.sparse_impl == "tensorcore":
            self.make_sparse_blocks()

    @torch.no_grad()
    def refresh_runtime(self):
        """Refresh derived output sums after changing scales during calibration."""
        k,_,r,_=self.spec['shape']
        flat=self.offset_sums.reshape(k,r*r)
        total=torch.zeros_like(self.bias)
        # Preserve the CUDA scalar loop's summation order.
        for j in range(r*r):total=total+flat[:,j]
        packed=torch.cat([flat.flatten(),total])
        if hasattr(self,'epilogue_sums'):
            self.epilogue_sums=packed
        else:
            self.register_buffer('epilogue_sums',packed,persistent=False)
        multiplier=self.inverse_smooth/self.activation[0]
        if hasattr(self,'quant_multiplier'):self.quant_multiplier=multiplier
        else:self.register_buffer('quant_multiplier',multiplier,persistent=False)
        if self.balance_csr and self.backend=='cuda':
            self.balanced_row=balance_csr_rows(self.row)

    def make_sparse_blocks(self):
        if hasattr(self, 'block_row'):
            return
        rows = self.row.cpu().tolist()
        indices = self.locations.cpu()
        values = self.values.cpu()
        k = self.spec['shape'][0]
        pointers, coords, coefficients = [0], [], []
        for first in range(0, k, 8):
            end = min(first+8, k)
            union = indices[rows[first]:rows[end]].unique(sorted=True)
            length = (union.numel()+15)//16*16
            loc = torch.full((length,), -1, dtype=torch.int32)
            loc[:union.numel()] = union
            weight = torch.zeros(length, 8, dtype=torch.float16)
            for oc in range(first, end):
                positions = torch.searchsorted(union, indices[rows[oc]:rows[oc+1]])
                weight[positions, oc-first] = values[rows[oc]:rows[oc+1]]
            pointers.append(pointers[-1]+length)
            coords.append(loc)
            coefficients.append(weight)
        for key, tensor in dict(block_row=torch.tensor(pointers,dtype=torch.int32),
                                block_locations=torch.cat(coords),block_values=torch.cat(coefficients)).items():
            self.register_buffer(key, tensor.to(self.values.device))

    def make_compact_sparse(self):
        """Omit planar channels never referenced by the unchanged CSR weights."""
        channels=(self.locations&4095).unique(sorted=True)
        c=self.spec['shape'][1]
        self.sparse_channels=int(channels.numel())
        mapping=torch.full((c,),-1,device=self.locations.device,dtype=torch.int32)
        mapping[channels.long()]=torch.arange(self.sparse_channels,device=mapping.device,dtype=torch.int32)
        locations=(self.locations&~4095)|mapping[(self.locations&4095).long()]
        for name,tensor in [('sparse_channel_map',mapping),('compact_locations',locations)]:
            if hasattr(self,name):setattr(self,name,tensor)
            else:self.register_buffer(name,tensor,persistent=False)
        self._linear_key=None

    def sparse_linear_offsets(self,locations,h,w,dilation,packed=False):
        # Derived inference cache: refresh_runtime invalidates it when model
        # buffers are refreshed. A new representation never mutates an earlier
        # returned offset tensor retained by a caller.
        key=(h,w,dilation,locations.data_ptr(),bool(packed),self.values.data_ptr() if packed else 0)
        if self._linear_key!=key:
            self.linear_offsets=(locations&4095).long()*h*w+((locations>>16)&15).long()*dilation*w+((locations>>12)&15).long()*dilation
            if packed:
                weight_bits=self.values.view(torch.int16).long().bitwise_and_(65535).bitwise_left_shift_(32)
                self.linear_offsets.bitwise_or_(weight_bits)
            self._linear_key=key
        return self.linear_offsets

    def validate(self):
        s = self.spec
        k, c, r, r2 = s["shape"]
        if s["bits"] not in (4, 8) or min(k,c,r) <= 0 or r != r2 or c>=4096 or r>=16 or k%4 or s["padded_channels"] < c:
            raise ValueError("invalid packed convolution metadata")
        if s['padded_channels'] % (32 if s['bits']==4 else 16) or s['stride'] <= 0 or s['dilation'] <= 0 or s['padding'] < 0:
            raise ValueError("invalid convolution geometry")
        required = dict(packed_weight=torch.uint8,row=torch.int32,indices=torch.int32,values=torch.float16,
                        locations=torch.int32,weight_scale=torch.float32,output_scale=torch.float32,
                        inverse_smooth=torch.float32,activation=torch.float32,bias=torch.float32,offset_sums=torch.float32)
        for name,dtype in required.items():
            value=getattr(self,name)
            if value.dtype != dtype or not value.is_contiguous() or not torch.isfinite(value).all():
                raise ValueError(f'invalid tensor {name}')
        if self.activation.shape != (2,) or self.weight_scale.shape != (k,) or self.output_scale.shape != (k,) or self.bias.shape != (k,) or self.inverse_smooth.shape != (c,) or self.offset_sums.numel()!=k*r*r:
            raise ValueError('invalid scale or bias shape')
        if (self.weight_scale <= 0).any() or (self.inverse_smooth <= 0).any() or self.activation[1] != self.activation[1].round() or not 0 <= self.activation[1] <= 2**s['bits']-1:
            raise ValueError('invalid scale or zero point')
        row = self.row.cpu().long()
        if row.numel() != k + 1 or row[0] != 0 or row[-1] != self.values.numel():
            raise ValueError("invalid CSR row pointers")
        if (row[1:] < row[:-1]).any() or self.indices.numel() != self.values.numel():
            raise ValueError("invalid CSR structure")
        if self.indices.numel() and (self.indices.min() < 0 or self.indices.max() >= c*r*r):
            raise ValueError("CSR index out of range")
        for start,end in zip(row[:-1],row[1:]):
            indices=self.indices[int(start):int(end)]
            if indices.numel()>1 and (indices[1:]<=indices[:-1]).any():
                raise ValueError('CSR row must contain strictly sorted unique indices')
        coords=(self.indices % c) | (((self.indices // c) % r) << 12) | ((self.indices // (c*r)) << 16)
        if not torch.equal(self.locations, coords):
            raise ValueError('derived sparse coordinates differ')
        if self.packed_weight.numel() != k*r*r*s["padded_channels"]*s["bits"]//8:
            raise ValueError("invalid packed weight size")
        if not torch.isfinite(self.activation).all() or self.activation[0] <= 0:
            raise ValueError("invalid activation quantizer")

    def dense_weight(self):
        if self._dense_cache is None or self._dense_cache.device != self.values.device:
            k, c, r, _ = self.spec["shape"]
            q = unpack_signed(self.packed_weight, self.spec["bits"])
            q = q.reshape(k, r, r, self.spec["padded_channels"])[..., :c].float()
            weight = q * self.weight_scale[:, None, None, None]
            row = self.row.cpu().tolist()
            for o in range(k):
                weight[o].flatten()[self.indices[row[o]:row[o+1]].long()] = self.values[row[o]:row[o+1]].float()
            self._dense_cache = weight.permute(0, 3, 1, 2).contiguous()
            sparse = torch.zeros_like(weight)
            for o in range(k):
                sparse[o].flatten()[self.indices[row[o]:row[o+1]].long()] = self.values[row[o]:row[o+1]].float()
            self._sparse_cache = sparse.permute(0, 3, 1, 2).contiguous()
        return self._dense_cache

    def fake_forward(self, x):
        s = self.spec
        a, z = self.activation.unbind()
        # Match the CUDA expression and half input when validating deployment.
        q = torch.round(x.float() * (self.inverse_smooth/a)[None, :, None, None] + z)
        q = (q.clamp(0, 2**s["bits"] - 1) - z) * a
        if self.main_half:
            normal=self.dense_weight()-self._sparse_cache
            out=F.conv2d(q,normal,self.bias,s['stride'],s['padding'],s['dilation']).half().float()
            out+=F.conv2d(q.half().float(),self._sparse_cache,None,s['stride'],s['padding'],s['dilation'])
            return F.relu(out) if s.get('relu',False) else out
        out = F.conv2d(q, self.dense_weight(), self.bias, s["stride"], s["padding"], s["dilation"])
        # Sparse operands are FP16; accumulation remains FP32. This correction
        # makes the oracle match operand rounding without changing dense INT MACs.
        out = out + F.conv2d(q.half().float()-q, self._sparse_cache, None,
                             s["stride"], s["padding"], s["dilation"])
        return F.relu(out) if s.get("relu", False) else out

    def uses_shared_pointwise(self):
        s=self.spec
        return (self.joint_variant==9 and self.joint_warps==8 and s['shape'][2]==1
                and s['stride']==1 and s['padding']==0
                and s['padded_channels']<=((512 if s['bits']==4 else 256) if self.retain_shared else (256 if s['bits']==4 else 128)))

    def supports_prepared_input(self):
        return (self.backend == 'cuda' and self.main_half and (self.sparse_enabled or self.diagnostic_main_only)
                and self.sparse_impl == 'csr' and self.compact_sparse
                and self.prepare_impl in ('pair','quad','quad_loop','quad_swizzle') and self.sparse_channels > 0)

    def supports_next_quant(self):
        return self.supports_prepared_input() and (self.joint_warps==0 or
                (self.joint_warps==8 and self.joint_variant in (8,9) and self.joint_next_mode>0))

    def supports_residual(self):
        return self.backend=='cuda' and self.main_half and (self.sparse_enabled or self.diagnostic_main_only) and self.sparse_impl=='csr'

    def can_quantize_for(self, consumer):
        return (isinstance(consumer, PackedConv) and self.supports_next_quant()
                and consumer.supports_prepared_input() and self.spec['shape'][0] % 16 == 0
                and consumer.spec['shape'][1] == self.spec['shape'][0] + self.spec['shape'][1]
                and self.spec['shape'][1] % 2 == 0)

    def can_quantize_second_for(self, middle, consumer):
        return (self.fuse_second_quant and self.can_quantize_for(middle)
                and middle.can_quantize_for(consumer)
                and self.joint_warps==8 and self.joint_next_mode>0
                and self.balance_csr and self.packed_csr_meta and self.spec['shape'][2]>1
                and self.spec['shape'][0]==middle.spec['shape'][0]==32
                and self.spec['stride']==middle.spec['stride']==1
                and all(m.prepare_impl=='quad_swizzle' for m in (self,middle,consumer)))

    def can_omit_cached_feature(self,x):
        common=(self.backend=='cuda' and self.main_half and (self.sparse_enabled or self.diagnostic_main_only)
                and self.sparse_impl=='csr' and self.prepare_impl=='quad_swizzle' and self.compact_sparse)
        joint=(self.joint_warps==8 and self.joint_next_mode>0 and self.joint_variant in (8,9)
               and self.balance_csr and self.packed_csr_meta and self.spec['shape'][2]>1
               and self.spec['padded_channels']>(64 if self.spec['bits']==4 else 48))
        separate=self.joint_warps==0 and self.csr_kernel=='divmod'
        n,_,h,w=x.shape
        return common and (joint or separate) and n*self.sparse_channels*h*w<2147483647

    def forward(self, x, next_quantizer=None, residual=None, future_quantizer=None, omit_feature=False):
        if omit_feature and (next_quantizer is None or residual is not None or not self.can_omit_cached_feature(x)):
            raise ValueError('unsupported omitted-feature producer')
        has_omitted=(x.has_omitted_features if isinstance(x,ChannelConcat)
                     else getattr(x,'_quantdemoire_omitted',False))
        if has_omitted and self.backend!='cuda':
            raise RuntimeError('omitted features require a CUDA consumer with owned caches')
        if next_quantizer is not None and not self.can_quantize_for(next_quantizer):
            raise ValueError('unsupported producer / next quantizer combination')
        if future_quantizer is not None and not self.can_quantize_second_for(next_quantizer,future_quantizer):
            raise ValueError('unsupported second consumer quantizer')
        if residual is not None and (not self.supports_residual() or next_quantizer is not None):
            raise ValueError('unsupported fused residual path')
        if self.backend == 'fake_fp16':
            x=torch.cat(x.parts,1) if isinstance(x,ChannelConcat) else x
            a,z=self.activation.unbind()
            q=(x.float()*(self.inverse_smooth/a)[None,:,None,None]+z).round().clamp(0,2**self.spec['bits']-1)
            dq=((q-z)*a).half()
            if self._half_cache is None:
                self._half_cache=self.dense_weight().half().contiguous(memory_format=torch.channels_last)
            y=F.conv2d(dq,self._half_cache,self.bias.half(),self.spec['stride'],self.spec['padding'],self.spec['dilation'])
            return F.relu(y) if self.spec.get('relu',False) else y
        if self.backend == "fake":
            return self.fake_forward(torch.cat(x.parts,1) if isinstance(x,ChannelConcat) else x)
        from kernel.ops import extension
        s = self.spec
        parts = x.parts if isinstance(x,ChannelConcat) else [x]
        if any(p.dtype != torch.float16 for p in parts):
            raise TypeError("CUDA deployment expects FP16 network features")
        op = extension()
        n, c, h, w = x.shape
        if has_omitted:
            prepared=x.prepared if isinstance(x,ChannelConcat) else None
            covered=0 if prepared is None else 1+int(x.second_cached is not None)
            normal=[]
            for index,part in enumerate(parts):
                if getattr(part,'_quantdemoire_omitted',False):
                    if index>=covered or self.prepare_impl!='quad_swizzle':
                        raise RuntimeError('omitted feature has no consumer-specific cache coverage')
                    if part.untyped_storage().nbytes()!=2 or part.stride()!=(0,0,0,0):
                        raise RuntimeError('invalid omitted-feature metadata tensor')
                    normal.append(part)
                else:normal.append(part.contiguous(memory_format=torch.channels_last))
            parts=normal
            del normal,prepared
        else:
            parts = [p.contiguous(memory_format=torch.channels_last) for p in parts]
        locations=self.locations
        prepared = x.prepared if isinstance(x,ChannelConcat) else None
        prepare_planar=self.sparse_enabled and not (self.shared_skip_planar and self.uses_shared_pointwise())
        if prepared is not None:
            consumer, packed, planar, skip = prepared
            if consumer is not self or not self.supports_prepared_input():
                raise ValueError('prepared activation belongs to a different consumer or execution mode')
            second_cached=x.second_cached
            if second_cached is not None:
                if self.prepare_impl!='quad_swizzle':raise ValueError('second cache requires quad_swizzle preparation')
                packed,planar=op.prepare_cached_second(parts,self.quant_multiplier,self.activation,s['bits'],s['padded_channels'],self.sparse_channel_map,self.sparse_channels,packed,planar,skip,prepare_planar,second_cached)
            else:
                packed,planar=op.prepare_next_cat(parts,self.quant_multiplier,self.activation,s['bits'],s['padded_channels'],self.sparse_channel_map,self.sparse_channels,packed,planar,skip,{'pair':2,'quad':3,'quad_loop':4,'quad_swizzle':5}[self.prepare_impl],prepare_planar)
            locations=self.compact_locations
            # Consume the cache once. Keeping it on the concat would retain the
            # integer input while the next producer allocates another one.
            x.prepared = None
            x.second_cached = None
            del second_cached
        elif self.prepare_impl!='scalar' and self.compact_sparse and self.sparse_impl=='csr' and self.sparse_channels:
            packed,planar=op.prepare_cached_cat(parts,self.quant_multiplier,self.activation,s['bits'],s['padded_channels'],self.sparse_channel_map,self.sparse_channels,{'cached':1,'pair':2,'quad':3,'quad_loop':4,'quad_swizzle':5}[self.prepare_impl],prepare_planar)
            locations=self.compact_locations
        elif self.compact_sparse and self.sparse_impl=='csr' and self.sparse_enabled and self.sparse_channels:
            packed,planar=op.prepare_compact_cat(parts,self.inverse_smooth,self.activation,s['bits'],s['padded_channels'],self.sparse_channel_map,self.sparse_channels)
            locations=self.compact_locations
        elif len(parts)==1:
            packed, planar = op.prepare(parts[0], self.inverse_smooth, self.activation, s["bits"], s["padded_channels"], self.sparse_enabled)
        else:
            packed, planar = op.prepare_cat(parts, self.inverse_smooth, self.activation, s["bits"], s["padded_channels"], self.sparse_enabled)
        del prepared
        k, _, r, _ = s["shape"]
        if self.diagnostic_main_only:
            from .branch_diagnostic import dense_forward
            return dense_forward(self, packed, planar, locations, next_quantizer,
                                 residual, future_quantizer, omit_feature)
        if self.main_half:
            if self.joint_warps:
                joint_locations=self.locations if self.uses_shared_pointwise() else locations
                balanced=self.balance_csr and self.joint_warps==8 and self.joint_variant in (8,9)
                joint_row=self.balanced_row if balanced else self.row
                packed_meta=(self.packed_csr_meta and self.sparse_enabled and self.joint_warps==8
                             and self.joint_variant in (8,9) and not self.uses_shared_pointwise()
                             and planar.numel()<2147483647)
                if next_quantizer is not None:
                    nxt=next_quantizer
                    if omit_feature:
                        joint_next=op.joint_next_two_omitted if future_quantizer is not None else op.joint_next_omitted
                    else:
                        joint_next=op.joint_next_two if future_quantizer is not None else op.joint_next
                    extra=() if future_quantizer is None else (future_quantizer.quant_multiplier,
                        future_quantizer.activation,future_quantizer.spec['bits'],32)
                    result=joint_next(packed,planar,self.packed_weight,self.output_scale,joint_row,joint_locations,
                    self.sparse_linear_offsets(locations,h,w,s['dilation'],packed=packed_meta),self.values,self.epilogue_sums,self.bias,self.activation,
                    residual if residual is not None else self.values,s['bits'],s['padded_channels'],k,r,s['stride'],s['padding'],s['dilation'],
                    s.get('relu',False),self.sparse_enabled,residual is not None,self.joint_warps,self.joint_variant,
                        nxt.quant_multiplier,nxt.activation,nxt.sparse_channel_map,nxt.spec['bits'],
                        nxt.spec['padded_channels'],nxt.sparse_channels,
                        not (nxt.shared_skip_planar and nxt.shared_skip_prefix_planar and nxt.uses_shared_pointwise()),self.joint_next_mode-1,balanced,self.retain_shared,self.few_channels,packed_meta,*extra)
                    if omit_feature:result[0]._quantdemoire_omitted=True
                    return result
                return op.joint_conv(packed,planar,self.packed_weight,self.output_scale,joint_row,joint_locations,
                    self.sparse_linear_offsets(locations,h,w,s['dilation'],packed=packed_meta),self.values,self.epilogue_sums,self.bias,self.activation,
                    residual if residual is not None else self.values,s['bits'],s['padded_channels'],k,r,s['stride'],s['padding'],s['dilation'],
                    s.get('relu',False),self.sparse_enabled,residual is not None,self.joint_warps,self.joint_variant,balanced,self.retain_shared,self.few_channels,packed_meta)
            out=op.fused_conv(packed,planar,self.packed_weight,self.output_scale,
                              self.row,locations,self.values,self.epilogue_sums,self.bias,self.activation,
                              s['bits'],s['padded_channels'],k,r,s['stride'],s['padding'],s['dilation'],
                              s.get('relu',False) and not self.sparse_enabled,False)
            if not self.sparse_enabled:
                return out
            if self.csr_kernel!='legacy':
                nxt=next_quantizer
                if nxt is not None:del packed
                finish=op.finish_half_omitted if omit_feature else op.finish_half_tuned
                result=finish(planar,out,self.row,locations,self.sparse_linear_offsets(locations,h,w,s['dilation']),
                    self.values,self.activation,s['bits'],r,s['stride'],s['padding'],s['dilation'],s.get('relu',False),
                    residual if residual is not None else out,residual is not None,
                    nxt.quant_multiplier if nxt is not None else self.activation,nxt.activation if nxt is not None else self.activation,
                    nxt.sparse_channel_map if nxt is not None else self.row,nxt.spec['bits'] if nxt is not None else 0,
                    nxt.spec['padded_channels'] if nxt is not None else 0,nxt.sparse_channels if nxt is not None else 0,
                    {'linear':1,'lut':2,'wide':3,'linear32':4,'divmod':5}[self.csr_kernel],
                    not (nxt is not None and nxt.shared_skip_planar and nxt.shared_skip_prefix_planar and nxt.uses_shared_pointwise()))
                if omit_feature:result[0]._quantdemoire_omitted=True
                return result if nxt is not None else result[0]
            if residual is not None:
                return op.finish_half_residual(planar,out,self.row,locations,self.values,self.activation,
                    s['bits'],r,s['stride'],s['padding'],s['dilation'],s.get('relu',False),residual)
            if next_quantizer is not None:
                # The integer convolution has consumed this buffer on the same
                # stream; its storage can be reused by the following epilogue.
                del packed
                nxt=next_quantizer
                return op.finish_half_next(planar,out,self.row,locations,self.values,self.activation,
                    s['bits'],r,s['stride'],s['padding'],s['dilation'],s.get('relu',False),
                    nxt.quant_multiplier,nxt.activation,nxt.sparse_channel_map,
                    nxt.spec['bits'],nxt.spec['padded_channels'],nxt.sparse_channels)
            return op.finish_half(planar,out,self.row,locations,self.values,self.activation,
                                  s['bits'],r,s['stride'],s['padding'],s['dilation'],s.get('relu',False))
        if self.fused or (n*h*w<=128 and r==1 and os.environ.get("QUANTDEMOIRE_FUSE_SMALL","1")=="1"):
            return op.fused_conv(packed, planar, self.packed_weight, self.output_scale,
                                 self.row, locations, self.values, self.epilogue_sums, self.bias,
                                 self.activation, s['bits'], s['padded_channels'], k, r,
                                 s['stride'], s['padding'], s['dilation'], s.get('relu',False), self.sparse_enabled)
        out = op.integer_conv(packed, self.packed_weight, self.output_scale,
                              s["bits"], n, h, w, s["padded_channels"], k, r,
                              s["stride"], s["padding"], s["dilation"])
        if self.sparse_impl == "tensorcore" and self.sparse_enabled:
            return op.finish_tc(planar, out, self.block_row, self.block_locations, self.block_values,
                                self.offset_sums, self.bias, self.activation, s["bits"], r,
                                s["stride"], s["padding"], s["dilation"], s.get("relu",False))
        return op.finish(planar, out, self.row, locations, self.values, self.epilogue_sums,
                         self.bias, self.activation, s["bits"], r, s["stride"],
                         s["padding"], s["dilation"], s.get("relu", False), self.sparse_enabled)


def _replace(model, name, layer):
    parent, _, child = name.rpartition(".")
    setattr(model.get_submodule(parent) if parent else model, child, layer)


@torch.no_grad()
def export_legacy(checkpoint, state_path, bits, output, tail_fraction=None, weight_range="stored"):
    if bits not in (4, 8):
        raise ValueError("bits must be 4 or 8")
    model = ESDNet(48, 32, 64, 32, 1).eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    states = torch.load(state_path, map_location="cpu", weights_only=True)
    tensors, layers = {}, {}
    for name, conv in model.named_modules():
        if not isinstance(conv, nn.Conv2d):
            continue
        state = states[name]
        if state["skip_quantization"]:
            tensors[name + ".weight"] = conv.weight.detach().contiguous()
            tensors[name + ".bias"] = conv.bias.detach().contiguous()
            layers[name] = {"kind": "float"}
            continue
        if not state["activation_ready"] or not state["weights_quantized"]:
            raise ValueError(f"uncalibrated layer {name}")
        if state.get("ocs_duplicate_index", -1) != -1 or state.get("activation_shift", torch.zeros(1)).abs().max() > 0:
            raise ValueError("OCS / activation shifts are not supported by this format")
        if state["weight_zero_point"].abs().max() > 0:
            raise ValueError("symmetric weights are required")
        k, c, r, _ = conv.weight.shape
        mask = state["outlier_mask"].bool()
        smooth = state["smooth_scale"].float()
        source = conv.weight.detach() * smooth[None, :, None, None]
        # Rank coverage verifies both tails independently, including ties.
        order = source.flatten().argsort(stable=True)
        required = math.ceil(source.numel() * .0025)
        if tail_fraction is not None:
            if not .0025 <= tail_fraction <= .005:
                raise ValueError("tail_fraction must be between 0.0025 and 0.005")
            count = math.ceil(source.numel()*tail_fraction)
            mask = torch.zeros_like(mask).flatten()
            mask[order[:count]] = True
            mask[order[-count:]] = True
            mask = mask.reshape_as(source)
        if not mask.flatten()[order[:required]].all() or not mask.flatten()[order[-required:]].all():
            raise ValueError(f"{name}: each weight tail must preserve at least 0.25%")
        scale = state["weight_scale"].float().flatten()
        weight_source = state["quantized_weight"]
        if weight_range != "stored":
            if weight_range not in ("minmax", "mse"):
                raise ValueError("unsupported weight range")
            weight_source = source
            normal = source.masked_fill(mask, 0).reshape(k, -1)
            peak = normal.abs().amax(1).clamp_min(1e-8)
            scale = peak / ((1 << (bits-1))-1)
            if weight_range == "mse":
                best_error = torch.full((k,), float('inf'))
                best_scale = scale.clone()
                for ratio in torch.linspace(.7, 1., 31):
                    candidate = scale*ratio
                    deq = (normal/candidate[:,None]).round().clamp(-(1<<(bits-1)),(1<<(bits-1))-1)*candidate[:,None]
                    error = (deq-normal).square().sum(1)
                    improve = error < best_error
                    best_error[improve] = error[improve]
                    best_scale[improve] = candidate[improve]
                scale = best_scale
        qw = torch.round(weight_source / scale[:, None, None, None])
        if tail_fraction is not None or weight_range != "stored":
            qw.clamp_(-(1 << (bits-1)), (1 << (bits-1))-1)
        qw[mask] = 0
        if qw.min() < -(1 << (bits-1)) or qw.max() >= (1 << (bits-1)):
            raise ValueError(f"weight codes do not fit INT{bits}: {name}")
        # Export exactly the calibrated outlier values, rounded to stored FP16.
        values_source = weight_source.permute(0, 2, 3, 1).contiguous()
        mask_nhwc = mask.permute(0, 2, 3, 1).contiguous().reshape(k, -1)
        row = torch.cat([torch.zeros(1, dtype=torch.int64), mask_nhwc.sum(1).cumsum(0)]).int()
        indices = torch.cat([torch.where(m)[0] for m in mask_nhwc]).int()
        values = values_source.flatten()[mask_nhwc.flatten()].half()
        a, z, _, _ = calculate_qparams_from_bounds(state["lb_a"].float(), state["ub_a"].float(), bits, False)
        cp = math.ceil(c / (32 if bits == 4 else 16)) * (32 if bits == 4 else 16)
        qrsc = qw.permute(0, 2, 3, 1).contiguous()
        padded = F.pad(qrsc, (0, cp-c)).to(torch.int8)
        params = dict(packed_weight=pack_signed(padded, bits), row=row, indices=indices, values=values,
                      weight_scale=scale, output_scale=scale*a, inverse_smooth=smooth.reciprocal(),
                      activation=torch.stack([a, z]).flatten(), bias=conv.bias.detach().float(),
                      offset_sums=qrsc.sum(-1)*scale[:, None, None]*a)
        spec = dict(kind="integer", bits=bits, shape=list(conv.weight.shape), padded_channels=cp,
                    stride=conv.stride[0], padding=conv.padding[0], dilation=conv.dilation[0],
                    sparse_format="csr_krsc", outliers=int(mask.sum()), min_tail_fraction=.0025,
                    lower_tail_count=int(mask.flatten()[order[:source.numel()//2]].sum()),
                    upper_tail_count=int(mask.flatten()[order[source.numel()//2:]].sum()))
        PackedConv(spec, params, "fake")  # validate before saving
        layers[name] = spec
        tensors.update({name + "." + key: v.contiguous() for key, v in params.items()})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output / "model.safetensors"), metadata={"format": "quantdemoire-v1"})
    metadata = dict(format="quantdemoire-v1", architecture="esdnet_48_32_64_32_1", bits=bits,
                    checkpoint_sha256=sha256(checkpoint), calibration_sha256=sha256(state_path),
                    weights_sha256=sha256(output / "model.safetensors"), layers=layers,
                    requested_tail_fraction=tail_fraction,
                    weight_range=weight_range,
                    semantics="asymmetric activation; symmetric integer weight; quantized activation times FP16 outlier")
    (output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def load_model(directory, backend="cuda", device="cuda", fuse_relu=True):
    if backend not in ('cuda','fake','fake_fp16'):
        raise ValueError('unsupported deployment backend')
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    if config["format"] != "quantdemoire-v1":
        raise ValueError("unsupported model format")
    if sha256(directory / "model.safetensors") != config["weights_sha256"]:
        raise ValueError("weight checksum mismatch")
    tensors = load_file(str(directory / "model.safetensors"))
    model = ESDNet(48, 32, 64, 32, 1).eval()
    for name, spec in config["layers"].items():
        local = {key[len(name)+1:]: value for key, value in tensors.items() if key.startswith(name + ".")}
        if spec["kind"] == "float":
            conv = model.get_submodule(name)
            conv.load_state_dict(local, strict=True)
            if backend in ("cuda",'fake_fp16'):
                conv.half()
        else:
            _replace(model, name, PackedConv(spec, local, backend))
    plan_path=os.environ.get('QUANTDEMOIRE_JOINT_PLAN')
    if backend=='cuda' and plan_path:
        plan=json.loads(Path(plan_path).read_text())
        if plan.get('format')!='quantdemoire-joint-plan-v1' or plan.get('model_sha256')!=config['weights_sha256']:
            raise ValueError('joint dispatch plan format or weight checksum mismatch')
        names={name for name,spec in config['layers'].items() if spec['kind']=='integer'}
        if set(plan['layers'])!=names or any(type(v) is not int or v not in (0,4,8) for v in plan['layers'].values()):
            raise ValueError('joint dispatch plan must cover every quantized layer with 0, 4 or 8 warps')
        next_modes=plan.get('joint_next_modes')
        if next_modes is not None:
            if set(next_modes)!=names or any(type(v) is not int or v not in (0,1,2) for v in next_modes.values()):
                raise ValueError('joint next-quant plan must cover every quantized layer with modes 0, 1 or 2')
            if any(mode and plan['layers'][name]!=8 for name,mode in next_modes.items()):
                raise ValueError('joint producer mode requires eight-warps dispatch')
        for name,warps in plan['layers'].items():
            m=model.get_submodule(name)
            if warps and (not m.main_half or m.sparse_impl!='csr'):
                raise ValueError('joint dispatch requires a half main output and CSR branch')
            m.joint_warps=warps
            # The global flag can disable all joint producers; enabled profiles
            # may override each layer's strategy or disable that layer alone.
            if next_modes is not None and m.joint_next_mode>0:
                m.joint_next_mode=next_modes[name]
                if m.joint_next_mode and m.joint_variant not in (8,9):
                    raise ValueError('joint producer plan requires variant 8 or 9')
    if fuse_relu:
        for module in model.modules():
            if isinstance(module, nn.Sequential):
                for i in range(len(module)-1):
                    if isinstance(module[i], PackedConv) and isinstance(module[i+1], nn.ReLU):
                        module[i].spec["relu"] = True
                        module[i+1] = nn.Identity()
    from .esdnet import DenseBlock, Decoder
    for module in model.modules():
        if isinstance(module,(DenseBlock,Decoder)):
            module.lazy_concat = backend == 'cuda'
        if isinstance(module,DenseBlock):
            module.fuse_next_quant = backend == 'cuda' and fuse_relu and os.environ.get('QUANTDEMOIRE_FUSE_NEXT_QUANT','0') == '1'
    return model.to(device=device, memory_format=torch.channels_last)
