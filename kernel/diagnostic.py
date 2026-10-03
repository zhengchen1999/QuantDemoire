"""Source-built dense counterfactuals for branch-cost measurements only.

These entry points retain the integer tile and consumer caches but deliberately
drop all FP16 tail weights. They must never be used for quality evaluation.
Generated sources live in their own content-addressed cache; no prebuilt object
files or historical prototype directory are needed.
"""
import hashlib
import os
from functools import lru_cache
from pathlib import Path

import torch


def _replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"diagnostic source adapter needs review: {old[:100]!r}")
    return text.replace(old, new, 1)


def generated_sources():
    root = Path(__file__).parent / "csrc"
    header = (root / "joint_common.cuh").read_text()
    # Four-warp integer tiles have eight output elements per access. Retain that
    # tile and emit two adjacent four-element prefix words from the epilogue.
    start = header.index('     static_assert(Access==4,"joint producer')
    end = header.index('\n   }\n }\n};', start)
    prefix = header[start:end]
    prefix = _replace_once(prefix,
        '     static_assert(Access==4,"joint producer currently uses four-wide output");',
        '     static_assert(Access==4||Access==8,"invalid diagnostic access");\n'
        '     #pragma unroll\n'
        '     for(int chunk=0;chunk<Access;chunk+=4) {\n'
        '       int prefix_oc=oc+chunk;')
    # Only rewrite the isolated prefix block, never convolution coordinates.
    import re
    prefix = re.sub(r'\boc\b', 'prefix_oc', prefix)
    prefix = prefix.replace('int prefix_oc=prefix_oc+chunk;', 'int prefix_oc=oc+chunk;')
    prefix = prefix.replace('float(out[t])', 'float(out[chunk+t])') + '\n     }'
    header = header[:start] + prefix + header[end:]
    if header.count('if constexpr(Variant>=3) {') != 2:
        raise RuntimeError('diagnostic launch adapter needs review')
    header = header.replace('if constexpr(Variant>=3) {', 'if constexpr(Variant>=3&&Warps==8) {')
    outputs = {'diagnostic_common.cuh': header}
    declarations, bindings = [], []
    for filename, original in (
        ('joint.cu', 'joint_next'), ('joint_two.cu', 'joint_next_two'),
        ('joint_omit.cu', 'joint_next_omitted'),
        ('joint_omit_two.cu', 'joint_next_two_omitted'),
    ):
        source = (root / filename).read_text()
        start = source.index('std::vector<torch::Tensor> ' + original + '(')
        end = source.index('\n}', start) + 2
        source = source[start:end]
        name = original.replace('joint_', 'dense_')
        source = _replace_once(source, original + '(', name + '(')
        declarations.append(source[:source.index(' {')] + ';')
        bindings.append(f'  m.def("{name}", &{name});')
        source = _replace_once(source,
            'TORCH_CHECK(warps==8&&sparse&&!add_residual&&(variant==8||variant==9),"unsupported joint producer");',
            'TORCH_CHECK(!sparse&&!next_planar&&!add_residual,"dense diagnostic must disable sparse buffers");')
        source = source.replace(
            ' TORCH_CHECK(r>1&&packed_metadata&&balance,"prototype requires spatial balanced packed-metadata CSR");\n', '')
        if 'omitted' in name:
            source = _replace_once(source,
                ' TORCH_CHECK((bits==4||bits==8)&&r>1&&balance&&packed_metadata&&cp>(bits==4?64:48),\n'
                '   "feature omission requires regular spatial balanced packed-metadata producer");',
                ' TORCH_CHECK(bits==4||bits==8,"invalid diagnostic bit width");')
        start = source.index(' #define RUN(')
        end = source.index(' #undef RUN', start)
        # Strategy 1 exists only to transpose the sparse planar prefix. With
        # that buffer absent, direct packed stores are the matched dense path.
        dispatch = ''' #define RUN(B,A,W,V) run_joint_selected<B,A,W,false,V>(x,weight,scale,out,cp,params,stream)
 #define BOTH(A,W,V) if(bits==4){RUN(4,A,W,V);}else{RUN(8,A,W,V);}
 if(warps==4) {BOTH(8,4,11);}
 else if(few) {BOTH(4,8,31);}
 else if(retain_input) {BOTH(4,8,23);}
 else {BOTH(4,8,11);}
 #undef BOTH
'''
        source = source[:start] + dispatch + source[end:]
        defines = ''
        if 'two' in name:
            defines += '#define QUANTDEMOIRE_TWO_SUCCESSOR 1\n'
        if 'omitted' in name:
            defines += '#define QUANTDEMOIRE_OMIT_FEATURE 1\n'
        outputs[name + '.cu'] = defines + '#include "diagnostic_common.cuh"\n' + source + '\n'
    outputs['bindings.cpp'] = ('#include <torch/extension.h>\n' + '\n'.join(declarations)
        + '\nPYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {\n' + '\n'.join(bindings) + '\n}\n')
    return outputs


@lru_cache(None)
def extension():
    return build_sources('branch_diagnostic', generated_sources())


def build_sources(label, sources):
    """Build an isolated, content-addressed experimental source set."""
    # Initializes and validates the same CUDA/CUTLASS toolchain as production.
    from .ops import extension as production_extension
    production_extension()
    from torch.utils import cpp_extension
    digest = hashlib.sha256(b''.join(name.encode() + value.encode()
        for name, value in sorted(sources.items()))).hexdigest()
    name = 'quantdemoire_' + label + '_' + digest[:12]
    directory = Path(cpp_extension._get_build_directory(name, False))
    for filename, content in sources.items():
        path = directory / filename
        if not path.exists() or path.read_text() != content:
            path.write_text(content)
    cutlass = Path(os.environ.get('CUTLASS_ROOT', '.third_party/cutlass')).resolve()
    return cpp_extension.load(name=name,
        sources=[str(directory / filename) for filename in sources if not filename.endswith('.cuh')],
        extra_include_paths=[str(cutlass / 'include')],
        extra_cflags=['-O3', '-std=c++17'],
        extra_cuda_cflags=['-O3', '-std=c++17', '-lineinfo', '--expt-relaxed-constexpr'],
        verbose=os.environ.get('QUANTDEMOIRE_BUILD_VERBOSE') == '1')
