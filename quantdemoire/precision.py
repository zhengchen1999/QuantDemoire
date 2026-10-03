"""Reproducible FP16 operands / FP32 convolution accumulation policy.

The pinned PyTorch 2.2.2 cuDNN v7 ConvolutionDescriptor explicitly sets
CUDNN_DATA_FLOAT for CUDNN_DATA_HALF input. Select that path before torch import.
"""
import os
os.environ.setdefault("TORCH_CUDNN_V8_API_DISABLED", "1")

import torch


def configure():
    if os.environ.get("TORCH_CUDNN_V8_API_DISABLED") != "1":
        raise RuntimeError("Set TORCH_CUDNN_V8_API_DISABLED=1 before importing torch")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.set_num_threads(4)


def metadata():
    from kernel import ops
    import hashlib
    from pathlib import Path
    source=Path(__file__).parent
    deployment_hash=hashlib.sha256(b''.join(name.encode()+(source/name).read_bytes() for name in ('deploy.py','esdnet.py'))).hexdigest()
    return dict(torch=torch.__version__, cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(),
                fp16_convolution_accumulator="float32", cudnn_api="v7",
                tf32=False, fp16_reduced_precision_reduction=False,
                sparse_implementation=os.environ.get('QUANTDEMOIRE_SPARSE_IMPL','csr'),
                csr_kernel=os.environ.get('QUANTDEMOIRE_CSR_KERNEL','legacy'),
                joint_warps=int(os.environ.get('QUANTDEMOIRE_JOINT_WARPS','0')),
                joint_variant=int(os.environ.get('QUANTDEMOIRE_JOINT_VARIANT','0')),
                joint_next_quant=int(os.environ.get('QUANTDEMOIRE_JOINT_NEXT_QUANT','0')),
                balance_csr=os.environ.get('QUANTDEMOIRE_BALANCE_CSR','0')=='1',
                shared_retain_stages=os.environ.get('QUANTDEMOIRE_SHARED_RETAIN_STAGES','0')=='1',
                few_channels=os.environ.get('QUANTDEMOIRE_FEW_CHANNELS','0')=='1',
                packed_csr_meta=os.environ.get('QUANTDEMOIRE_PACKED_CSR_META','0')=='1',
                omit_cached_features=int(os.environ.get('QUANTDEMOIRE_OMIT_CACHED_FEATURES','0')),
                shared_skip_planar=os.environ.get('QUANTDEMOIRE_SHARED_SKIP_PLANAR','0')=='1',
                shared_skip_prefix_planar=os.environ.get('QUANTDEMOIRE_SHARED_SKIP_PREFIX_PLANAR','0')=='1',
                joint_plan_sha256=hashlib.sha256(Path(os.environ['QUANTDEMOIRE_JOINT_PLAN']).read_bytes()).hexdigest() if os.environ.get('QUANTDEMOIRE_JOINT_PLAN') else None,
                main_half_override=os.environ.get('QUANTDEMOIRE_MAIN_HALF'),
                compact_sparse=os.environ.get('QUANTDEMOIRE_COMPACT_SPARSE','1')=='1',
                prepare_implementation=os.environ.get('QUANTDEMOIRE_PREPARE_IMPL','pair'),
                fuse_next_quant=os.environ.get('QUANTDEMOIRE_FUSE_NEXT_QUANT','0')=='1',
        fuse_second_quant=os.environ.get('QUANTDEMOIRE_FUSE_SECOND_QUANT','0')=='1',
                fuse_attention=os.environ.get('QUANTDEMOIRE_FUSE_ATTENTION','0')=='1',
                fuse_residual=os.environ.get('QUANTDEMOIRE_FUSE_RESIDUAL','0')=='1',
                deployment_source_sha256=deployment_hash,
                release_features=os.environ.get("QUANTDEMOIRE_RELEASE_FEATURES","0")=="1",
                kernel_source_sha256=ops.loaded_source_sha256)
