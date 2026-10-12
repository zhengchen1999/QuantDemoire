# Third-party code

The ESDNet architecture in `quantdemoire/esdnet.py` is adapted from
[CVMI-Lab/UHDM](https://github.com/CVMI-Lab/UHDM), through the existing local
QuantDemoire implementation. Its Apache-2.0 license is retained in
[`licenses/UHDM-Apache-2.0.txt`](licenses/UHDM-Apache-2.0.txt).
Deployment-specific changes defer selected concatenations until activation
quantization. The network architecture and checkpoint tensor names are retained.

The CUDA extension depends on NVIDIA CUTLASS 3.5.1 (BSD-3-Clause), fetched as a
separate dependency by `uv run python -m kernel`. The extension instantiates
CUTLASS convolution templates and provides its own quantization, sparse branch,
and output operations. The dependency's copyright and license remain in its
source tree. CUDA, cuDNN, PyTorch, torchvision, LPIPS and other dependencies
retain their respective licenses. Datasets and original pretrained checkpoints
are not included in the Git repository.
