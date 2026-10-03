"""CUDA Graph inference with changing inputs of a fixed shape."""
import torch


class CudaGraphRunner:
    """Outputs are reused; clone them to retain results across subsequent calls."""
    @torch.inference_mode()
    def __init__(self, model, example, warmup=3):
        if not example.is_cuda or example.dtype!=torch.float16:
            raise ValueError('graph runner requires CUDA FP16 inputs')
        self.model=model.eval()
        self.input=example.clone(memory_format=torch.channels_last)
        stream=torch.cuda.Stream(device=example.device)
        stream.wait_stream(torch.cuda.current_stream(example.device))
        with torch.cuda.stream(stream):
            for _ in range(warmup):model(self.input)
        torch.cuda.current_stream(example.device).wait_stream(stream)
        torch.cuda.synchronize(example.device)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output=model(self.input)

    @torch.inference_mode()
    def __call__(self, tensor):
        if tensor.shape!=self.input.shape or tensor.dtype!=self.input.dtype or tensor.device!=self.input.device:
            raise ValueError('input shape, dtype or device differs from captured graph')
        self.input.copy_(tensor)
        self.graph.replay()
        return self.output
