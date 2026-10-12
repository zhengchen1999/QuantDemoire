# QuantDemoire deployment format, version 1

Each model folder contains `config.json` and `model.safetensors`. Inference
requires no pickle checkpoint, calibration object, or original FP32 weights.
The config records architecture, bit width, convolution geometry, quantization
provenance, tail counts, and SHA-256 of the tensor file. Optional search,
calibration and evaluation files document the selected artifact.

For each of the 157 quantized convolutions:

| Tensor | Layout and meaning |
|---|---|
| `packed_weight` | Signed two's-complement INT4/INT8 in K,R,S,C order; INT4's first code occupies the low nibble. Input channels are padded to 32 (INT4) or 16 (INT8). |
| `weight_scale` | Positive FP32 scale per output channel. |
| `activation` | Two FP32 values: positive activation step and integer unsigned zero point. |
| `inverse_smooth` | FP32 inverse smoothing scale per input channel. |
| `row`, `indices`, `values` | CSR rows are output channels; indices flatten R,S,C **without** channel padding; values are FP16. Integer weights at these coordinates are zero. |
| `locations` | Optional derived sparse coordinates: `ic | (rx << 12) | (ry << 16)`. |
| `output_scale` | Activation step times weight scale. |
| `offset_sums` | Per-output, per-filter-position integer weight sums times output scale, for asymmetric zero-point correction at image boundaries. |
| `bias` | FP32 convolution bias. |

The first and final convolutions keep their floating weights and are executed
with FP16 operands and FP32 accumulation. Quantized layers use
`q_u = clip(round(x * inverse_smooth / a + zero_point), 0, 2**bits - 1)`.
The packed signed activation is `q_u - 2**(bits-1)`; the output operation applies
the remaining zero-point correction, taking convolution padding into account.
The sparse branch receives `FP16((q_u - zero_point) * a)` and FP16 weights and
accumulates in FP32. The main integer branch accumulates in INT32; their scaled
sum and bias are combined in FP32 before the network feature is stored as FP16.

An optional per-layer `main_output: float16` first applies the dense branch's
zero-point correction and bias in FP32, stores that corrected dense result in
FP16, and then adds the FP32 sparse result. This extra rounding is explicit and
is reproduced by the FP32 fake oracle. It reduces intermediate storage; the
uncorrected large INT32 result is never rounded to FP16 before zero-point
cancellation. `main_output: float32` retains the full floating intermediate.

`fake_fp16` is a conventional fakequant experiment using the same packed codes,
CSR mask and FP16 outlier values, materialized as floating weights. Its cuDNN
convolutions use FP16 operands and FP32 accumulation, and features remain FP16.
Rounding dequantized normal weights to FP16 can differ from integer accumulation
followed by FP32 rescaling. `fake` retains an FP32 numerical oracle, including
the explicitly selected dense-output rounding, for layer-level validation.

The optional Tensor Core sparse implementation derives eight-output-channel
blocks from the union of exact CSR indices. Zero-filled coefficients preserve
the same outliers. These derived blocks do not change fakequant's sparse mask.
The ordinary CSR implementation remains available for comparison.

Export checks both ranked tails of each original smoothed weight tensor.
The metrics workflow also checks integer packing, padding, CSR consistency,
checksums and exact correspondence with the saved calibration state.

## Execution settings

`runtime.json` binds execution flags to the model and config checksums.
`joint_plan.json` binds per-layer kernel choices to the same weights.
They do not change the packed weight format or fakequant sparse structure.

```python
from quantdemoire.profiles import apply_execution_profile
from quantdemoire.precision import configure
from quantdemoire.deploy import load_model
folder = "checkpoints/reproduction/deploy/w4a4"
apply_execution_profile(folder)
configure()
model = load_model(folder, backend="cuda")
```

The helper validates checksums and resolves the plan inside the model directory.
Newly reproduced weights use the architecture-checked templates in
`config/release/`; their runtime status remains `unmeasured_proposal`, without
copied quality or performance claims. The reproduction commands generate fresh
quality, Graph equivalence, speed and memory reports for those weights.

The validated historical packages indexed by `models.json` additionally contain
model cards, provenance, reports and file manifests. They are local safetensors
packages requiring this CUDA loader, not Transformers AutoModel repositories.
Hugging Face links remain unset until the weights are uploaded.
