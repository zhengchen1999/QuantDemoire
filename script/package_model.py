"""Create a self-contained, local Hugging Face model folder. Never uploads."""
import argparse
import json
import shutil
from pathlib import Path
from quantdemoire.deploy import sha256


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--quality',help='optional measured quality JSON for this exact checkpoint')
    a=p.parse_args();source=Path(a.model);out=Path(a.output)
    if source.resolve()==out.resolve():p.error('output must differ from source')
    if out.exists() and any(out.iterdir()):p.error('use an empty output folder to avoid stale weights, provenance or runtime profiles')
    config=json.loads((source/'config.json').read_text())
    if config['weights_sha256']!=sha256(source/'model.safetensors'):raise ValueError('weight checksum mismatch')
    quality=None
    if a.quality:
        quality=json.loads(Path(a.quality).read_text())
        if quality['model_sha256']!=config['weights_sha256']:raise ValueError('quality result belongs to different weights')
        if quality.get('config_sha256')!=sha256(source/'config.json'):
            raise ValueError('quality result belongs to a different deployment configuration')
        if not quality.get('complete'):
            raise ValueError('only a completed evaluation can be attached to a model package')
        if quality.get('environment',{}).get('main_half_override') is not None:
            raise ValueError('evaluate the saved configuration without a main-output environment override before packaging')
    out.mkdir(parents=True,exist_ok=True)
    for name in ['config.json','model.safetensors','range_search.json','selection.json','calibration.json','training.json','runtime.json','joint_plan.json',
                 'provenance.json','calibration_history.json','config.yaml','reproduction.json']:
        if (source/name).is_file():shutil.copyfile(source/name,out/name)
    if quality:
        (out/'evaluation.json').write_text(json.dumps(quality,indent=2)+'\n')
    bits=config['bits']
    profile_setup=('from quantdemoire.profiles import apply_execution_profile\n'
                   f'apply_execution_profile("{out.name}")\n'
                   if (out/'runtime.json').is_file() else '')
    training=config.get('polish_method','calibration and activation-range selection; no additional weight training')
    if (source/'reproduction.json').is_file():
        reproduction=json.loads((source/'reproduction.json').read_text())
        training=(f"raw-checkpoint PTQ, {reproduction['recipe']} tail setting; activation bounds only. "
                  'The exact sampling assumptions and frozen-weight checks are in reproduction.json and calibration_history.json')
    card=f'''---
library_name: quantdemoire
pipeline_tag: image-to-image
tags:
- image-demoireing
- quantization
- int{bits}
- esdnet
---

# ESDNet UHDM W{bits}A{bits} + FP16 outliers

Local deployment artifact for [QuantDemoire](https://github.com/zhengchen1999/QuantDemoire).
Format: `quantdemoire-v1`; tensor checksum: `{config['weights_sha256']}`.

157 convolutions use packed signed INT{bits} weights and INT{bits} Tensor Core
convolution with INT32 accumulation. The first and last convolutions use FP16
operands with FP32 accumulation. Sparse weights and sparse activation operands
are FP16, with FP32 accumulation. Network features are FP16.
Each quantized layer retains at least 0.25% of each original smoothed weight tail.
Fake quantization and CUDA inference load identical integer codes and sparse
indices from this folder. A CUDA toolkit and the project's kernel are required;
this is not a Transformers `AutoModel` checkpoint.

The saved `main_output` setting specifies whether the corrected integer-branch
intermediate is rounded to FP16 before the sparse contribution is added.
Training provenance: {training}.

```python
from quantdemoire.precision import configure
from quantdemoire.deploy import load_model
{profile_setup}\
configure()
model = load_model("{out.name}", backend="cuda")
```

Inputs must be CUDA FP16 RGB tensors, padded to multiples of 32; the first of
the three outputs is the final restored image. See `script/evaluate.py` for
padding, output cropping and metrics.

This artifact has not been uploaded. Hub repository ID and public download URL
will be added only after publication. No claim of a measured speedup follows
from the low-bit format alone. Performance and quality depend on the saved
evaluation and benchmark reports. The image hashes and count in evaluation.json
define the evaluated split; a development subset must not be described as the
complete UHDM test set.
'''
    if quality:
        card+=f"\nMeasured {quality['backend']} quality on {quality['count']} images: `{json.dumps(quality['mean'])}`. See `evaluation.json` for image hashes and protocol.\n"
    if (out/'runtime.json').is_file():
        from quantdemoire.profiles import profile_environment
        profile_environment(out)
        profile=json.loads((out/'runtime.json').read_text())
        description=('an unmeasured execution proposal' if profile.get('status')=='unmeasured_proposal'
                     else 'a verified execution profile')
        card+=f'''\n<!-- quantdemoire-runtime-profile -->
This folder includes {description} in `runtime.json`. Apply it
before loading the model; existing explicit environment overrides are preserved:

```python
from quantdemoire.profiles import apply_execution_profile
apply_execution_profile("{out.name}")
```

Only measured profiles carry performance evidence for the recorded GPU, software,
and input. An unmeasured proposal requires fresh validation and timing.
<!-- /quantdemoire-runtime-profile -->
'''
    (out/'README.md').write_text(card)
    manifest={p.name:sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='manifest.json'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
