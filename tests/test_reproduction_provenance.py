"""Reject accidental mixing of calibration recipes and floating boundary weights."""
import json
from pathlib import Path
import pytest
import yaml
from script._internal.reproduce_ptq import configuration,validate_calibration_source
from quantdemoire.deploy import sha256


def test_saved_calibration_export_identity(tmp_path):
    checkpoint=tmp_path/'fp32.pth';checkpoint.write_bytes(b'original floating weights')
    run=tmp_path/'run';run.mkdir()
    config=configuration('paper',4,'full_if_pooled',checkpoint,tmp_path/'train',tmp_path/'test',run,sampling_rate=.01)
    (run/'config.yaml').write_text(yaml.safe_dump(config))
    provenance=dict(config_sha256=sha256(run/'config.yaml'),checkpoint_sha256=sha256(checkpoint))
    (run/'provenance.json').write_text(json.dumps(provenance))
    validate_calibration_source(run,checkpoint,config)
    changed=configuration('paper',4,'full_if_pooled',checkpoint,tmp_path/'train',tmp_path/'test',run,sampling_rate=.05)
    with pytest.raises(ValueError,match='ACT_SAMPLE_RATE'):validate_calibration_source(run,checkpoint,changed)
    checkpoint.write_bytes(b'different floating boundary weights')
    with pytest.raises(ValueError,match='checkpoint differs'):validate_calibration_source(run,checkpoint,config)
    checkpoint.write_bytes(b'original floating weights')
    with (run/'config.yaml').open('a') as f:f.write('\n# changed after calibration\n')
    with pytest.raises(ValueError,match='config was modified'):validate_calibration_source(run,checkpoint,config)
