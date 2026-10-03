"""Packaging failures must not leave a newly written runtime profile behind."""
import sys

import pytest

from script import attach_runtime_profile


@pytest.mark.parametrize('card', [None, '<!-- quantdemoire-runtime-profile -->',
                                 '<!-- /quantdemoire-runtime-profile -->'])
def test_invalid_model_card_preserves_runtime(tmp_path, monkeypatch, card):
    model = tmp_path / 'model'
    model.mkdir()
    runtime = model / 'runtime.json'
    original = b'{"status": "unmeasured_proposal"}\n'
    runtime.write_bytes(original)
    if card is not None:
        (model / 'README.md').write_text(card)
    benchmark = tmp_path / 'benchmark.json'
    benchmark.write_text('{}')
    monkeypatch.setattr(sys, 'argv', ['attach_runtime_profile.py', '--model', str(model),
        '--benchmark', str(benchmark), '--variant', 'w4a4', '--validation', 'unused.json'])
    with pytest.raises(ValueError, match='model card'):
        attach_runtime_profile.main()
    assert runtime.read_bytes() == original
    assert not (model / 'manifest.json').exists()
