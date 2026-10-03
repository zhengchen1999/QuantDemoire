import json
import pytest
from quantdemoire.deploy import sha256
from quantdemoire.evaluation_subset import selected_pairs


def test_holdout_selection_checks_content_and_order(tmp_path):
    rows=[]
    for directory in ('pair_02','pair_00'):
        root=tmp_path/directory;root.mkdir()
        gt=root/'0000_gt.jpg';gt.write_bytes(directory.encode())
        source=root/'0000_moire.jpg';source.write_bytes(b'input'+directory.encode())
        rows.append(dict(image=str(gt.relative_to(tmp_path)),gt_sha256=sha256(gt),input_sha256=sha256(source)))
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(heldout_training=rows)))
    paths=selected_pairs(tmp_path,manifest,'heldout_training')
    assert [str(p.relative_to(tmp_path)) for p in paths]==[r['image'] for r in rows]
    source.write_bytes(b'changed')
    with pytest.raises(ValueError,match='contents differ'):selected_pairs(tmp_path,manifest,'heldout_training')


def test_manifest_rejects_escape_and_duplicates(tmp_path):
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(test=[dict(image='../escape_gt.jpg')])))
    with pytest.raises(ValueError,match='invalid manifest'):selected_pairs(tmp_path,manifest)
    gt=tmp_path/'0000_gt.jpg';gt.write_bytes(b'gt')
    src=tmp_path/'0000_moire.jpg';src.write_bytes(b'source')
    row=dict(image=gt.name,gt_sha256=sha256(gt),input_sha256=sha256(src))
    manifest.write_text(json.dumps(dict(test=[row,row])))
    with pytest.raises(ValueError,match='duplicate'):selected_pairs(tmp_path,manifest)
