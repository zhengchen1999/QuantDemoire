"""Select exact, hash-verified evaluation pairs from a recorded data split."""
import json
from pathlib import Path
from .deploy import sha256


def selected_pairs(root, manifest=None, subset='test', limit=None):
    root=Path(root).resolve()
    if limit is not None and limit<1:
        raise ValueError('evaluation limit must be positive')
    if manifest is None:
        paths=sorted(root.rglob('*_gt.jpg'))
    else:
        payload=json.loads(Path(manifest).read_text())
        if subset not in payload or not isinstance(payload[subset],list):
            raise ValueError(f'manifest has no pair list: {subset}')
        paths=[]
        for record in payload[subset]:
            relative=Path(record['image'])
            path=(root/relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(root) or not path.name.endswith('_gt.jpg'):
                raise ValueError('invalid manifest image path')
            source=path.with_name(path.name.replace('_gt.jpg','_moire.jpg'))
            if sha256(path)!=record['gt_sha256'] or sha256(source)!=record['input_sha256']:
                raise ValueError(f'manifest image contents differ: {relative}')
            paths.append(path)
        if len(set(paths))!=len(paths):
            raise ValueError('duplicate evaluation pair in manifest')
    paths=paths[:limit]
    if not paths:
        raise ValueError('empty evaluation dataset')
    return paths
