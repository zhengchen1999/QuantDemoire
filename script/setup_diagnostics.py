"""Install the matching NVIDIA cuobjdump locally for compiled resource reports."""
import hashlib
import io
from pathlib import Path
import tarfile
import urllib.request


def main():
    root=Path('.third_party/cuda-diagnostics-12.1.111');root.mkdir(parents=True,exist_ok=True)
    relative='cuda_cuobjdump/linux-x86_64/cuda_cuobjdump-linux-x86_64-12.1.111-archive.tar.xz'
    expected='1098b58dac2be496bf9a2766c9cdc17c20848a0dd42fc7283aa9c669c0afda52'
    with urllib.request.urlopen('https://developer.download.nvidia.com/compute/cuda/redist/'+relative,timeout=30) as response:data=response.read()
    if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('NVIDIA cuobjdump archive hash mismatch')
    with tarfile.open(fileobj=io.BytesIO(data),mode='r:xz') as archive:
        for member in archive.getmembers():
            if not member.isfile():continue
            parts=Path(member.name).parts[1:]
            if not parts or any(x in ('..','/') for x in parts):raise ValueError('invalid archive member')
            target=root.joinpath(*parts);target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(archive.extractfile(member).read());target.chmod(member.mode&0o777)
    print(root/'bin/cuobjdump')


if __name__=='__main__':main()
