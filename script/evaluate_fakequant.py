"""FP16 fakequant quality; --fp32 selects the slower FP32 numerical oracle."""
import sys
from evaluate import main

if __name__=='__main__':
    backend='fake_fp16'
    if '--fp32' in sys.argv:
        sys.argv.remove('--fp32');backend='fake'
        if not any(arg=='--memory-format' or arg.startswith('--memory-format=') for arg in sys.argv):
            sys.argv.extend(['--memory-format','contiguous'])
    sys.argv.extend(['--backend',backend])
    main()
