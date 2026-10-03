import sys
from evaluate import main

if __name__=='__main__':
    sys.argv.extend(['--backend','cuda'])
    main()
