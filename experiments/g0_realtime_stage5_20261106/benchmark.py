"""Compatibility CLI: the former v5 arm now explicitly executes v4 NumPy."""
import sys
from experiments.g0_realtime_stage4_20261105.benchmark import main as v4_main

def main():
    args = sys.argv[1:]
    for index, value in enumerate(args):
        if value == '--arm' and index+1 < len(args) and args[index+1] == 'v5-serial':
            args[index+1] = 'v4-serial'
        elif value == '--arm=v5-serial':
            args[index] = '--arm=v4-serial'
    print('The DLL experiment is withdrawn; this entry uses the v4 NumPy benchmark.', flush=True)
    sys.argv = [sys.argv[0], *args]
    v4_main()

if __name__ == '__main__':
    main()
