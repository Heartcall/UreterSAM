"""Retired native adapter; historical code is preserved in the run archives."""
class Native:
    def __init__(self, *args, **kwargs):
        raise RuntimeError('The Windows x64 DLL experiment has been withdrawn. Use the v4 NumPy Runtime.')
