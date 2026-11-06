"""Withdrawal checks; no custom DLL is loaded."""
import unittest
import sys
from unittest.mock import patch
from experiments.g0_realtime_stage4_20261105.runtime import Runtime as V4
from .runtime import Runtime
from .native import Native

class WithdrawalTests(unittest.TestCase):
    def test_runtime_is_exact_v4_without_custom_dll(self):
        self.assertIs(Runtime, V4)
        with patch('ctypes.CDLL', side_effect=AssertionError('Custom DLL load attempted')):
            runtime = Runtime()
            self.assertFalse(hasattr(runtime, 'backend'))
            runtime.reset()

    def test_native_entry_disabled_before_loading(self):
        with patch('ctypes.CDLL', side_effect=AssertionError('Custom DLL load attempted')):
            with self.assertRaisesRegex(RuntimeError, 'withdrawn'):
                Native(None)

    def test_compatibility_cli_records_the_v4_arm(self):
        from . import benchmark
        for options in (['--arm', 'v5-serial'], ['--arm=v5-serial']):
            with patch.object(sys, 'argv', ['benchmark', *options, '--diagnostic']):
                with patch.object(benchmark, 'v4_main') as entry:
                    benchmark.main()
                    entry.assert_called_once_with()
                    self.assertNotIn('v5-serial', sys.argv)
                    self.assertTrue('v4-serial' in sys.argv or '--arm=v4-serial' in sys.argv)

if __name__ == '__main__':
    unittest.main()
