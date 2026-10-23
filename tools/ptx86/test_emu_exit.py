"""CPU control-flow check: a failed emulation check must fail the command."""
import ast
import argparse
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch


class EmulationExitTests(unittest.TestCase):
    def test_main_propagates_each_failed_check(self):
        # Execute the actual main function without loading a driver or creating a GPU context.
        tree = ast.parse(Path(__file__).with_name("test_emu.py").read_text(encoding="utf-8"))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        code = compile(ast.Module(body=[function], type_ignores=[]), "test_emu.py", "exec")
        checks = ("test_decode", "test_encode", "test_mma", "test_fq", "test_mma_fused")
        for failed in (None, *checks):
            with self.subTest(failed=failed):
                env = {"argparse": argparse, "ACC": {},
                       "load": Mock(return_value=object()), "rewrite": Mock(return_value=("", {})),
                       "PTX": "", "print": Mock(), **{name: Mock(return_value=name != failed) for name in checks}}
                exec(code, env)
                with patch.object(sys, "argv", ["test_emu"]):
                    self.assertEqual(env["main"](), 0 if failed is None else 1)
                for name in checks:
                    env[name].assert_called_once()


if __name__ == "__main__":
    unittest.main()
