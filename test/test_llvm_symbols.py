import os
import subprocess
import sys
import torch
import unittest


class TestLLVMSymbols(unittest.TestCase):
    def _getVMMaps(self):
        if sys.platform == "linux":
            with open(f"/proc/{os.getpid()}/maps") as f:
                return f.readlines()
        if sys.platform == "darwin":
            with subprocess.Popen(
                ["vmmap", "-wide", str(os.getpid())],
                stdout=subprocess.PIPE,
                encoding="utf-8",
            ) as proc:
                return proc.stdout.readlines()
        self.skipTest("Can't check leaking symbols on this platform")

    def _getLibTorchPath(self):
        for mapping in self._getVMMaps():
            if "libtorch_cpu" in mapping:
                libtorch_path = mapping.split(" ")[-1].rstrip()
        return libtorch_path

    def assertLLVMSymbolLocal(self, symtype, name):
        if symtype == "T":
            self.assertNotRegex(name, "^__?ZNK?4llvm")

    def test_all_local(self):
        libtorch_path = self._getLibTorchPath()
        with subprocess.Popen(
            ["nm", libtorch_path], stdout=subprocess.PIPE, encoding="utf-8"
        ) as proc:
            for line in proc.stdout:
                parts = line.split(" ")
                if len(parts) == 3:
                    addr, symtype, name = parts
                    self.assertLLVMSymbolLocal(symtype, name)


if __name__ == "__main__":
    unittest.main()
