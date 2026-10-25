from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class NativeCopyTests(unittest.TestCase):
    def test_strided_partial_copy_matches_scalar_and_preserves_padding(self):
        compiler=shutil.which('clang++') or shutil.which('g++')
        if compiler is None:self.skipTest('C++ compiler unavailable')
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            binary=Path(tmp)/'tensor_copy_test'
            subprocess.run([compiler,'-std=c++17','-O2','-Wall','-Wextra','-I',str(root/'native'),
                            str(root/'tests/tensor_copy_test.cc'),'-o',str(binary)],check=True,capture_output=True)
            result=subprocess.run([str(binary)],check=True,capture_output=True,text=True)
            self.assertIn('PASS 2000',result.stdout)


if __name__=='__main__':unittest.main()
