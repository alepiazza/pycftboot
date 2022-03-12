import unittest
import shutil
import os
import subprocess
import filecmp

from pycftboot import SdpbDocker, SdpbBinary


def have_binary(bin_name):
    bin_in_path = shutil.which(bin_name)
    return bin_in_path is not None and os.path.isfile(bin_in_path)


def check_directory_equal(dir1, dir2):
    """https://stackoverflow.com/questions/4187564/recursively-compare-two-directories-to-ensure-they-have-the-same-files-and-subdi
    """

    dirs_cmp = filecmp.dircmp(dir1, dir2)
    if len(dirs_cmp.left_only) > 0 or len(dirs_cmp.right_only) > 0 or \
            len(dirs_cmp.funny_files) > 0:
        return False
    (_, mismatch, errors) = filecmp.cmpfiles(
        dir1, dir2, dirs_cmp.common_files, shallow=False)
    if len(mismatch) > 0 or len(errors) > 0:
        return False
    for common_dir in dirs_cmp.common_dirs:
        new_dir1 = os.path.join(dir1, common_dir)
        new_dir2 = os.path.join(dir2, common_dir)
        if not check_directory_equal(new_dir1, new_dir2):
            return False
    return True


@unittest.skipUnless(have_binary('docker'), "No docker")
class TestSdpbDocker(unittest.TestCase):
    def setUp(self):
        self.test_volume = 'test_output'
        os.makedirs(self.test_volume, exist_ok=True)

        self.s = SdpbDocker(volume=self.test_volume)

    def tearDown(self):
        shutil.rmtree(self.test_volume)

    def test_sdpb_docker_run_command(self):
        command = 'echo "running in docker"'.split()
        out_docker = self.s.run_command(command)
        out_shell = subprocess.run(command, capture_output=True, check=True, text=True)

        self.assertEqual(out_docker.__dict__, out_shell.__dict__)

        self.s.run_command("touch 'test'")
        self.assertTrue(os.path.isfile(f'{self.test_volume}/test'))

        self.s.run_command("rm 'test'")
        self.assertFalse(os.path.isfile(f'{self.test_volume}/test'))

    def test_sdpb_docker_sdpb(self):
        shutil.copy('tests/input/test.xml', self.test_volume)

        self.s.pvm2sdp_run("660 test.xml test_pvm2sdp".split())
        self.assertTrue(filecmp.cmp(f'{self.test_volume}/test_pvm2sdp', 'tests/check_output/test_pvm2sdp'))

        self.s.run("--procsPerNode 1 -s test_pvm2sdp -o test_sdpb".split())
        self.assertTrue(check_directory_equal(f'{self.test_volume}/test_pvm2sdp.ck', 'tests/check_output/test_pvm2sdp.ck'))
        self.assertTrue(check_directory_equal(f'{self.test_volume}/test_sdpb', 'tests/check_output/test_sdpb'))


if __name__ == '__main__':
    unittest.main()
