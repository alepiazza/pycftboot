import unittest
import shutil
import os
import subprocess
from pprint import pprint
import filecmp
from symengine.lib.symengine_wrapper import RealMPFR

from pycftboot import SdpbDocker, SdpbBinary
from pycftboot.constants import prec


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


class TestSdpb(unittest.TestCase):
    def setUp(self):
        self.volume = 'test_output'
        os.makedirs(self.volume, exist_ok=True)

        if have_binary('docker'):
            self.s = SdpbDocker(volume=self.volume)
        elif have_binary('sdpb'):
            self.s = SdpbBinary()
        else:
            raise RuntimeError("Nor docker or sdpb found")

    def tearDown(self):
        shutil.rmtree(self.volume)

    @unittest.skipUnless(have_binary('docker'), "No docker")
    def test_sdpb_docker_run_command(self):
        command = 'echo "running in docker"'.split()
        out_docker = self.s.run_command(command)
        out_shell = subprocess.run(command, capture_output=True, check=True, text=True)

        self.assertEqual(out_docker.__dict__, out_shell.__dict__)

        self.s.run_command("touch 'test'")
        self.assertTrue(os.path.isfile(f'{self.volume}/test'))

        self.s.run_command("rm 'test'")
        self.assertFalse(os.path.isfile(f'{self.volume}/test'))

    def test_sdpb_sdpb(self):
        shutil.copy('tests/input/test.xml', self.volume)

        self.s.pvm2sdp_run("660 test.xml test_pvm2sdp".split())
        self.assertTrue(filecmp.cmp(f'{self.volume}/test_pvm2sdp', 'tests/check_output/test_pvm2sdp'))

        self.s.run("--procsPerNode 1 -s test_pvm2sdp -o test_sdpb".split())
        self.assertTrue(check_directory_equal(f'{self.volume}/test_pvm2sdp.ck', 'tests/check_output/test_pvm2sdp.ck'))
        self.assertTrue(check_directory_equal(f'{self.volume}/test_sdpb', 'tests/check_output/test_sdpb'))

    def test_sdpb_read_sdpb_output(self):
        expected = {
            'terminateReason': "found primal-dual optimal solution",
            'primalObjective': RealMPFR('1.84026576313204924668804017173055420056358532030282556465761906133430166726537336826049865612094019021116018862947817214304719196101000427864203352107112262936760692514062283196788975004021011672107', prec),
            'dualObjective': RealMPFR('1.84026576313204924668804017172924388084784907020307957926406455972756967820389551729116356865203683721324847695046740812192888147479629469781056654543846872510659962749879756855722780845863763393790', prec),
            'dualityGap': RealMPFR('3.56013718775636270149999059635335050723442743109168831293885607041894974620853522385695694898029533051814224367926288833813664613892167772499464803601004077512494651654320924529335252052851986330126e-31', prec),
            'primalError': RealMPFR('3.02599720266600806524000028915989450062982004925818215153860399689135685232954868045195617059093162777053090078176088450807518064138449768061426525659372084875033647651339829218635516934714536708774e-213', prec),
            'dualError': RealMPFR('4.46281245187788768570163247269790454899928515696183592865699887013584285017886599785053051433102982386812845275899249791404197108679379367457100621059380942879597102272199695686175968991580692060326e-209', prec),
            'Solver runtime': float(0),
            'y': [RealMPFR('-1.84026576313204924668804017172924388084784907020307957926406455972756967820389551729116356865203683721324847695046740812192888147479629469781056654543846872510659962749879756855722780845863763393790', prec)]
        }

        shutil.copy('tests/input/test.xml', self.volume)
        self.s.pvm2sdp_run(f"{prec} test.xml test_pvm2sdp".split())
        self.s.run(f"--procsPerNode 1 -s test_pvm2sdp --precision {prec}".split())

        output = self.s.read_output('test_output/test_pvm2sdp')

        expected = {k: str(v) for k, v in expected.items()}
        output = {k: str(v) for k, v in output.items()}

        self.assertEqual(output, expected)


if __name__ == '__main__':
    unittest.main()
