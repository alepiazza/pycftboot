import unittest
import shutil
import os
import subprocess
import filecmp
from symengine.lib.symengine_wrapper import RealMPFR

from pycftboot import SdpbDocker, SdpbBinary, SdpbSingularity
from pycftboot.constants import prec


DIR = 'test_output'


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
        self.volume = DIR
        os.makedirs(self.volume, exist_ok=True)

        if have_binary('docker'):
            self.s = SdpbDocker(volume=self.volume)
        elif have_binary('singularity'):
            self.s = SdpbSingularity(volume=self.volume)
        elif have_binary('sdpb'):
            self.s = SdpbBinary()
        else:
            raise RuntimeError("Nor docker or sdpb found")

        self.input_xml = f"{DIR}/test.xml"
        self.s.set_option("sdpDir", f"{DIR}/test_pvm2sdp")
        self.s.set_option("outDir", f"{DIR}/test_sdpb")
        self.s.set_option("procsPerNode", 1)

    def tearDown(self):
        shutil.rmtree(self.volume)

    @unittest.skipUnless(have_binary('docker'), "No docker")
    def test_sdpb_docker_run_command(self):
        command = 'echo "running in docker"'.split()
        out_docker = self.s.run_command(command)
        out_shell = subprocess.run(command, capture_output=True, check=True, text=True)

        self.assertEqual(out_docker.__dict__, out_shell.__dict__)

        self.s.run_command(f"touch '{DIR}/test'")
        self.assertTrue(os.path.isfile(f'{DIR}/test'))

        self.s.run_command(f"rm '{DIR}/test'")
        self.assertFalse(os.path.isfile(f'{DIR}/test'))

    @unittest.skipUnless(have_binary('singularity'), "No singularity")
    def test_sdpb_singularity_run_command(self):
        command = 'echo "running in singularity"'.split()
        out_docker = self.s.run_command(command)
        out_shell = subprocess.run(command, capture_output=True, check=True, text=True)

        self.assertEqual(out_docker.__dict__, out_shell.__dict__)

        self.s.run_command(f"touch '{DIR}/test'")
        self.assertTrue(os.path.isfile(f'{DIR}/test'))

        self.s.run_command(f"rm '{DIR}/test'")
        self.assertFalse(os.path.isfile(f'{DIR}/test'))

    def test_sdpb_sdpb(self):
        shutil.copy('tests/input/test.xml', self.input_xml)

        self.s.pvm2sdp_run(self.input_xml, self.s.get_option("sdpDir"))
        self.assertTrue(filecmp.cmp(f'{DIR}/test_pvm2sdp', 'tests/check_output/test_pvm2sdp'))

        self.s.run()
        self.assertTrue(check_directory_equal(f'{DIR}/test_pvm2sdp.ck', 'tests/check_output/test_pvm2sdp.ck'))
        self.assertTrue(check_directory_equal(f'{DIR}/test_sdpb', 'tests/check_output/test_sdpb'))

    def test_sdpb_read_sdpb_output(self):
        shutil.copy('tests/input/test.xml', self.input_xml)
        self.s.pvm2sdp_run(self.input_xml, self.s.get_option("sdpDir"))
        self.s.run()

        output = self.s.read_output(self.s.get_option("outDir"))

        expected = {
            'terminateReason': "found primal-dual optimal solution",
            'primalObjective': RealMPFR('1.84026576313204924668804017173055420056358532030282556465761906133430166726537336826049865612094019021116018862947817214304719196101000427864203352107112262936760692514062283196788975004021011672191701829389064342827013315072579637164495584368311174278794769234879785974306722994648049078923989863964029855554', prec),
            'dualObjective': RealMPFR('1.840265763132049246688040171729243880847849070203079579264064559727569678203895517291163568652036837213248476950467408121928881474796294697810566545438468725106599627498797568557227808458637633937332359918960117371256786874466681129552618022085670010433051735401216868782118286089664817126001406913220905555463', prec),
            'dualityGap': RealMPFR('3.560137187756362701499990596353350507234427431091688312938856070418949746208535223856956948980295330518142243679262888338136646138921677724994648036010040775124946520398537720149932531944731039705895343895277575266502476933002060070003605905232816428900288987873169918364268606634162390953818492501919215781071e-31', prec),
            'primalError': RealMPFR('2.844993327070838830130404255408100652278454061145271401392370018205740571241281944918206767249616473586629641188871834355246588893411940440767801421709599557240323086373643840314772276916295023932018238704214514716024246720832734528902639021973916313223446141658051582294251221841484722792097493738006415876931e-309', prec),
            'dualError': RealMPFR('7.771626719301027630788410323462754295301666615759809501165134335332783289696312778570047503640947867075146200845575328888246164417765847795206837259028027412504524836908901139782658342480083994890204046317708429917656302657254928941332784668143495568835298449382300596070356830709351456934695701910135947007622e-305', prec),
            'Solver runtime': float(0),
            'y': [RealMPFR('-1.84026576313204924668804017172924388084784907020307957926406455972756967820389551729116356865203683721324847695046740812192888147479629469781056654543846872510659962749879756855722780845863763393733235991896011737125678687446668112955261802208567001043305173540121686878211828608966481712600140691322090555546', prec)]
        }
        expected = {k: str(v) for k, v in expected.items()}
        output = {k: str(v) for k, v in output.items()}

        self.assertEqual(output, expected)


if __name__ == '__main__':
    unittest.main()
