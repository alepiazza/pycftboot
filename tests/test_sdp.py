import unittest
import shutil
import os

from pycftboot import ConformalBlockTable, ConvolvedBlockTable, SDP


def have_binary(bin_name):
    bin_in_path = shutil.which(bin_name)
    return bin_in_path is not None and os.path.isfile(bin_in_path)


class TestSDP(unittest.TestCase):
    def setUp(self):
        table1 = ConformalBlockTable(3, 10, 10, 2, 4)
        table2 = ConvolvedBlockTable(table1)

        if have_binary('docker'):
            sdpb_mode = 'docker'
            sdpb_kwargs = {'volume': 'test_output'}
        elif have_binary('sdpb'):
            sdpb_mode = 'binary'
            sdpb_kwargs = {}
            os.makedirs('test_output', exist_ok=True)
        else:
            raise RuntimeError("Nor docker or sdpb found")

        self.sdp = SDP(0.518, table2, sdpb_mode=sdpb_mode, sdpb_kwargs=sdpb_kwargs)
        self.sdp.sdpb.set_option("procsPerNode", 1)

    def tearDown(self):
        shutil.rmtree("test_output")

    def test_sdp_bisect(self):
        tol = 0.01
        result = self.sdp.bisect(0.7, 1.7, tol, 0, name="test_output/test_bisect")

        self.assertTrue(abs(result - 1.4170708508484153) < tol)

        self.sdp.set_bound(0, 3)
        self.sdp.add_point(0, result)
        allowed = self.sdp.iterate(name="test_output/test_add_point")

        self.assertTrue(allowed)

    def test_sdp_openmax(self):
        self.sdp.opemax(1.2, 0, name="test_output/test_openmax")

    def test_sdp_extremal(self):
        func = self.sdp.solution_functional(1.2, [0, 0], name="test_output/test_functional")
        dims = self.sdp.extremal_dimensions(func, [0, 0], 0.001, tmp_file="test_output/extremal_dimensions.tmp")
        self.sdp.extremal_coefficients(dims, [0, 0], tmp_name="test_output/tmp")


if __name__ == '__main__':
    unittest.main()
