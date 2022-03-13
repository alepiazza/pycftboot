import unittest
import shutil
import os

from pycftboot import ConformalBlockTable, ConvolvedBlockTable, SDP


def have_binary(bin_name):
    bin_in_path = shutil.which(bin_name)
    return bin_in_path is not None and os.path.isfile(bin_in_path)


class TestCBTSeed1(unittest.TestCase):
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

    def test_sdp_bisect(self):
        self.sdp.sdpb.set_option("procsPerNode", 1)
        self.sdp.sdpb.set_option("sdpDir", "test_output/test")
        self.sdp.bisect(0.7, 1.7, 0.05, 0)


if __name__ == '__main__':
    unittest.main()
