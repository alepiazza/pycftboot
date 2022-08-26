import unittest
import shutil
import os

from pycftboot import ConformalBlockTable
from pycftboot.serialize import read_table, write_table


class TestCBT(unittest.TestCase):
    def test_CBT(self):
        cbt = ConformalBlockTable(3, 3, 4, 3, 1, odd_spins=True)
        write_table(cbt, "tab.json", form="json")
        os.remove("tab.json")


if __name__ == '__main__':
    unittest.main()
