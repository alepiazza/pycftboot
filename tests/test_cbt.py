import unittest

from pycftboot import ConformalBlockTable
from pycftboot.cbt_seed1 import ConformalBlockTableSeed1


class TestCBTSeed1(unittest.TestCase):
    def test_CBTSeed1(self):
        ConformalBlockTableSeed1(2.00001, 2, 2, 2, 2, 0.1, -0.1)

    def test_CBTSeed1_oddspins(self):
        ConformalBlockTableSeed1(3, 1, 1, 3, 0, odd_spins=True)


class TestCBT(unittest.TestCase):

    def test_CBT(self):
        ConformalBlockTable(3, 3, 4, 3, 1, odd_spins=True)

    def test_CBT_oddspins(self):
        ConformalBlockTable(3, 3, 4, 3, 1, odd_spins=True)

    def test_CBT_even_dimension(self):
        ConformalBlockTable(2, 2, 2, 2, 2, 0.1, -0.1)
        ConformalBlockTable(4, 1, 2, 1, 1)
        ConformalBlockTable(6, 1, 2, 1, 1)
