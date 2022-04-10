import unittest
import itertools

from pycftboot import ConformalBlockTable, ConvolvedBlockTable
from pycftboot.cbt_seed1 import ConformalBlockTableSeed1
from pycftboot.cbt_seed2 import ConformalBlockTableSeed2


class TestCBTSeed1(unittest.TestCase):
    def test_CBTSeed1(self):
        """Test ConformalBlockTableSeed1 also for m_max > 3 and n_max != 0
        """
        ConformalBlockTableSeed1(2.00001, 1, 1, 6, 2, 0.1, -0.1)


class TestCBTSeed2(unittest.TestCase):
    def test_CBTSeed2(self):
        """Test ConformalBlockTableSeed1 also for m_max > 3, n_max != 0 and
        not even dimension

        """
        ConformalBlockTableSeed2(3, 4, 1, 4, None)

    def test_CBTSeed2_error(self):
        """Test ValueError if n_max is not None
        """
        with self.assertRaises(ValueError):
            ConformalBlockTableSeed2(2.00001, 2, 2, 2, 2, 0.1, -0.1)


class TestCBT(unittest.TestCase):
    def test_CBT(self):
        ConformalBlockTable(3, 3, 4, 3, 1, odd_spins=True)

    def test_CBT_oddspins(self):
        ConformalBlockTable(3, 3, 4, 3, 1, odd_spins=True)

    def test_CBT_even_dimension(self):
        ConformalBlockTable(2, 2, 2, 2, 2, 0.1, -0.1)
        ConformalBlockTable(6, 1, 2, 1, 1, odd_spins=True)

    def test_CBT_convert_table(self):
        cbt1 = ConformalBlockTable(3, 1, 1, 1, 1, odd_spins=True)
        cbt2 = ConformalBlockTable(3, 2, 1, 2, 2, odd_spins=True)

        with self.assertRaises(TypeError):
            cbt1.convert_table('')

        cbt1.convert_table(cbt2)


class TestConvolvedBlockTable(unittest.TestCase):
    def test_ConvBT(self):
        for conformal_spin, convolved_spin in itertools.product((True, False), repeat=2):
            cbt = ConformalBlockTable(3, 2, 2, 1, 1, odd_spins=conformal_spin)
            ConvolvedBlockTable(cbt, odd_spins=convolved_spin)

        cbt = ConformalBlockTable(3, 2, 2, 1, 1)
        ConvolvedBlockTable(cbt, spins=[1, 2], symmetric=True)

        with self.assertRaises(TypeError):
            ConvolvedBlockTable('')


if __name__ == '__main__':
    unittest.main()
