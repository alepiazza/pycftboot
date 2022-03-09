import unittest
import json
import os

from pycftboot import ConformalBlockTable, read_table, write_table
from pycftboot.compat_json import json_write


class TestSerializer(unittest.TestCase):
    def test_write_table_error(self):
        with self.assertRaises(TypeError):
            write_table('', '')


class TestJsonSerializer(unittest.TestCase):
    def test_read_write_table_json(self):
        cbt = ConformalBlockTable(3, 1, 1, 1, 1)
        write_table(cbt, 'tests/test_write_table.json', 'json')
        read_table('tests/test_write_table.json')

        os.remove('tests/test_write_table.json')

    def test_json_errors(self):
        with self.assertRaises(TypeError):
            json_write('', '')

        with self.assertRaises(TypeError):
            with open('not_cbt', 'w') as f:
                json.dump({'not': 'conformal block table'}, f)
            read_table('not_cbt')

        os.remove('not_cbt')
