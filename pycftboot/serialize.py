import re
import os

from .conformal_block_table import ConformalBlockTable
from .compat_json import json_write, json_read
from .compat_juliboots import juliboots_write, juliboots_read
from .compat_scalar_blocks import scalar_blocks_write, scalar_blocks_read


def dump_table_contents(block_table, name):
    """
    This is called by `ConformalBlockTable` and `ConformalBlockTableSeed`. It
    writes executable Python code to a file designed to recreate the full set of
    the table's attributes as quickly as possible.
    """
    dump_file = open(name, 'w')

    dump_file.write("self.dim = " + block_table.dim.__str__() + "\n")
    dump_file.write("self.k_max = " + block_table.k_max.__str__() + "\n")
    dump_file.write("self.l_max = " + block_table.l_max.__str__() + "\n")
    dump_file.write("self.m_max = " + block_table.m_max.__str__() + "\n")
    dump_file.write("self.n_max = " + block_table.n_max.__str__() + "\n")
    dump_file.write("self.delta_12 = " + block_table.delta_12.__str__() + "\n")
    dump_file.write("self.delta_34 = " + block_table.delta_34.__str__() + "\n")
    dump_file.write("self.odd_spins = " + block_table.odd_spins.__str__() + "\n")
    dump_file.write("self.m_order = " + block_table.m_order.__str__() + "\n")
    dump_file.write("self.n_order = " + block_table.n_order.__str__() + "\n")
    dump_file.write("self.table = []\n")

    for l in range(0, len(block_table.table)):
        dump_file.write("derivatives = []\n")
        for i in range(0, len(block_table.table[0].vector)):
            poly_string = block_table.table[l].vector[i].__str__()
            poly_string = re.sub("([0-9]+\.[0-9]+e?-?[0-9]+)", r"RealMPFR('\1', prec)", poly_string)
            dump_file.write("derivatives.append(" + poly_string + ")\n")
        dump_file.write("self.table.append(PolynomialVector(derivatives, " + block_table.table[l].label.__str__() + ", " + block_table.table[l].poles.__str__() + "))\n")

    dump_file.close()


def read_table(name, form="json"):
    if form == "json":
        return json_read(name)
    elif form == "scalar_blocks":
        return scalar_blocks_read(name)
    elif form == "juliboots":
        return juliboots_read(name)
    else:
        raise ValueError("form parameter must be either json, scalar_blocks or juliboots")


def write_table(block_table, name, form="json"):
    """
    Saves a table of conformal block derivatives to a file. Unless overridden,
    the file is valid Python code which manually populates the entries of
    `table` when executed.

    Parameters
    ----------
    name: The path to use for output.
    form: [Optional] A string indicating that the file should be saved in
          another program's format if it is equal to "scalar_blocks" or
          "juliboots". Any other value will be ignored. Defaults to `None`.
    """
    if isinstance(block_table, ConformalBlockTable):
        if form == "json":
            # dump_table_contents(block_table, name)
            json_write(block_table, name)
        elif form == "juliboots":
            juliboots_write(block_table, name)
        elif form == "scalar_blocks":
            scalar_blocks_write(block_table, name)
        else:
            raise ValueError("form parameter must be either json, scalar_blocks or juliboots")
    else:
        raise TypeError(f"{block_table} is not a ConformalBlockTable")
