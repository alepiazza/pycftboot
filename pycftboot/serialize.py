from .conformal_block_table import ConformalBlockTable
from .compat_json import json_write, json_read
from .compat_juliboots import juliboots_write, juliboots_read
from .compat_scalar_blocks import scalar_blocks_write, scalar_blocks_read


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
          "juliboots". Any other value will be ignored. Defaults to "json".
    """
    if isinstance(block_table, ConformalBlockTable):
        if form == "json":
            json_write(block_table, name)
        elif form == "juliboots":
            juliboots_write(block_table, name)
        elif form == "scalar_blocks":
            scalar_blocks_write(block_table, name)
        else:
            raise ValueError("form parameter must be either json, scalar_blocks or juliboots")
    else:
        raise TypeError(f"{block_table} is not a ConformalBlockTable")
