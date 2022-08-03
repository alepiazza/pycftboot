import json
import os
from symengine.lib.symengine_wrapper import Add, RealMPFR, ComplexInfinity, Mul

from .polynomial_vector import PolynomialVector
from .conformal_block_table import ConformalBlockTable
from .convolved_block_table import ConvolvedBlockTable
from .constants import prec


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ConformalBlockTable):
            return {
                "_type": "pycftboot.ConformalBlockTable",
                "_value": obj.__dict__
            }
        if isinstance(obj, ConvolvedBlockTable):
            return {
                "_type": "pycftboot.ConvolvedBlockTable",
                "_value": obj.__dict__
            }
        elif isinstance(obj, PolynomialVector):
            return {
                "_type": "pycftboot.PolynomialVector",
                "_value": obj.__dict__
            }
        elif isinstance(obj, (RealMPFR, ComplexInfinity, Add)):
            return {
                "_type": "symengine.Add",
                "_value": obj.__str__()
            }
        elif isinstance(obj, Mul):
            return {
                "_type": "symengine.Mul",
                "_value": obj.__str__()
            }
        else:
            return super().default(obj)


class CustomDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        if "_type" not in obj:
            return obj

        _type = obj["_type"]
        if _type == "pycftboot.ConformalBlockTable":
            vals = obj["_value"]
            cbt = ConformalBlockTable(
                dim=vals["dim"], k_max=vals["k_max"], l_max=vals["l_max"], m_max=vals["m_max"], n_max=vals["n_max"],
                delta_12=vals["delta_12"], delta_34=vals["delta_34"], odd_spins=vals["odd_spins"],
                compute=False
            )
            cbt.m_order = vals["m_order"]
            cbt.n_order = vals["n_order"]
            cbt.table = vals["table"]
            return cbt
        if _type =="pycftboot.ConvolvedBlockTable":
            vals = obj["_value"]
            convbt = ConvolvedBlockTable(compute=False)
            convbt.dim = vals["dim"]
            convbt.k_max = vals["k_max"]
            convbt.l_max = vals["l_max"]
            convbt.m_max = vals["m_max"]
            convbt.n_max = vals["n_max"]
            convbt.delta_12 = vals["delta_12"]
            convbt.delta_34 = vals["delta_34"]
            convbt.m_order = vals["m_order"]
            convbt.n_order = vals["n_order"]
            convbt.odd_spins = vals["odd_spins"]
            convbt.table = vals["table"]
            return convbt
        elif _type == "pycftboot.PolynomialVector":
            return PolynomialVector(
                derivatives=obj["_value"]["vector"],
                spin_irrep=obj["_value"]["label"],
                poles=obj["_value"]["poles"]
            )
        elif _type == "symengine.Add":
            return Add(obj["_value"])
        elif _type == "symengine.Mul":
            return Mul(obj["_value"])
        return obj


def json_write(block_table, name):
    if isinstance(block_table, (ConformalBlockTable, ConvolvedBlockTable)):
        if os.path.dirname(name) != '':
            os.makedirs(os.path.dirname(name), exist_ok=True)

        with open(name, 'w') as f:
            json.dump(block_table, f, cls=CustomEncoder)
    else:
        raise TypeError(f"{block_table} is not a ConformalBlockTable or a ConvolvedBlockTable")


def json_read(name):
    with open(name, 'r') as f:
        block_table = json.load(f, cls=CustomDecoder)

    if isinstance(block_table, (ConformalBlockTable, ConvolvedBlockTable)):
        return block_table
    else:
        raise TypeError(f"{name} is not a json-encoded ConformalBlockTable or a ConvolvedBlockTable")
