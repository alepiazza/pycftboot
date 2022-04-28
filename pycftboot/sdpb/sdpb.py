from abc import ABC, abstractmethod
import re
import os
from typing import Union
from symengine.lib.symengine_wrapper import RealMPFR
from subprocess import CompletedProcess

from ..constants import prec


class Sdpb(ABC):
    """Abstract class for interacting with `SDPB <https://github.com/davidsd/sdpb>`_

    The actual :func:`pycftboot.sdpb.Sdpb.run_command` must be implemented by a
    derived class, but this class implements everything else is needed
    to interact with the ``SDPB`` program

    Attributes:
        version (int): version of the sdpb program
        options (dict): manually set options, managed by :func:`pycftboot.sdpb.Sdpb.set_option`
        defaults (dict): default options of sdpb
    """

    def __init__(self):
        self.version = self.get_version()

        self.options = {'precision': prec}
        self.default_options = self.get_all_default_options()

        if self.version not in (1, 2):
            raise ValueError(f"Unkwon sdpb version = {self.version}")

    @abstractmethod
    def run_command(self, command: list) -> CompletedProcess:
        """Abstract function to run an arbitrary command

        Args:
            command: command as a list
        """
        pass

    def run(self, extra_options: dict = {}) -> CompletedProcess:
        """Runs an sdpb command with options specified by self.options

        Args:
           extra_options: extra options passed to sdpb, it's recommended
                          to use :func:`pycftboot.sdpb.Sdpb.set_option` instead
        """
        if 'sdpDir' not in self.options.keys():
            raise RuntimeError("sdpDir is mandatory argument of sdpb but was not set")

        for key, val in extra_options.items():
            self.set_option(key, val)

        args = self.options_to_args()

        if self.version == 1:
            sdpb_out = self.run_command([self.bin] + args)
        elif self.version == 2:
            if 'procsPerNode' not in self.options.keys():
                raise RuntimeError("procsPerNode is mandatory argument of sdpb (v2) but was not set")

            procs_per_node = self.options['procsPerNode']
            sdpb_out = self.run_command(
                [self.mpirun_bin] + ["--allow-run-as-root"] + ["-n", f"{procs_per_node}"] +
                [self.bin] + args
            )

        for key in extra_options.keys():
            self.set_default_option(key)

        return sdpb_out

    def pvm2sdp_run(self, input_xml: str, output_file: str) -> CompletedProcess:
        """Runs ``pvm2spb <input_xml> <output_file>`` with ``mpirun``. This is
        converts the xml file into the sdpb input
        """
        if self.version == 1:
            raise RuntimeError(f"Sdpb version {self.version} is not meant to be used with pvm2sdp")
        if 'procsPerNode' not in self.options.keys():
            raise RuntimeError("procsPerNode is mandatory argument of sdpb (v2) but was not set")
        procs_per_node = self.options['procsPerNode']
        return self.run_command(
            [self.mpirun_bin] + ["--allow-run-as-root"] + ["-n", f"{procs_per_node}"] +
            [self.pvm2sdp_bin, str(self.options["precision"]), input_xml, output_file]
        )

    def unisovle_run(self, precision: int, input_file: str) -> CompletedProcess:
        """Runs the ``unisolve`` on given file
        """
        return self.run_command([self.unisolve_bin, "-H1", "-o" + str(prec), "-Oc", "-Ga", input_file])

    def get_version(self) -> int:
        """Find out the version of sdpb that we are using
        """
        proc = self.run_command([self.bin] + ["--version"])

        # Assume that this is version 1.x, which didn't support --version
        # Otherwise parse the output of --version
        if proc.returncode != 0:
            return 1
        else:
            m = re.search(r"SDPB ([0-9])", str(proc.stdout))
            if m is None:
                raise RuntimeError("Failed to retrieve SDPB version.")
            return int(m.group(1))

    def get_all_default_options(self) -> dict:
        """
        Returns:
             dictionary with the default SDPB options as given by ``sdpb --help``
        """
        default_sdpDir = 'work_sdpb'
        options = {
            'sdpDir': default_sdpDir,
            'checkpointInterval': 3600,
            'maxIterations': 500,
            'maxRuntime': 86400,
            'dualityGapThreshold': 1e-30,
            'primalErrorThreshold': 1e-30,
            'dualErrorThreshold': 1e-30,
            'initialMatrixScalePrimal': 1e+20,
            'initialMatrixScaleDual': 1e+20,
            'feasibleCenteringParameter': 0.1,
            'infeasibleCenteringParameter': 0.3,
            'stepLengthReduction': 0.7,
            'maxComplementarity': 1e+100
        }

        if self.version == 1:
            options.update({
                'maxThreads': 4,
                'choleskyStabilizeThreshold': 1e-40
            })
        elif self.version == 2:
            options.update({
                'procsPerNode': 1,
                'paramFile': '',
                'outDir': f'{default_sdpDir}_out',
                'noFinalCheckpoint': False,
                'writeSolutions': 'x,y',
                'procGranularity': 1,
                'verbosity': 1,
                'precision': prec,
                'findPrimalFeasible': False,
                'findDualFeasible': False,
                'detectPrimalFeasibleJump': False,
                'detectDualFeasibleJump': False,
                'minPrimalStep': 0,
                'minDualStep': 0,
                'checkpointDir': f'{default_sdpDir}.ck',
                'initialCheckpointDir': f'{default_sdpDir}.ck'
            })
            options['maxRuntime'] = 922337203685477580

        return options

    def options_to_args(self) -> list:
        """
        Returns:
            :attr:`pycftboot.sdpb.Sdpb.options` as command line options
        """
        options_without_bools = self.options
        for (key, value) in self.options.items():
            if key == 'paramFile' and value == '':
                options_without_bools.pop(key)
            elif isinstance(value, bool):
                if value is True:
                    options_without_bools[key] = ''
                else:
                    options_without_bools.pop(key)

        return (' '.join([f'--{key} {value}' for key, value in options_without_bools.items()])).split()

    def get_option(self, key: str) -> str:
        """This is the recommended way to set options
        Args:
            key: the name of the ``SDPB`` parameter

        Returns:
            a value that ``SDPB`` will use, whether or not it has been explicitly set.
        """
        keys = self.options.keys()
        if key not in keys:
            if key == "outDir":
                if self.version == 1:
                    return self.options["sdpDir"] + ".out"
                elif self.version == 2:
                    return self.options["sdpDir"] + "_out"
            elif key == "checkpointDir":
                return self.options["sdpDir"] + ".ck"
            elif key == "initialCheckpointDir":
                if "checkpointDir" in keys:
                    return self.options["checkpointDir"]
                else:
                    return self.options["sdpDir"] + ".ck"
            elif key in self.default_options.keys():
                return self.default_options[key]
            else:
                raise ValueError(f"key = {key} not found in self.options or self.default_options")
        else:
            return self.options[key]

    def set_option(self, key: str = None, value: Union[float, str, bool] = None):
        """
        Sets the value of an option that should be passed to ``SDPB`` on the command
        line. ``SDPB`` options that do not take a parameter are managed by ``True``
        or ``False``, for instance calling ``set_option('findPrimalFeasible', True)``
        will have the effect of calling ``sdpb [...] --findPrimalFeasible``

        Args:
            key: the name of the ``SDPB`` parameter being set without any
                 ``--`` at the beginning or ``=`` at the end. Defaults to ``None`` which
                 means all parameters will be reset to their default values.
            value: the value that should accompany ``key``. Defaults to ``None`` which
                   means that the parameter for ``key`` will be reset to its default value.
        """
        if key is None:
            self.options = {}
        elif value is None:
            self.set_default_option(key)
        elif key in self.default_options.keys():
            self.options[key] = value
        else:
            raise ValueError(f"key = {key} is not a sdpb valid option")

    def set_default_option(self, key: str):
        """Sets an ``SDPB`` option to default

        Args:
            key: the name of the ``SDPB`` option to restore to default
        """
        self.options.pop(key)

    def read_output(self, name: str) -> dict:
        """
        Reads an ``SDPB`` output file and returns a dictionary

        Args:
            name: the name of the file (sdpb v1) or directory (sdpb v2)
        """
        ret = {}

        if self.version == 1:
            if not os.path.isfile(name):
                raise ValueError(f'name = {name} must be a file in sdpb v1')
            out_name = name
        else:
            if not os.path.isdir(name):
                raise ValueError(f'output name = {name} must be a directory in sdpb v2')
            out_name = f'{name}/out.txt'
            y_name = f"{name}/y.txt"

        with open(out_name, "r") as out_file:
            lines = out_file.read().splitlines()

        for line in lines:
            (key, value) = line.strip(';').split(' = ')

            key = key.strip()
            if key == "terminateReason":
                ret[key] = value.strip('"')
            elif key == "Solver runtime":
                ret[key] = float(value)
            else:
                ret[key] = RealMPFR(value, prec)

        if self.version == 2:
            y = []
            with open(y_name, "r") as out_file:
                lines = out_file.read().splitlines()[1:]

            for value in lines:
                y.append(RealMPFR(value, prec))

            ret["y"] = y

        return ret
