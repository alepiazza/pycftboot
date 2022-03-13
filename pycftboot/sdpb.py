from abc import ABC, abstractmethod
import re
import os
from typing import Union
from symengine.lib.symengine_wrapper import RealMPFR

from .constants import prec


class Sdpb(ABC):
    """Abstract class for SDPB

    Attributes
    ----------
    version: version of the sdpb program
    options: available options
    defaults: default values of sdpb

    """

    def __init__(self):
        self.version = self.get_version()

        self.options = {'precision': prec}
        self.default_options = self.get_all_default_options()

        if self.version not in (1, 2):
            raise ValueError(f"Unkwon sdpb version = {self.version}")

    @abstractmethod
    def run_command(self, command: list):
        """Abstract function to run an arbitrary command

        Parameters
        ----------
        command: command as a list

        Returns
        -------
        returns a subprocess.CompletedProcess object
        """
        pass

    def run(self):
        """Runs an sdpb command with options specifie by self.options

        Returns
        -------
        returns a subprocess.CompletedProcess object
        """
        if 'sdpDir' not in self.options.keys():
            raise RuntimeError("sdpDir is mandatory argument of sdpb but was not set")

        args = self.options_to_args()

        if self.version == 1:
            self.run_command([self.path] + args)
        elif self.version == 2:
            if 'procsPerNode' not in self.options.keys():
                raise RuntimeError("procsPerNode is mandatory argument of sdpb (v2) but was not set")

            procs_per_node = self.options['procsPerNode']
            self.run_command(
                [self.mpirun_path] + ["--allow-run-as-root"] + ["-n", f"{procs_per_node}"] + \
                [self.path] + args
            )

    def pvm2sdp_run(self, input_xml, output_file):
        if self.version == 1:
            raise RuntimeError(f"Sdpb version {self.version} is not meant to be used with pvm2sdp")
        if 'procsPerNode' not in self.options.keys():
            raise RuntimeError("procsPerNode is mandatory argument of sdpb (v2) but was not set")
        procs_per_node = self.options['procsPerNode']
        self.run_command(
            [self.mpirun_path] + ["--allow-run-as-root"] + ["-n", f"{procs_per_node}"] + \
            [self.pvm2sdp_path, str(self.options["precision"]), input_xml, output_file]
        )

    def get_version(self):
        proc = self.run_command([self.path] + ["--version"])

        # Assume that this is version 1.x, which didn't support --version
        # Otherwise parse the output of --version
        if proc.returncode != 0:
            return 1
        else:
            m = re.search(r"SDPB ([0-9])", str(proc.stdout))
            if m is None:
                raise RuntimeError("Failed to retrieve SDPB version.")
            return int(m.group(1))

    def get_all_default_options(self):
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

    def options_to_args(self):
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

    def get_option(self, key: str):
        """
        Returns the string representation of a value that `SDPB` will use, whether
        or not it has been explicitly set.

        Parameters
        ----------
        key: The name of the `SDPB` parameter without any "--" at the beginning or
        "=" at the end.
        """
        keys = self.options.keys()
        if key not in keys:
            if key == "outDir":
                return self.options["sdpDir"] + "_out"
            elif key == "checkpointDir":
                return self.options["sdpDir"] + ".ck"
            elif key == "initialCheckpointDir":
                if "checkpointDir" in keys:
                    return self.options["checkpointDir"]
                else:
                    return self.options["sdpDir"] + ".ck"
            else:
                raise ValueError(f"key = {key} not found in self.options")
        else:
            return self.options[key]

    def set_required_options(self):
        self.options['sdpDir'] = self.sdpDir
        if self.version == 2:
            self.procsPerNode['procsPerNode'] = self.procsPerNode

    def set_option(self, key: str = None, value: Union[float, str, bool] = None):
        """
        Sets the value of a switch that should be passed to `SDPB` on the command
        line. `SDPB` options that do not take a parameter are handled by other
        methods so it should not be necessary to pass them.

        Parameters
        ----------
        key:   [Optional] The name of the `SDPB` parameter being set without any
               "--" at the beginning or "=" at the end. Defaults to `None` which
               means all parameters will be reset to their default values.
        value: [Optional] The string or numerical value that should accompany `key`.
               Defaults to `None` which means that the parameter for `key` will be
               reset to its default value.
        """
        if key is None:
            self.options = {}
        elif key in self.default_options.keys():
            self.options[key] = value
        else:
            raise ValueError(f"key = {key} is not a sdpb valid option")

    def read_output(self, name):
        """
        Reads an `SDPB` output file and returns a dictionary in which all entries
        have been converted to their respective Python types.

        Parameters
        ----------
        name:       [Optional] The name of the file without any ".out" at the end.
                    Defaults to "mySDP".
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
