from abc import ABC, abstractmethod
import re
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

    def __init__(self, procs_per_node: int = 1):
        self.version = self.get_version()
        self.option = self.get_options()
        self.defaults = self.get_defaults()
        self.procs_per_node = procs_per_node

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

    def run(self, args: list):
        """Runs an sdpb command

        Parameters
        ----------
        args: sdpb options

        Returns
        -------
        returns a subprocess.CompletedProcess object
        """
        if self.version == 1:
            self.run_command([self.path] + args)
        elif self.version == 2:
            self.run_command([self.mpirun_path] + ["-n", f"{self.procs_per_node}"] + [self.path] + args)

    def pvm2sdp_run(self, args: list):
        if self.version == 1:
            raise RuntimeError(f"Sdpb version {self.version} is not meant to be used with pvm2sdp")
        self.run_command([self.mpirun_path] + ["-n", f"{self.procs_per_node}"] + [self.pvm2sdp_path] + args)

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

    def get_options(self):
        COMMON_OPTIONS = ["checkpointInterval", "maxIterations", "maxRuntime", "dualityGapThreshold", "primalErrorThreshold", "dualErrorThreshold", "initialMatrixScalePrimal", "initialMatrixScaleDual", "feasibleCenteringParameter", "infeasibleCenteringParameter", "stepLengthReduction", "maxComplementarity"]
        if self.version == 1:
            options_extra = ["maxThreads", "choleskyStabilizeThreshold"]
        else:
            options_extra = ["procsPerNode", "procGranularity", "verbosity"]
        return options_extra + COMMON_OPTIONS

    def get_defaults(self):
        COMMON_DEFAULTS = ["3600", "500", "86400", "1e-30", "1e-30", "1e-30", "1e+20", "1e+20", "0.1", "0.3", "0.7", "1e+100"]
        if self.version == 1:
            defaults_extra = ["4", "1e-40"]
        else:
            defaults_extra = ["0", "1", "1"]
        return defaults_extra + COMMON_DEFAULTS

    def read_output(self, name="mySDP"):
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
            out_file_name = f"{name}.out"
        else:
            out_file_name = f"{name}_out/out.txt"

        with open(out_file_name, "r") as out_file:
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
            with open(f"{name}_out/y.txt", "r") as out_file:
                lines = out_file.read().splitlines()[1:]

            for value in lines:
                y.append(RealMPFR(value, prec))

            ret["y"] = y

        return ret
