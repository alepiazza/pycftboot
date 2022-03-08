from abc import ABC, abstractmethod

import os
import subprocess
import shutil
import re
import warnings
import docker


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
        self.option = self.get_options()
        self.defaults = self.get_defaults()

    def get_version(self):
        proc = self.run('--version')

        # Assume that this is version 1.x, which didn't support --version
        # Otherwise parse the output of --version
        if proc.returncode != 0:
            return 1
        else:
            m = re.search(r"SDPB ([0-9])", str(proc.stdout))
            if m is None:
                raise RuntimeError("Failed to retrieve SDPB version.")
            return int(m.group(1))

    @abstractmethod
    def run(self, args):
        """Run sdpb command with arguments given

        Parameters
        ----------
        args: sdpb arguments

        Returns
        -------
        object with at least stdout, stderr, returncode attributes
        """
        pass

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


class SdpbBinary(Sdpb):
    """ Class for SDPB when installed as a binary
    """
    def __init__(self, sdpb_bin='/usr/bin/sdpb', mpirun_bin='/usr/bin/mpirun'):
        self.path = self.__find_executable(sdpb_bin)

        super().__init__()

        if self.version > 1:
            self.mpirun_path = self.__find_executable(mpirun_bin)

    def __find_executable(self, name):
        """Searches for the executable given by `name`. If not found we try to
        find the executable in path
        """
        if os.path.isfile(name):
            return name
        else:
            bin_name = os.path.basename(name)
            bin_in_path = shutil.which(bin_name)
            if bin_in_path is None:
                raise EnvironmentError(f"{bin_name} not found or not in PATH")
            elif not os.path.isfile(bin_in_path):
                raise EnvironmentError(f"{bin_in_path} is in PATH but was not found")
            warnings.warn(f"{name} was not found, but found {bin_in_path} in PATH")
            return bin_in_path

    def run(self, *args):
        return subprocess.run([self.path] + list(args), capture_output=True, check=True, text=True)


class SdpbDocker(Sdpb):
    """ Class for SDPB when running in docker
    """

    def __init__(self, image="wlandry/sdpb:2.5.1"):
        self.image = image
        self.__client = docker.from_env()

        super().__init__()

    def run_in_docker(self, command):
        """Run command in a docker image specified by `image`

        Attributes
        ----------
        command: string or list commnad

        Returns
        -------
        object with at stdout, stderr, returncode attributes of the command
        """
        container = self.__client.containers.run(self.image, command=command, detach=True)

        result = container.wait()
        stdout = container.logs(stdout=True, stderr=False).decode("utf-8")
        stderr = container.logs(stdout=False, stderr=True).decode("utf-8")

        if result["StatusCode"] != 0:
            raise RuntimeError(stderr)

        container.remove()

        return type("run", (object,), {"stdout": stdout, "stderr": stderr, "returncode": result["StatusCode"]})

    def run(self, *args):
        return self.run_in_docker(["/usr/local/bin/sdpb"] + list(args))
