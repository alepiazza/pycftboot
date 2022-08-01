import os
import subprocess
import asyncio
import shutil

from .sdpb import Sdpb
from .async_subprocess import read_and_display


class SdpbBinary(Sdpb):
    """Interface for running ``SDPB`` binary and related software

    Warning:
        To use this interface the ``sdpb``, ``pvm2sdp``, ``mpirun`` and
        ``unisolve`` must be installed

    The default binaries locations are some sensible ones, but they may not be correct.
    Hence, if not found we look for the binaries in ``PATH``.

    Args:
        sdpb_bin: path of the ``sdpb`` binary
        pvm2sdp_bin: path of the ``pvm2sdp`` binary
        mpirun_bin: path of the ``mpirun`` binary
        unisolve_bin: path of the ``unisolve`` binary

    Attributes:
        debug: flags for debugging (makes :func:`~pycftboot.sdpb.sdpb_binary.SdpbBinary.run_command` output the command to ``stdout``)
    """

    def __init__(self, realtime_output=False, sdpb_bin='/usr/bin/sdpb', pvm2sdp_bin='/usr/bin/pvm2sdp', mpirun_bin='/usr/bin/mpirun', unisolve_bin='/usr/bin/unisolve'):
        self.bin = self.__find_executable(sdpb_bin)

        self.debug = False
        self.realtime_output = realtime_output

        super().__init__()

        if self.version == 2:
            self.pvm2sdp_bin = self.__find_executable(pvm2sdp_bin)
            self.mpirun_bin = self.__find_executable(mpirun_bin)
        self.unisolve_bin = self.__find_executable(unisolve_bin)

    def __find_executable(self, name: str) -> str:
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
            return bin_in_path

    def run_command(self, command: list) -> subprocess.CompletedProcess:
        """This is just a wrapper around :func:`subprocess.run`

        If the ``debug`` attribute is ``True`` it prints to ``stdout`` the ``command``

        Args:
            command: command to run as a list
        """
        # Print command to stdout if debug is true
        if self.debug:
            print(" ".join(command))

        current_env = os.environ

        if self.realtime_output is True:
            loop = asyncio.get_event_loop()
            rc, stdout, stderr = loop.run_until_complete(read_and_display(*command))

            completed_process = subprocess.CompletedProcess(
                args=command,
                returncode=rc,
                stdout=stdout,
                stderr=stderr
            )
            completed_process.check_returncode()
        else:
            completed_process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, env=current_env)

        return completed_process
