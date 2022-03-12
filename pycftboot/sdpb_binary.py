import os
import subprocess
import shutil

from .sdpb import Sdpb


class SdpbBinary(Sdpb):
    """ Class for SDPB when installed as a binary
    """
    def __init__(self, procs_per_node=1, sdpb_bin='/usr/bin/sdpb', pvm2sdp_bin='/usr/bin/pvm2sdp', mpirun_bin='/usr/bin/mpirun'):
        self.path = self.__find_executable(sdpb_bin)

        super().__init__(procs_per_node)

        if self.version == 2:
            self.pvm2sdp_path = self.__find_executable(pvm2sdp_bin)
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
            return bin_in_path

    def run_command(self, command):
        return subprocess.run(command, capture_output=True, check=True, text=True)
