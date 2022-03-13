import os
import subprocess
import docker

from .sdpb import Sdpb


class SdpbDocker(Sdpb):
    """ Class for SDPB when running in docker
    """

    def __init__(self, volume='.', user=None, image="wlandry/sdpb:2.5.1", sdpb_path="/usr/local/bin/sdpb", pvm2sdp_path="/usr/local/bin/pvm2sdp", mpirun_path="/usr/bin/mpirun", unisolve_path="/usr/local/bin/unisolve"):
        # User and docker volume stuff
        if user is None:
            user = os.getuid()
            self.user = f'{user}:{user}'
        self.volume = volume
        self.volume_abs = os.path.abspath(volume)

        os.makedirs(self.volume, exist_ok=True)

        self.image = image

        self.path = sdpb_path
        self.pvm2sdp_path = pvm2sdp_path
        self.mpirun_path = mpirun_path
        self.unisovle_path = unisolve_path

        super().__init__()

    def run_command(self, command):
        """Run command in a docker image specified by `image`

        Attributes
        ----------
        command: string or list command

        Returns
        -------
        object with at stdout, stderr, returncode attributes of the command
        """
        # Initialize docker client
        client = docker.from_env()

        container = client.containers.run(
            self.image,
            command=command,
            user=self.user,
            environment={'OMPI_ALLOW_RUN_AS_ROOT': '1', 'OMPI_ALLOW_RUN_AS_ROOT_CONFIRM': '1'},
            volumes={self.volume_abs: {'bind': f'/work/{self.volume}', 'mode': 'rw'}},
            working_dir='/work',
            detach=True
        )

        result = container.wait()
        stdout = container.logs(stdout=True, stderr=False).decode("utf-8")
        stderr = container.logs(stdout=False, stderr=True).decode("utf-8")

        if result["StatusCode"] != 0:
            raise RuntimeError(stderr)

        container.remove()
        client.close()

        completed_process = subprocess.CompletedProcess(
            args=command,
            returncode=result["StatusCode"],
            stdout=stdout,
            stderr=stderr
        )
        completed_process.check_returncode()

        return completed_process
