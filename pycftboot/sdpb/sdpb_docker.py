import os
from subprocess import CompletedProcess
import docker

from .sdpb import Sdpb


class SdpbDocker(Sdpb):
    """Interface for running ``SDPB`` and related software in docker container

    Warning:
        To use this interface docker must be installed and has to be able to pull
        the specified image

    It's recommended to have at least a basic knowledge of how `docker <https://www.docker.com/>`_
    works, in particular about `volumes <https://docs.docker.com/storage/volumes/>`_.

    By default it uses the `official docker image <https://hub.docker.com/r/wlandry/sdpb>`_
    and the executable path are the ones of this docker image.

    Args:
        volume: directory that will be mounted inside the docker container,
                defaults to ``'.'`` which is the current directory
        user: user that will run the docker command, defaults to ``None``
              which is the current user
        image: docker image of ``SDPB``
        sdpb_bin: path of the ``sdpb`` binary inside the docker image
        pvm2sdp_bin: path of the ``pvm2sdp`` binary inside the docker image
        mpirun_bin: path of the ``mpirun`` binary inside the docker image
        unisolve_bin: path of the ``unisolve`` binary inside the docker image

    Attributes:
        volume_abs: absolute path of ``volume``
        debug: flags for debugging (makes :func:`~pycftboot.sdpb.sdpb_docker.SdpbDocker.run_command`
               output the command to ``stdout``)
    """

    def __init__(self, volume: str = '.', user: str = None, image: str = "wlandry/sdpb:2.5.1", sdpb_bin: str = "/usr/local/bin/sdpb", pvm2sdp_bin: str = "/usr/local/bin/pvm2sdp", mpirun_bin: str = "/usr/bin/mpirun", unisolve_bin: str = "/usr/local/bin/unisolve"):
        # User and docker volume stuff
        if user is None:
            user = os.getuid()
            self.user = f'{user}:{user}'
        self.volume = volume
        self.volume_abs = os.path.abspath(volume)

        os.makedirs(self.volume, exist_ok=True)

        self.image = image

        self.bin = sdpb_bin
        self.pvm2sdp_bin = pvm2sdp_bin
        self.mpirun_bin = mpirun_bin
        self.unisolve_bin = unisolve_bin

        self.debug = False

        super().__init__()

    def run_command(self, command: list) -> CompletedProcess:
        """Run command in the docker container specified by ``image``

        This is basically a wrapper of docker API which runs equivalent
        command line command::

            docker run -v <volume>:/work/<volume> -w /work -u <user> -d <image> <command>

        If the ``debug`` attribute is ``True`` it prints to ``stdout`` the ``command``

        Args:
            command: command to run as a list
        """
        # Print command to stdout if debug is true
        if self.debug:
            print(" ".join(command))

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

        completed_process = CompletedProcess(
            args=command,
            returncode=result["StatusCode"],
            stdout=stdout,
            stderr=stderr
        )
        completed_process.check_returncode()

        return completed_process
