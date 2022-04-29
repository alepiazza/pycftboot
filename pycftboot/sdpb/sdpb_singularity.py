import os
from subprocess import CompletedProcess
from spython.main import Client

from .sdpb_docker import SdpbDocker


class SdpbSingularity(SdpbDocker):
    """Interface for running ``SDPB`` and related software in sigularity container


    Warning:
        To use this interface singularity must be installed and has to be able to pull
        the specified image

    By default it uses the `official docker image <https://hub.docker.com/r/wlandry/sdpb>`_
    and the executable path are the ones of this docker image.
    """

    def __init__(self, volume: str = '.', image: str = "docker://wlandry/sdpb:2.5.1", sdpb_bin: str = "/usr/local/bin/sdpb", pvm2sdp_bin: str = "/usr/local/bin/pvm2sdp", mpirun_bin: str = "/usr/bin/mpirun", unisolve_bin: str = "/usr/local/bin/unisolve"):
        super().__init__(volume, None, image, sdpb_bin, pvm2sdp_bin, mpirun_bin, unisolve_bin)
        self.debug = False

    def run_command(self, command: list) -> CompletedProcess:
        """Run command in the singularity container specified by ``image``

        This is basically a wrapper of :func:`spython.main.Client.execute` which
        runs equivalent command line command::

            singularity -s exec -B <volume>:/work/<volume> -W /work <image> <command>

        If the ``debug`` attribute is ``True`` it prints to ``stdout`` the ``command``

        Args:
            command: command to run as a list
        """
        # Print command to stdout if debug is true
        if self.debug:
            print(" ".join(command))

        # Run command in singularity container
        result = Client.execute(
            image=self.image,
            command=command,
            singularity_options=['--silent'],
            options=[
                '--workdir', '/work',
                '--env', 'OMPI_ALLOW_RUN_AS_ROOT=1',
                '--env', 'OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1'
            ],
            bind=f'{self.volume_abs}:{os.path.join("/work", self.volume)}',
            return_result=True
        )

        # For some reason spython drops and sqashes the singularity
        # stdout and stderr (which is a subprocess.Popen output) into
        # message. We try to reconstrut the tuple (stdout, stderr). We
        # guess that if the message is a string, output is stdout or
        # stderr based on the exit code. beware that this might not
        # always be correct.
        #
        # https://github.com/singularityhub/singularity-cli/blob/master/spython/main/base/command.py#L103
        # https://github.com/singularityhub/singularity-cli/blob/master/spython/utils/terminal.py#L162

        if isinstance(result['message'], str):
            if result['return_code'] == 0:
                result['message'] = (result['message'], '')
            else:
                result['message'] = ('', result['message'])
        elif isinstance(result['message'], list):
            if len(result['message']) == 0:
                result['message'] = ('', '')
            if len(result['message']) == 2:
                result['message'] = (result['message'][0], result['message'][1])
            else:
                raise RuntimeError("Failed to interpret singularity cast into (stdout, stderr)")
        else:
            raise RuntimeError("Failed to interpret singularity cast into (stdout, stderr)")

        completed_process = CompletedProcess(
            args=command,
            returncode=result['return_code'],
            stdout=result['message'][0],
            stderr=result['message'][1]
        )

        completed_process.check_returncode()

        return completed_process
