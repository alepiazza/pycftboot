from .sdpb_binary import SdpbBinary
from .sdpb_docker import SdpbDocker
from .sdpb_singularity import SdpbSingularity

__all__ = ['SdpbBinary', 'SdpbDocker', 'SdpbSingularity']


INTERFACES = ('binary', 'docker', 'singularity')


def sdpb_switch(sdpb_mode, sdpb_kwargs):
    if sdpb_mode not in INTERFACES:
        raise ValueError(f"sdpb_mode = {sdpb_mode} must be {' or '.join(INTERFACES)}")

    if sdpb_mode == 'binary':
        return SdpbBinary(**sdpb_kwargs)
    elif sdpb_mode == 'docker':
        return SdpbDocker(**sdpb_kwargs)
    elif sdpb_mode == 'singularity':
        return SdpbSingularity(**sdpb_kwargs)
