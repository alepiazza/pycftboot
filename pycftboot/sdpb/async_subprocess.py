"""https://stackoverflow.com/questions/17190221/subprocess-popen-cloning-stdout-and-stderr-both-to-terminal-and-variables/
"""

import asyncio
import sys
from asyncio.subprocess import PIPE

@asyncio.coroutine
def read_stream_and_display(stream, display):
    """Read from stream line by line until EOF, display, and capture the lines.
    """
    output = []
    while True:
        line = yield from stream.readline()
        if not line:
            break
        output.append(line)
        display(line) # assume it doesn't block
    return b''.join(output)

@asyncio.coroutine
def read_and_display(*cmd):
    """Capture cmd's stdout, stderr while displaying them as they arrive
    (line by line).
    """
    # start process
    process = yield from asyncio.create_subprocess_exec(*cmd, stdout=PIPE, stderr=PIPE)

    # read child's stdout/stderr concurrently (capture and display)
    try:
        stdout, stderr = yield from asyncio.gather(
            read_stream_and_display(process.stdout, sys.stdout.buffer.write),
            read_stream_and_display(process.stderr, sys.stderr.buffer.write)
        )
    except Exception:
        process.kill()
        raise
    finally:
        # wait for the process to exit
        rc = yield from process.wait()
    return rc, stdout, stderr
