#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# graph_tool -- a general graph manipulation python module
#
# Copyright (C) 2006-2026 Tiago de Paula Peixoto <tiago@skewed.de>
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import pickle
import atexit
import io
import os
from . import libgraph_tool_core

# IStream and OStream need to be tweaked a little to become a real file-like
# object...

def IStream_read(self, n=None, buflen=1048576):
    if n is None:
        data = b""
        while True:
            buf = self.read_buf(buflen)
            data += buf
            if len(buf) < buflen:
                break
        return data
    else:
        return self.read_buf(n)

def IStream_readline(self, n=None):
    c = None
    line = b""
    while c != "" and c != "\n" and len(line) < n:
        c = self.read_buf(1)
        line += c
    return line

def OStream_write(self, s):
    data = s
    self.write_buf(data, len(s))

libgraph_tool_core.IStream.read_buf = libgraph_tool_core.IStream.read
libgraph_tool_core.OStream.write_buf = libgraph_tool_core.OStream.write
libgraph_tool_core.IStream.read = IStream_read
libgraph_tool_core.IStream.readline = IStream_readline
libgraph_tool_core.OStream.write = OStream_write

# define and set the pickler/unpickler functions

GT_PICKLE_PROTOCOL = -1

def pickler(stream, obj):
    sstream = io.BytesIO()
    pickle.dump(obj, sstream, GT_PICKLE_PROTOCOL)
    stream.write(sstream.getvalue())

def unpickler(stream):
    data = stream.read()
    sstream = io.BytesIO(data)
    return pickle.load(sstream, encoding="bytes")

libgraph_tool_core.set_pickler(pickler)
libgraph_tool_core.set_unpickler(unpickler)

def clean_picklers():
    libgraph_tool_core.set_pickler(None)
    libgraph_tool_core.set_unpickler(None)

atexit.register(clean_picklers)

class BufferedPipe:
    def __init__(self, buffer_size=1 << 27, pipe_size=1 << 20):
        r_fd, w_fd = os.pipe()
        self.size = buffer_size
        try:
            import fcntl
            F_SETPIPE_SZ = 1031
            fcntl.fcntl(w_fd, F_SETPIPE_SZ, pipe_size)
        except (ImportError, OSError):
            pass
        self.writer = io.BufferedWriter(io.FileIO(w_fd, "w", closefd=True),
                                        buffer_size=buffer_size)
        self.reader = io.BufferedReader(io.FileIO(r_fd, "r", closefd=True),
                                        buffer_size=buffer_size)
