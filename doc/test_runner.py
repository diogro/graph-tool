#! /usr/bin/env python
# graph-tool -- a general graph modification and manipulation thingy
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
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""Parallel job runner shared by the out-of-tree inference test scripts.

Each job runs in a persistent worker process (started with "spawn", so that
no OpenMP state is inherited from the master), with a configurable number of
OpenMP threads. The stdout/stderr of each job (including output from C++ and
a traceback in case of a segfault) go to a per-job log file. The master
process displays an overall progress bar, plus one bar per worker with the
progress of its current job. A worker that dies (e.g. segfault) marks its
job as crashed, and is replaced.

Job functions must be defined at module level (so they can be pickled), and
report progress via :func:`progress` and :func:`set_total`.
"""

import os
import sys
import time
import queue
import traceback
import faulthandler
import multiprocessing as mp
from tqdm import tqdm

_ctx = mp.get_context("spawn")

# worker-side state
_wid = None
_out_q = None
_last_report = 0
_pending = 0


def set_total(total):
    """Set the total number of steps of the current job."""
    _out_q.put(("total", _wid, total))


def progress(n=1):
    """Advance the progress of the current job by ``n`` steps."""
    global _last_report, _pending
    _pending += n
    t = time.monotonic()
    if t - _last_report > .1:
        _out_q.put(("progress", _wid, _pending))
        _pending = 0
        _last_report = t


def _flush_progress():
    global _pending
    if _pending > 0:
        _out_q.put(("progress", _wid, _pending))
        _pending = 0


def _redirect(path):
    sys.stdout.flush()
    sys.stderr.flush()
    f = open(path, "w", buffering=1)
    os.dup2(f.fileno(), 1)
    os.dup2(f.fileno(), 2)
    faulthandler.enable(file=f, all_threads=True)
    return f


def _worker(wid, in_q, out_q, omp_threads, init, init_args, log_dir):
    global _wid, _out_q
    _wid = wid
    _out_q = out_q
    devnull = os.open(os.devnull, os.O_WRONLY)
    f = _redirect(os.path.join(log_dir, f"worker{wid:02d}-init.log"))
    import graph_tool
    graph_tool.openmp_set_num_threads(omp_threads)
    if init is not None:
        init(*init_args)
    faulthandler.disable()
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    f.close()
    out_q.put(("ready", wid))
    while True:
        task = in_q.get()
        if task is None:
            break
        jid, name, func, args, log_path = task
        f = _redirect(log_path)
        print(f"# {name}", flush=True)
        try:
            ret = func(*args)
            ok, info = True, ret
        except BaseException:
            traceback.print_exc()
            ok, info = False, traceback.format_exc(limit=-1).strip()
        _flush_progress()
        sys.stdout.flush()
        sys.stderr.flush()
        faulthandler.disable()
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        f.close()
        try:
            out_q.put(("done", wid, jid, ok, info))
        except Exception:
            out_q.put(("done", wid, jid, ok, repr(info)))


class Job:
    """A job to be run. ``func(*args)`` is called in a worker process; its
    return value (if picklable) is made available in ``Job.result``."""

    def __init__(self, name, func, args=(), total=None):
        self.name = name
        self.func = func
        self.args = args
        self.total = total
        self.ok = None
        self.result = None
        self.log = None


def run_jobs(jobs, workers=os.cpu_count(), omp_threads=1, log_dir="logs",
             init=None, init_args=(), title="jobs"):
    """Run ``jobs`` in parallel in ``workers`` processes, each using
    ``omp_threads`` OpenMP threads. ``init(*init_args)`` is called once in
    each worker. Returns the list of failed jobs."""

    os.environ.setdefault("OMP_WAIT_POLICY", "passive")
    os.makedirs(log_dir, exist_ok=True)
    workers = max(1, min(workers, len(jobs)))
    out_q = _ctx.Queue()

    width = max([len(j.name) for j in jobs] + [10])
    main_bar = tqdm(total=len(jobs), position=0, desc=title, unit="job",
                    dynamic_ncols=True)
    bars = [tqdm(total=0, position=i + 1, leave=False, dynamic_ncols=True,
                 bar_format="{desc} {percentage:3.0f}%|{bar}| "
                            "{n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
            for i in range(workers)]

    procs = [None] * workers
    in_qs = [None] * workers
    current = [None] * workers
    ready = [False] * workers
    todo = list(range(len(jobs)))
    todo.reverse()
    failed = []
    ndone = 0

    def start(w):
        in_qs[w] = _ctx.Queue()
        procs[w] = _ctx.Process(target=_worker,
                                args=(w, in_qs[w], out_q, omp_threads, init,
                                      init_args, log_dir),
                                daemon=True)
        procs[w].start()
        ready[w] = False
        current[w] = None
        bars[w].reset(total=0)
        bars[w].set_description_str(f"[{w:2d}] (starting)".ljust(width + 5))

    def dispatch(w):
        if not todo:
            in_qs[w].put(None)
            current[w] = None
            bars[w].set_description_str(f"[{w:2d}] (idle)".ljust(width + 5))
            return
        jid = todo.pop()
        job = jobs[jid]
        job.log = os.path.join(log_dir, f"{jid:05d}.log")
        current[w] = jid
        bars[w].reset(total=job.total or 0)
        bars[w].set_description_str(f"[{w:2d}] {job.name}".ljust(width + 5))
        in_qs[w].put((jid, job.name, job.func, job.args, job.log))

    def finish(jid, ok, info):
        nonlocal ndone
        job = jobs[jid]
        job.ok = ok
        if ok:
            job.result = info
        else:
            job.result = None
            failed.append(job)
            main_bar.write(f"FAILED: {job.name} (log: {job.log})\n"
                           f"    {info}")
        ndone += 1
        main_bar.update(1)
        main_bar.set_postfix(failed=len(failed))

    for w in range(workers):
        start(w)

    try:
        while ndone < len(jobs):
            try:
                msg = out_q.get(timeout=.5)
            except queue.Empty:
                msg = None

            if msg is not None:
                kind, w = msg[0], msg[1]
                if kind == "ready":
                    ready[w] = True
                    dispatch(w)
                elif kind == "total":
                    bars[w].reset(total=msg[2])
                elif kind == "progress":
                    bars[w].update(msg[2])
                elif kind == "done":
                    jid, ok, info = msg[2:]
                    current[w] = None
                    finish(jid, ok, info)
                    dispatch(w)

            for w in range(workers):
                p = procs[w]
                if p.is_alive() or p.exitcode is None:
                    continue
                if current[w] is not None:
                    # drain messages the worker might have sent before dying
                    if msg is not None or not out_q.empty():
                        continue
                    jid = current[w]
                    finish(jid, False,
                           f"worker crashed with exit code {p.exitcode}")
                    start(w)
                elif not ready[w]:
                    # the worker died during initialization
                    raise RuntimeError(f"worker {w} failed to initialize "
                                       f"(exit code {p.exitcode}), see "
                                       f"{log_dir}/worker{w:02d}-init.log")
                elif todo:
                    start(w)
    except KeyboardInterrupt:
        main_bar.write("interrupted")
        for p in procs:
            p.terminate()
        raise
    finally:
        for b in bars:
            b.close()
        main_bar.close()
        for w, p in enumerate(procs):
            if p.is_alive():
                try:
                    in_qs[w].put(None)
                except Exception:
                    pass
                p.join(timeout=5)
                if p.is_alive():
                    p.terminate()

    return failed


def summary(jobs, failed, log_dir):
    print(f"\n{len(jobs) - len(failed)} of {len(jobs)} jobs passed.")
    if failed:
        print(f"{len(failed)} jobs FAILED (logs in {log_dir}):")
        for job in failed:
            print(f"  {job.name}: {job.log}")
    return 0 if not failed else 1
