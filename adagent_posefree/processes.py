"""Terminate only subprocess groups started by this run."""
import os
import signal
import subprocess
import threading

class Processes:
    def __init__(self):
        self.stop = threading.Event()
        self.lock = threading.RLock()
        self.children = {}

    def cancel(self):
        self.stop.set()
        with self.lock:
            for child in self.children.values():
                if child.poll() is None:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass

    @staticmethod
    def terminate_group(child):
        """Reap our process tree even when its leader exits before its children."""
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            child.wait()
            return
        try:
            child.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        # killpg still reaches descendants if their original leader has exited.
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()

    def run(self, command, env, log):
        with self.lock:
            if self.stop.is_set():
                raise RuntimeError("preparation cancelled")
            child = subprocess.Popen(command, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            self.children[child.pid] = child
        try:
            while True:
                try:
                    code = child.wait(timeout=1)
                    break
                except subprocess.TimeoutExpired:
                    if self.stop.is_set():
                        raise RuntimeError("preparation cancelled; intermediate cache retained")
            if code != 0:
                self.cancel()
                raise RuntimeError(f"preparation exited with {code}; inspect worker log")
        finally:
            if child.poll() is None or self.stop.is_set() or child.returncode != 0:
                self.terminate_group(child)
            with self.lock:
                self.children.pop(child.pid, None)
