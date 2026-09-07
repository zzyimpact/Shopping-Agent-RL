"""无 Web UI 的简洁 collection 终端日志。"""

from __future__ import annotations

from typing import TextIO
import sys


class ProgressLogger:
    def __init__(self, stream: TextIO = sys.stdout):
        self.stream = stream

    def line(self, message: str) -> None:
        print(message, file=self.stream, flush=True)

    def retry(self, status: str, retry: int, maximum: int, delay: float) -> None:
        self.line(f"[API] {status} | retry {retry}/{maximum} | next in {delay:.1f}s")

    def infrastructure_stop(self, resume_command: str) -> None:
        self.line("[STOP] API unavailable after retries.")
        self.line("Completed trajectories have been preserved.")
        self.line(f"Resume with: {resume_command}")
