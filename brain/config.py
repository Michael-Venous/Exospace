"""Runtime configuration, loaded from the environment (and `.env`)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    gemini_api_key: str
    gemini_model: str
    gemini_thinking: str | None
    pico_serial: str
    pico_baud: int
    capture_device: str
    capture_width: int
    capture_height: int
    session_path: Path
    bench_csv: Path
    planner_model: str = "gemini-4-argon"
    planner_thinking: str | None = "medium"
    model_timeout_s: float = 30.0
    planner_timeout_s: float = 120.0
    model_log: Path = Path("bench/model_calls.jsonl")
    executor: str = "jev-flash"
    typesafe_api_key: str = ""
    jev_model: str = "jev-1.14.0"
    jev_timeout_s: float = 20.0
    jev_min_presence: float = 0.85
    jev_min_confidence: float = 0.6

    @classmethod
    def load(cls) -> Config:
        load_dotenv()
        env = os.environ.get
        return cls(
            gemini_api_key=env("GEMINI_API_KEY", ""),
            gemini_model=env("FAST_MODEL", env("GEMINI_MODEL", "gemini-3.8-flash")),
            gemini_thinking=env("GEMINI_THINKING", "low") or None,
            pico_serial=env("PICO_SERIAL", "/dev/ttyAMA0"),
            pico_baud=int(env("PICO_BAUD", "115200")),
            capture_device=env("CAPTURE_DEVICE", "/dev/video0"),
            capture_width=int(env("CAPTURE_WIDTH", "1920")),
            capture_height=int(env("CAPTURE_HEIGHT", "1080")),
            session_path=Path(env("SESSION_PATH", "session.json")),
            bench_csv=Path(env("BENCH_CSV", "bench/results.csv")),
            planner_model=env("PLANNER_MODEL", "gemini-4-argon"),
            planner_thinking=env("PLANNER_THINKING", "medium") or None,
            model_timeout_s=float(env("MODEL_TIMEOUT_S", "30")),
            planner_timeout_s=float(env("PLANNER_TIMEOUT_S", "120")),
            model_log=Path(env("MODEL_LOG", "bench/model_calls.jsonl")),
            executor=env("EXECUTOR", "jev-flash"),
            typesafe_api_key=env("TYPESAFE_API_KEY", ""),
            jev_model=env("JEV_MODEL", "jev-1.14.0"),
            jev_timeout_s=float(env("JEV_TIMEOUT_S", "20")),
            jev_min_presence=float(env("JEV_MIN_PRESENCE", "0.85")),
            jev_min_confidence=float(env("JEV_MIN_CONFIDENCE", "0.6")),
        )
