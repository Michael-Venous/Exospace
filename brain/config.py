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
    fallback_model: str = "gemini-3.5-flash-lite"
    planner_model: str = "gemini-4-argon"
    planner_thinking: str | None = "medium"
    model_timeout_s: float = 30.0
    planner_timeout_s: float = 120.0
    model_log: Path = Path("bench/model_calls.jsonl")
    executor: str = "race"
    typesafe_api_key: str = ""
    jev_model: str = "jev-1.14.0"
    jev_timeout_s: float = 20.0
    jev_min_presence: float = 0.85
    jev_min_confidence: float = 0.6
    jev_solo_skills: bool = False  # experimental: triage opens apps and sites by keyboard alone
    zai_api_key: str = ""
    zai_model: str = "glm-4.6v-flashx"
    zai_planner_model: str = "glm-4.6v"
    zai_thinking: str | None = None  # thinking off: GLM's reasoning costs seconds per step
    zai_fallback_model: str = "glm-4.6v"
    zai_base_url: str = "https://api.z.ai/api/paas/v4"
    target_os: str = ""
    local_keyboard: str = ""

    @property
    def uses_zai(self) -> bool:
        """z.ai and Gemini are interchangeable: ZAI_API_KEY, when set, serves every executor."""
        return bool(self.zai_api_key)

    @property
    def model_name(self) -> str:
        """The language model the executor's own calls use (see `language_model`)."""
        return self.zai_model if self.uses_zai else self.gemini_model

    @classmethod
    def load(cls) -> Config:
        load_dotenv()
        env = os.environ.get
        executor = env("EXECUTOR") or "race"
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
            fallback_model=env("FALLBACK_MODEL", "gemini-3.5-flash-lite"),
            planner_model=env("PLANNER_MODEL", "gemini-4-argon"),
            planner_thinking=env("PLANNER_THINKING", "medium") or None,
            # z.ai flashx, 1,299 calls: p99 14 s, and a hung call runs to the timeout. At 12 s
            # 1.9% retry and a hang costs 12 s, not 30.
            model_timeout_s=float(env("MODEL_TIMEOUT_S") or ("12" if env("ZAI_API_KEY") else "30")),
            planner_timeout_s=float(env("PLANNER_TIMEOUT_S", "120")),
            model_log=Path(env("MODEL_LOG", "bench/model_calls.jsonl")),
            executor=executor,
            typesafe_api_key=env("TYPESAFE_API_KEY", ""),
            jev_model=env("JEV_MODEL", "jev-1.14.0"),
            jev_timeout_s=float(env("JEV_TIMEOUT_S", "20")),
            jev_min_presence=float(env("JEV_MIN_PRESENCE", "0.85")),
            jev_min_confidence=float(env("JEV_MIN_CONFIDENCE", "0.6")),
            jev_solo_skills=env("JEV_SOLO_SKILLS", "").strip().lower() in ("1", "true", "yes"),
            zai_api_key=env("ZAI_API_KEY", ""),
            zai_model=env("ZAI_MODEL", "glm-4.6v-flashx"),
            zai_planner_model=env("ZAI_PLANNER_MODEL", "glm-4.6v"),
            zai_thinking=env("ZAI_THINKING") or None,
            zai_fallback_model=env("ZAI_FALLBACK_MODEL", "glm-4.6v"),
            zai_base_url=env("ZAI_BASE_URL", "https://api.z.ai/api/paas/v4"),
            target_os=env("TARGET_OS", ""),
            local_keyboard=env("LOCAL_KEYBOARD", ""),
        )
