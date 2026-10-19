"""Explicit executor selection. Default `race`: Jev triage racing the Jev + language-model planner,
with or without ZAI_API_KEY; `jev` is race with Jev's steps only (Flash only for text Jev can't write, no deep hand-off, fastest);
`direct` is one model call per step."""
from brain.agent.jev_flash import JevFlashExecutor
from brain.agent.race import RaceExecutor
from brain.agent.executor import ModelExecutor
from brain.agent.orchestrator import ModelPlanner, Orchestrator
from brain.agent.deep import DeepFallback
from brain.agent.replay import ReplayCache


def build_executor(cfg, token, cleanup, status):
    executor = _build(cfg, token, cleanup, status)
    if cfg.executor == 'race' and cfg.deep_model and cfg.gemini_api_key:
        executor = DeepFallback(executor, cfg, token, status)
    return ReplayCache(executor, cfg.replay_path) if cfg.replay_path else executor


def _build(cfg, token, cleanup, status):
    if cfg.executor in ('race', 'jev'):
        executor = RaceExecutor(cfg, token)
        cleanup.callback(executor.close)
        return executor
    if cfg.executor == 'jev-flash':
        executor = JevFlashExecutor(cfg, token)
        cleanup.callback(executor.close)
        return executor
    if cfg.executor == 'direct':
        executor = ModelExecutor(cfg, token, routing=False)
        cleanup.callback(executor.close)
        return executor
    if cfg.executor == 'legacy-dual':
        fast = ModelExecutor(cfg, token)
        cleanup.callback(fast.close)
        planner = ModelPlanner(cfg, token)
        cleanup.callback(planner.close)
        return Orchestrator(fast, planner, token, status)
    raise ValueError('Unknown EXECUTOR; use race, jev, jev-flash, direct or legacy-dual')
