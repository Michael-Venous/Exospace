"""Explicit executor selection behind the model-free quick start. Default `race`: Jev triage racing the Jev + Flash
planner, with or without ZAI_API_KEY; `jev` is race with no language model; `direct` is one model call per step."""
from brain.agent.jev import JevVision
from brain.agent.jev_flash import JevFlashExecutor
from brain.agent.jev_only import JevOnlyPlanner
from brain.agent.race import RaceExecutor
from brain.agent.executor import ModelExecutor
from brain.agent.orchestrator import ModelPlanner, Orchestrator
from brain.agent.deep import DeepFallback
from brain.agent.quick import QuickStart
from brain.agent.replay import ReplayCache


def build_executor(cfg, token, cleanup, status):
    executor = _build(cfg, token, cleanup, status)
    if cfg.executor == 'race' and cfg.deep_model and cfg.gemini_api_key:
        executor = DeepFallback(executor, cfg, token, status)
    executor = QuickStart(executor)  # a repeat replays before quick.py; a first-time command acts with no model call
    return ReplayCache(executor, cfg.replay_path) if cfg.replay_path else executor


def _build(cfg, token, cleanup, status):
    if cfg.executor == 'race':
        executor = RaceExecutor(cfg, token)
        cleanup.callback(executor.close)
        return executor
    if cfg.executor == 'jev':
        jev = JevVision(cfg.typesafe_api_key, token, cfg.model_log, cfg.jev_model, cfg.jev_timeout_s,
                        cfg.jev_min_presence, cfg.jev_min_confidence)
        executor = RaceExecutor(cfg, token, JevOnlyPlanner(jev))
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
