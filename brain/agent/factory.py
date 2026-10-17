"""Explicit executor selection: jev-flash (default), direct, or legacy-dual."""
from brain.agent.jev_flash import JevFlashExecutor
from brain.agent.executor import ModelExecutor
from brain.agent.orchestrator import ModelPlanner, Orchestrator


def build_executor(cfg, token, cleanup, status):
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
    raise ValueError('Unknown EXECUTOR; use jev-flash, direct or legacy-dual')
