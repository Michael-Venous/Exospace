"""Explicit executor selection. Default `race`: Jev triage racing the Jev + Flash planner."""
from brain.agent.jev_flash import JevFlashExecutor
from brain.agent.race import RaceExecutor
from brain.agent.executor import GeminiExecutor
from brain.agent.orchestrator import GeminiPlanner, Orchestrator


def build_executor(cfg, token, cleanup, status):
    if cfg.executor == 'race':
        executor = RaceExecutor(cfg, token)
        cleanup.callback(executor.close)
        return executor
    if cfg.executor == 'jev-flash':
        executor = JevFlashExecutor(cfg, token)
        cleanup.callback(executor.close)
        return executor
    if cfg.executor == 'legacy-dual':
        fast = GeminiExecutor(cfg, token)
        cleanup.callback(fast.close)
        planner = GeminiPlanner(cfg, token)
        cleanup.callback(planner.close)
        return Orchestrator(fast, planner, token, status)
    raise ValueError('Unknown EXECUTOR; use race, jev-flash or legacy-dual')
