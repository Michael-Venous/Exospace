"""Compact offline demo gate: python -m bench.demo_smoke.

Exercises real routing, runtime and browser code with deterministic model/HID doubles.
This does not prove live model, network, ASR or hardware performance.
"""
import unittest

CASES = {
    'tests.test_race.RaceTests': (
        'test_a_direct_click_never_starts_a_plan',
        'test_the_planner_runs_once_when_triage_asks_for_a_plan',
        'test_a_launch_needs_no_plan_and_the_rest_is_planned_on_the_app',
        'test_launch_completion_keeps_both_gates_in_one_call',
        'test_unchanged_launch_cannot_be_claimed_done',
        'test_discarded_launch_retriages_fresh_screen_and_preserves_approval',
        'test_irreversible_request_waits_for_approval',
        'test_pending_approval_is_dropped_after_a_device_swap',
        'test_correction_after_a_draft_request_asks_before_the_send',
        'test_a_scroll_with_no_effect_goes_to_the_planner',
    ),
    'tests.test_runtime.RuntimeTests': (
        'test_words_said_mid_run_join_the_request_at_the_next_step',
        'test_cancelled_planner_result_is_discarded',
        'test_batch_stops_before_a_click_aimed_at_a_changed_screen',
        'test_blank_screen_ends_the_run_without_resume',
    ),
    'tests.test_voice_browser.VoiceBrowserTests': (
        'test_final_clauses_are_sent_while_the_microphone_is_still_recording',
        'test_commands_wait_for_the_previous_response',
        'test_server_voiced_lines_are_not_spoken_twice',
        'test_stalled_stream_cannot_block_the_next_press',
    ),
}


def main():
    names = [f'{cls}.{method}' for cls, methods in CASES.items() for method in methods]
    names.append('tests.test_cancel_nested')
    suite = unittest.defaultTestLoader.loadTestsFromNames(names)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() and not result.skipped else 1


if __name__ == '__main__':
    raise SystemExit(main())
