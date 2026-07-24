"""Pure tests for the persistent deployment command gate."""

from __future__ import annotations

import unittest

from deploy.common.session_control import SessionControl


class SessionControlTest(unittest.TestCase):
    def _running_control(self) -> tuple[SessionControl, int]:
        control = SessionControl()
        control.mark_enabled()
        generation = control.begin_start()
        self.assertTrue(control.finish_start(generation, "pick up the water bottle"))
        return control, generation

    def test_stop_invalidates_an_inflight_action_before_publish(self) -> None:
        control, generation = self._running_control()
        published: list[str] = []
        self.assertTrue(control.pause())
        self.assertFalse(control.publish_if_current(generation, lambda: published.append("action")))
        self.assertEqual(published, [])

    def test_continue_uses_a_new_generation_and_resets_step_budget(self) -> None:
        control, generation = self._running_control()
        control.record_published_step(generation, max_steps=1)
        paused = control.snapshot()
        self.assertEqual(paused.mode, "paused")

        pending = control.begin_continue(max_steps=1)
        self.assertIsNotNone(pending)
        task, generation = pending
        self.assertEqual(task, "pick up the water bottle")
        self.assertTrue(control.finish_continue(generation))
        resumed = control.snapshot()
        self.assertEqual(resumed.mode, "running")
        self.assertEqual(resumed.step, 0)
        self.assertNotEqual(resumed.generation, paused.generation)

    def test_exit_prevents_a_transition_from_reactivating_the_session(self) -> None:
        control = SessionControl()
        control.mark_enabled()
        generation = control.begin_start()
        control.request_exit()
        self.assertFalse(control.finish_start(generation, "ignored"))
        self.assertEqual(control.snapshot().mode, "exiting")

    def test_repeated_enable_does_not_pause_a_running_task(self) -> None:
        control, _ = self._running_control()
        self.assertTrue(control.mark_enabled())
        self.assertEqual(control.snapshot().mode, "running")

    def test_stale_step_does_not_modify_a_new_task(self) -> None:
        control, old_generation = self._running_control()
        new_generation = control.begin_start()
        self.assertTrue(control.finish_start(new_generation, "put the bottle in the box"))
        self.assertFalse(control.record_published_step(old_generation, max_steps=0))
        self.assertEqual(control.snapshot().step, 0)


if __name__ == "__main__":
    unittest.main()
