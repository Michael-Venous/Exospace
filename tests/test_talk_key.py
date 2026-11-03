"""Home as push-to-talk: Home's presses come back, every other key goes on to the computer, repeats are ignored."""

import os
import unittest
from unittest.mock import patch

from brain.voice import talk_key
from brain.voice.talk_key import EV_KEY, EV_SYN, EVENT, PassThrough


def key(code, value):
    return EVENT.pack(0, 0, EV_KEY, code, value) + EVENT.pack(0, 0, EV_SYN, 0, 0)


class TalkKeyTests(unittest.TestCase):
    def test_home_is_kept_and_other_keys_are_passed_on(self):
        keys = PassThrough.__new__(PassThrough)  # no real keyboards or /dev/uinput in a test
        out_r, keys.virtual = os.pipe()
        self.addCleanup(lambda: (os.close(out_r), os.close(keys.virtual)))
        home = keys.forward(key(30, 1) + key(102, 1) + key(30, 0) + key(71, 0))
        self.assertEqual(home, [(102, 1), (71, 0)])
        passed = os.read(out_r, 4096)
        codes = [EVENT.unpack_from(passed, i)[3] for i in range(0, len(passed), EVENT.size)
                 if EVENT.unpack_from(passed, i)[2] == EV_KEY]
        self.assertEqual(codes, [30, 30])  # the "a" press and release, never Home

    def test_watch_reports_press_and_release_and_skips_repeats(self):
        real_r, real_w = os.pipe()
        virtual_r, virtual_w = os.pipe()
        self.addCleanup(lambda: [os.close(fd) for fd in (real_w, virtual_r, virtual_w)])

        class Keys(PassThrough):
            def __init__(self, paths):
                self.real, self.virtual = [real_r], virtual_w

            def close(self):
                os.close(real_r)

        class Enough(Exception):
            pass

        seen = []

        def changed(down):  # runs in line; the second event ends watch, which closes its keyboard
            seen.append(down)
            if len(seen) == 2:
                raise Enough

        os.write(real_w, key(102, 1) + key(102, 2) + key(102, 2) + key(102, 0))
        with patch.object(talk_key, "PassThrough", Keys), patch.object(talk_key, "keyboards", lambda: []):
            with self.assertRaises(Enough):
                talk_key.watch(changed)
        self.assertEqual(seen, [True, False])

if __name__ == "__main__":
    unittest.main()
