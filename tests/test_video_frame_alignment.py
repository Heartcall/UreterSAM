from __future__ import annotations

import unittest

from tools.video_frames import read_forward


class SequentialFrameTests(unittest.TestCase):
    def test_gaps_are_decoded_not_seeked(self):
        class Capture:
            index = 0
            def read(self):
                frame = self.index
                self.index += 1
                return frame < 10, frame
            def set(self, *_args):
                raise AssertionError("Random seek would invalidate decode-order identity")
        capture = Capture()
        first, next_index = read_forward(capture, 0, 3)
        second, next_index = read_forward(capture, next_index, 7)
        self.assertEqual((first, second, next_index, capture.index), (3,7,8,8))
        with self.assertRaisesRegex(ValueError, "revisit"):
            read_forward(capture, next_index, 6)
        with self.assertRaisesRegex(RuntimeError, "frame 10"):
            read_forward(capture, next_index, 12)


if __name__ == "__main__":
    unittest.main()
