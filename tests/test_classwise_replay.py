import unittest
import numpy as np
import torch
from PIL import Image

from tools.replay_classwise_experts import intervals, mask_diagnostics, pil_input, presence_runs, summarize_diagnostics
from ureter_semis.data import _tensor


class ClasswiseReplayTests(unittest.TestCase):
    def test_exact_inclusive_user_interval_mapping(self):
        protocol = {'user_events_one_based_inclusive': {'v': {'occlusion': [[330, 722]], 'wall': [[33, 37]]}}}
        result = intervals(protocol, 'v', 722)
        self.assertEqual((result[0]['start_frame'], result[0]['end_frame_exclusive'], result[0]['frames']), (329, 722, 393))
        self.assertEqual((result[1]['start_frame'], result[1]['end_frame_exclusive'], result[1]['frames']), (32, 37, 5))
        with self.assertRaises(ValueError): intervals(protocol, 'v', 721)

    def test_decoded_bgr_pil_tensor_matches_validation(self):
        rgb = np.arange(13 * 17 * 3, dtype=np.uint8).reshape(13, 17, 3)
        actual = pil_input(rgb[:, :, ::-1].copy(), 8, torch.device('cpu'))
        expected = _tensor(Image.fromarray(rgb).resize((8, 8), Image.Resampling.BILINEAR)).unsqueeze(0)
        self.assertTrue(torch.equal(actual, expected))

    def test_interval_runs_and_differences_stay_inside_selected_window(self):
        arrays = [np.ones((2, 2), dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8), np.ones((2, 2), dtype=np.uint8)]
        rows = []; previous = None
        for i, mask in enumerate(arrays):
            rows.append({'frame_index': 32 + i, **mask_diagnostics(mask, np.ones((2, 2), bool), previous)})
            previous = mask
        result = summarize_diagnostics(rows, 30)
        self.assertEqual(result['stone']['nonempty_to_empty'], 1)
        self.assertEqual(result['stone']['empty_to_nonempty'], 1)
        self.assertEqual(result['large_stone_region_candidate_frames'], 2)
        self.assertEqual(result['uncompensated_label_change_fraction']['mean'], 1)
        self.assertEqual(result['stone']['presence_runs'][-1]['end_frame_exclusive'], 35)
        self.assertEqual(summarize_diagnostics(rows[1:], 30)['stone']['nonempty_to_empty'], 0)
        self.assertIsNone(summarize_diagnostics(rows[:1], 30)['uncompensated_label_change_fraction'])
        self.assertNotIn('accuracy', result)
        self.assertEqual(presence_runs([], 'stone'), [])


if __name__ == '__main__': unittest.main()
