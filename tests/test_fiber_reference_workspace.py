import base64
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from ureter_semis.experiment import atomic_json, sha256
from tools.fiber_reference_evaluation import confusion, metrics, stratified_estimate, tip_metrics
from tools.fiber_reference_workspace import ReferenceStore, validate_reference
from tools.review_store import ConflictError


class ReferenceTests(unittest.TestCase):
    def test_tip_unknown_and_visible_shaft_are_not_precision_negatives(self):
        ref = np.array([[1, 0, 2, 255]], np.uint8)
        result = tip_metrics(np.array([[2, 2, 2, 2]], np.uint8), ref, [0, 0, 4, 1])
        self.assertEqual((result['tp'], result['fp'], result['fn']), (1, 1, 0))
        self.assertEqual(result['precision'], .5)
        self.assertEqual(metrics(confusion(np.array([[2, 2]], np.uint8), np.array([[2, 255]], np.uint8)))['fiber']['fp'], 0)

    def test_no_estimate_from_convenience_frames_or_partial_sample(self):
        rows = [{'id': str(i), 'sampling_selected': i < 2, 'stratum_id': 'h', 'stratum_n': 2, 'stratum_N': 4} for i in range(4)]
        observation = {'confusion_matrix': [[3, 0, 0], [0, 1, 0], [0, 0, 2]], 'unknown_pixels': 1,
                       'boundaries': {str(c): {'predicted_boundary': 1, 'target_boundary': 1, 'matched_predicted': 1, 'matched_target': 1} for c in (1, 2)}}
        self.assertIsNone(stratified_estimate(rows, {'0': observation, '2': observation})['metrics'])
        full = stratified_estimate(rows, {'0': observation, '1': observation, '2': observation})
        self.assertEqual(full['metrics']['fiber']['tp'], 8)
        self.assertEqual(full['metrics']['estimated_unknown_pixels'], 4)
        self.assertEqual(full['metrics']['valid_pixels'], 24)
        self.assertEqual(full['intervals']['fiber']['recall']['approximate_95_interval'], [1, 1])

    def test_approval_does_not_invent_hidden_tip_or_accept_empty_truth(self):
        row = {'kind': 'tip', 'width': 3, 'height': 2}
        base = {'status': 'approved', 'reviewer': 'human', 'independent_review': True,
                'visibility': 'visible', 'tip_roi': [0, 0, 3, 2], 'tip_point': [1, 0], 'roi_reviewed': True}
        mask = np.array([[0, 1, 255], [0, 2, 255]], np.uint8)
        self.assertTrue(validate_reference(row, base, mask)['tip_metrics_eligible'])
        for changed in ({'visibility': 'occluded'}, {'tip_roi': None}, {'reviewer': ''}, {'independent_review': False}):
            with self.assertRaises(ValueError): validate_reference(row, {**base, **changed}, mask)
        with self.assertRaises(ValueError): validate_reference(row, base, np.full((2, 3), 255, np.uint8))
        with self.assertRaises(ValueError): validate_reference({'kind': 'semantic'}, base, mask)

    def test_revision_conflict_readonly_original_and_identity_checks(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'runs/engineering') as folder:
            root = Path(folder); image = root / 'image.png'; original = root / 'original.png'
            Image.new('RGB', (3, 2), (100, 100, 100)).save(image)
            Image.fromarray(np.zeros((2, 3), np.uint8)).save(original)
            row = {'id': 'semantic:video_1:0', 'kind': 'semantic', 'video_id': 'video_1', 'frame_index': 0,
                   'width': 3, 'height': 2, 'image': str(image), 'image_sha256': sha256(image), 'sampling_selected': True}
            frozen = {**row, 'id': 'semantic:video_1:1', 'reference_mask': str(original),
                      'reference_mask_sha256': sha256(original), 'reference_reviewer': 'old'}
            atomic_json(root / 'manifest.json', {'records': [row, frozen]})
            store = ReferenceStore(root / 'manifest.json', root / 'session')
            self.assertTrue(np.all(np.frombuffer(base64.b64decode(store.item(row['id'])['mask_base64']), np.uint8) == 255))
            payload = {'revision': 0, 'mask_base64': base64.b64encode(bytes(6)).decode(), 'status': 'pending'}
            self.assertEqual(store.save(row['id'], payload)['revision'], 1)
            with self.assertRaises(ConflictError): store.save(row['id'], payload)
            with self.assertRaises(ValueError): store.save(frozen['id'], payload)
            self.assertEqual(sha256(original), frozen['reference_mask_sha256'])
            refs = json.loads((root / 'session/references.json').read_text())['records']
            self.assertFalse(refs[0]['eligible']); self.assertTrue(refs[1]['eligible'])
            Image.new('RGB', (3, 2), (0, 0, 0)).save(image)
            with self.assertRaises(ValueError): store.item(row['id'])


if __name__ == '__main__': unittest.main()
