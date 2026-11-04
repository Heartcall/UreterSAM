import unittest
from pathlib import Path

from tools.freeze_video_reference_supplement import approved_semantic_decisions, within


class VideoReferenceSupplementTests(unittest.TestCase):
    def test_only_approved_whole_frame_semantic_references_enter_supplement(self):
        manifest = {'records': [{'id': 'a', 'kind': 'semantic'}, {'id': 'b', 'kind': 'semantic'}, {'id': 'c', 'kind': 'tip'}]}
        approved = {'status': 'approved', 'reviewer': 'human', 'independent_review': True, 'whole_frame_reviewed': True}
        manifest['records'][0].update(video_id='video_1', frame_index=10)
        state = {'decisions': {'a': approved, 'b': {'status': 'pending'}, 'c': {'status': 'approved'}}}
        self.assertEqual([r['id'] for r, _ in approved_semantic_decisions(manifest, state)], ['a'])
        with self.assertRaises(ValueError):
            approved_semantic_decisions(manifest, {'decisions': {'a': {**approved, 'whole_frame_reviewed': False}}})

    def test_existing29_and_unknown_ids_cannot_be_relabelled_as_new(self):
        row = {'id': 'a', 'kind': 'semantic', 'reference_mask': 'frozen.png'}
        with self.assertRaises(ValueError): approved_semantic_decisions({'records': [row]}, {'decisions': {'a': {'status': 'approved'}}})
        with self.assertRaises(ValueError): approved_semantic_decisions({'records': []}, {'decisions': {'a': {'status': 'approved'}}})

    def test_revision_paths_cannot_escape_session(self):
        base = Path(__file__).resolve().parent
        self.assertEqual(within(base, 'revisions/mask.png'), base / 'revisions/mask.png')
        with self.assertRaises(ValueError): within(base, '../outside.png')


if __name__ == '__main__': unittest.main()
