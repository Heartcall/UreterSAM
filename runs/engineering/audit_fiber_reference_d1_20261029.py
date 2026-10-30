"""Independent D1 sampling, provenance, existing-output and local-API checks."""
from collections import Counter
import base64
import json
from pathlib import Path
import urllib.request
import urllib.error

import numpy as np

from tools.fiber_reference_workspace import DEFAULT, ROOT, read, sha256
from ureter_semis.experiment import atomic_json

checks = []
def check(name, passed, detail=None):
    checks.append({'name': name, 'passed': bool(passed), 'detail': detail})
    if not passed: raise AssertionError(name)

manifest = read(DEFAULT / 'manifest.json'); records = manifest['records']
semantic = [r for r in records if r['kind'] == 'semantic']
tip = [r for r in records if r['kind'] == 'tip']
check('2694_semantic_12_tip_records_and_unique_ids', len(semantic) == 2694 and len(tip) == 12 and len({r['id'] for r in records}) == 2706)
check('frozen_sources_and_inputs_unchanged', all(sha256(ROOT / p) == h for p, h in manifest['sources'].items()) and all(sha256(p) == h for p, h in manifest['inputs'].items()))
check('all_validation_images_match_saved_file_identities', all(sha256(r['image']) == r['image_sha256'] for r in semantic))
for vid, count in [('video_1', 722), ('video_2', 1972)]:
    rows = [r for r in semantic if r['video_id'] == vid]
    logs = [json.loads(s) for s in (ROOT / f'runs/a1_fiber_tip_d0_20261029/full_video/{vid}/frames.jsonl').read_text().splitlines()]
    check(vid + '_every_frame_raw_rgb_identity_matches_D0', len(rows) == count and all(r['frame_index'] == i and r['rgb_raw_bytes_sha256'] == logs[i]['rgb_raw_bytes_sha256'] for i, r in enumerate(rows)))
    rng = np.random.default_rng(manifest['sampling']['seed_by_video'][vid])
    expected = set(int(i) for layer in np.array_split(np.arange(count), 12) for i in rng.choice(layer, size=5, replace=False))
    selected = [r for r in rows if r['sampling_selected']]
    check(vid + '_fixed_model_independent_60_frame_sample', len(selected) == 60 and set(r['frame_index'] for r in selected) == expected and all(r['sampling_weight'] == r['stratum_N'] / 5 for r in selected))
    check(vid + '_12_disjoint_time_strata_five_samples_each', set(Counter(r['stratum_id'] for r in selected).values()) == {5} and len({r['stratum_id'] for r in selected}) == 12)
old = [r for r in semantic if r.get('reference_mask')]
check('all29_original_masks_immutable', len(old) == 29 and all(sha256(r['reference_mask']) == r['reference_mask_sha256'] for r in old))
check('one_random_sample_has_existing_reference', sum(r['sampling_selected'] for r in old) == 1)
session_path = DEFAULT / 'session/session.json'; before = sha256(session_path)
session = read(session_path)
check('no_annotations_generated_by_agent', not session['decisions'])
summary = read(DEFAULT / 'initial_accuracy_audit/summary.json')
check('all11_schemes_have2694_valid_outputs', len(summary['models']) == 11 and all(m['prediction_complete'] and m['prediction_frames'] == 2694 and not m['failures'] for m in summary['models'].values()))
frame_rows = 0
for name, model in summary['models'].items():
    rows = [json.loads(line) for line in Path(model['all_frame_output_identity_log']).read_text().splitlines()]
    check(name + '_no_unlabelled_frame_accuracy', len(rows) == 2694 and sum(r['metrics'] is not None for r in rows) == 29 and sha256(model['all_frame_output_identity_log']) == model['identity_log_sha256'])
    frame_rows += len(rows)
check('all_dense_and_incomplete_probability_estimates_remain_null', all(m['dense_exact_accuracy'] is None and m['tip_metrics'] is None and m['stratified_pooled_estimate']['metrics'] is None and all(c['dense_exact_accuracy'] is None and c['stratified_whole_video_estimate']['metrics'] is None for c in m['cases'].values()) for m in summary['models'].values()))
historical = read(ROOT / 'runs/a1_fiber_tip_d0_20261029/video_native/summary.json')['models']['classwise']['metrics']
current = summary['models']['classwise']['sparse_video29']['metrics']
check('baseline29_metrics_exactly_match_frozen_history', all(current[c][metric] == historical[c + '_' + metric] for c in ('stone', 'fiber') for metric in ('tp', 'fp', 'fn', 'dice', 'precision', 'recall')))
d0 = read(ROOT / 'runs/a1_fiber_tip_d0_20261029/preflight.json')
check('D0_all_frozen_sources_still_unchanged', all(sha256(ROOT / p) == h for p, h in d0['sources'].items()))

def request(path, payload=None, origin=None):
    headers = {}
    if payload is not None: headers['Content-Type'] = 'application/json'
    if origin: headers['Origin'] = origin
    req = urllib.request.Request('http://127.0.0.1:8768' + path,
        data=json.dumps(payload).encode() if payload is not None else None, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            return response.status, response.read(), dict(response.headers)
    except urllib.error.HTTPError as error:
        return error.code, error.read(), dict(error.headers)

status, data, headers = request('/')
check('local_page_CSP_and_no_cache', status == 200 and 'frame-ancestors' in headers['Content-Security-Policy'] and headers['Cache-Control'] == 'no-store')
status, data, _ = request('/api/items?kind=semantic')
items = json.loads(data)['items']
check('UI120_samples_with_one_readonly_reference', status == 200 and len(items) == 120 and sum(r['readonly'] for r in items) == 1)
status, data, _ = request('/api/item?id=tip:video_1:2'); item = json.loads(data)
check('tip_UI_starts_all_unknown_without_predicted_points', status == 200 and item['status'] == 'pending' and set(base64.b64decode(item['mask_base64'])) == {255} and item.get('tip_roi') is None and item.get('tip_point') is None)
status, _, _ = request('/api/save', {'id': 'tip:video_1:2', 'revision': -1})
check('stale_revision_rejected409', status == 409)
status, _, _ = request('/api/save', {'id': 'semantic:video_1:77', 'revision': 0})
check('original_reference_write_rejected400', status == 400)
status, _, _ = request('/api/save', {'id': 'tip:video_1:2', 'revision': 0, 'status': 'approved', 'reviewer': 'synthetic_test',
    'independent_review': True, 'mask_base64': item['mask_base64']})
check('all_unknown_approval_rejected400', status == 400)
status, _, _ = request('/api/items?kind=tip', origin='https://example.invalid')
check('foreign_origin_rejected403', status == 403)
check('API_checks_wrote_no_reference_revisions', sha256(session_path) == before and not read(session_path)['decisions'])
result = {'checks': checks, 'passed': all(r['passed'] for r in checks), 'count': len(checks),
          'prediction_identity_records': frame_rows, 'new_human_references_created': 0,
          'source_inputs_and_29_labels_unchanged': True, 'tip_metrics': None, 'dense_accuracy': None,
          'sampled_estimates': None, 'training_updates': 0, 'reserved_test_pixels_read': 0,
          'protocol_sha256': sha256(ROOT / 'docs/evaluation/fiber_tip_and_full_video_sampling_20261029.md')}
atomic_json(DEFAULT / 'engineering_checks.json', result)
print(json.dumps({'passed': result['passed'], 'count': len(checks), 'prediction_identity_records': frame_rows}))
