"""Independent, revisioned D1 tip and dense-video reference workspace (no models)."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
import mimetypes
from pathlib import Path
import threading
from urllib.parse import parse_qs, urlparse
import uuid
import zipfile

import cv2
import numpy as np
from PIL import Image

from ureter_semis.experiment import atomic_json, sha256 as path_sha256
from .review_store import ConflictError, decode_array, encode_array

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'runs/fiber_reference_d1_20261029'


def sha256(path):
    return path_sha256(Path(path))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_mask(path, shape, classes):
    with Image.open(path) as image:
        mask = np.asarray(image).copy()
    if mask.shape != shape or mask.dtype != np.uint8 or not set(np.unique(mask)) <= set(classes):
        raise ValueError('Invalid native reference mask')
    return mask


def check_point(point, width, height):
    if point is None:
        return None
    if (not isinstance(point, list) or len(point) != 2 or
            any(type(x) is not int for x in point) or not 0 <= point[0] < width or not 0 <= point[1] < height):
        raise ValueError('Point must be an integer [x,y] inside the native image')
    return point


def check_roi(roi, width, height):
    if roi is None:
        return None
    if (not isinstance(roi, list) or len(roi) != 4 or any(type(x) is not int for x in roi) or
            not 0 <= roi[0] < roi[2] <= width or not 0 <= roi[1] < roi[3] <= height):
        raise ValueError('ROI must be [x0,y0,x1,y1], upper bounds exclusive')
    return roi


def validate_reference(row, payload, mask, context=None):
    """Human approval is explicit; pending pixels never become background truth."""
    status = payload.get('status', 'pending')
    if status not in {'pending', 'approved'}:
        raise ValueError('Invalid status')
    if not set(np.unique(mask)) <= {0, 1, 2, 255}:
        raise ValueError('Unsupported reference labels')
    reviewer = str(payload.get('reviewer', '')).strip()
    if status == 'approved' and (not reviewer or payload.get('independent_review') is not True):
        raise ValueError('Approval requires reviewer and explicit independent visible-pixel review')
    if status == 'approved' and not (mask != 255).any():
        raise ValueError('An all-unknown mask cannot be approved')
    result = {'status': status, 'reviewer': reviewer,
              'independent_review': payload.get('independent_review') is True,
              'note': str(payload.get('note', ''))[:10000], 'annotation_basis': 'human_visible_pixels',
              'unknown_pixels': int((mask == 255).sum()), 'valid_pixels': int((mask != 255).sum())}
    if row['kind'] == 'semantic':
        if status == 'approved' and payload.get('whole_frame_reviewed') is not True:
            raise ValueError('Semantic approval requires whole-frame review, not a tip-only ROI')
        result['whole_frame_reviewed'] = payload.get('whole_frame_reviewed') is True
        return result
    visibility = payload.get('visibility', 'unreviewed')
    if visibility not in {'unreviewed', 'visible', 'partial_occlusion', 'occluded', 'out_of_view', 'uncertain'}:
        raise ValueError('Invalid tip visibility')
    roi = check_roi(payload.get('tip_roi'), row['width'], row['height'])
    tip = check_point(payload.get('tip_point'), row['width'], row['height'])
    entry = check_point(payload.get('entry_point'), row['width'], row['height'])
    inside = np.zeros(mask.shape, bool)
    if roi:
        x0, y0, x1, y1 = roi
        inside[y0:y1, x0:x1] = True
    if ((mask == 1) & ~inside).any():
        raise ValueError('Visible tip pixels must lie inside the manually defined tip ROI')
    visible = visibility in {'visible', 'partial_occlusion'}
    if status == 'approved':
        if visibility == 'unreviewed':
            raise ValueError('Review tip visibility first')
        if visible:
            if not roi or not tip or not inside[tip[1], tip[0]] or not (mask == 1).any():
                raise ValueError('Visible tip needs a manual ROI, visible tip pixels and a distal point inside the ROI')
            if not (inside & (mask == 0)).any() or payload.get('roi_reviewed') is not True:
                raise ValueError('Review the tip ROI, including confirmed non-fiber negatives for Precision')
        elif (mask == 1).any() or tip is not None:
            raise ValueError('Do not invent a visible tip or point when visibility is unresolved/hidden')
        extra = int(((mask == 1) & (context != 2) & (context != 255)).sum()) if context is not None else 0
        if extra and not result['note'].strip():
            raise ValueError('Explain disagreements with the frozen whole-fiber context; do not overwrite it')
    result.update(visibility=visibility, tip_roi=roi, tip_point=tip, entry_point=entry,
                  roi_reviewed=payload.get('roi_reviewed') is True,
                  tip_pixel_count=int((mask == 1).sum()), shaft_pixel_count=int((mask == 2).sum()),
                  tip_metrics_eligible=status == 'approved' and visible)
    return result


class ReferenceStore:
    def __init__(self, manifest, session):
        self.manifest_path = Path(manifest).resolve()
        manifest = read(self.manifest_path)
        self.records = {r['id']: r for r in manifest['records']}
        if len(self.records) != len(manifest['records']):
            raise ValueError('Duplicate reference ids')
        self.directory = Path(session).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.state_path = self.directory / 'session.json'
        identity = {'manifest_sha256': sha256(self.manifest_path)}
        if self.state_path.exists():
            self.state = read(self.state_path)
            if self.state['inputs'] != identity:
                raise ValueError('Manifest changed; use a new session')
        else:
            if any(self.directory.iterdir()):
                raise FileExistsError('Session directory is not empty')
            self.state = {'schema_version': 1, 'inputs': identity, 'decisions': {}}
            atomic_json(self.state_path, self.state)
        self.export()

    def verify(self, row):
        for field in ('image', 'context_mask', 'reference_mask'):
            if row.get(field) and sha256(row[field]) != row[field + '_sha256']:
                raise ValueError('Frozen input changed: ' + field)

    def decision(self, row):
        decision = self.state['decisions'].get(row['id'])
        if decision:
            path = self.directory / decision['mask']
            if sha256(path) != decision['mask_sha256']:
                raise ValueError('Saved annotation changed outside the revision store')
            return decision, path
        if row.get('reference_mask'):
            return {'status': 'approved', 'revision': 0, 'reviewer': row['reference_reviewer'],
                    'basis': 'inherited_frozen_video29', 'whole_frame_reviewed': True}, Path(row['reference_mask'])
        return {'status': 'pending', 'revision': 0}, None

    def items(self, kind, scope='sampled'):
        with self.lock:
            return [{'id': r['id'], 'video_id': r['video_id'], 'frame_index': r['frame_index'],
                     'status': self.decision(r)[0]['status'], 'readonly': bool(r.get('reference_mask'))}
                    for r in self.records.values() if r['kind'] == kind and
                    (kind == 'tip' or scope == 'all' or r.get('sampling_selected'))]

    def image(self, key):
        row = self.records[key]
        self.verify(row)
        return Path(row['image'])

    def item(self, key):
        with self.lock:
            row = self.records[key]
            self.verify(row)
            decision, path = self.decision(row)
            shape = (row['height'], row['width'])
            mask = load_mask(path, shape, {0, 1, 2, 255}) if path else np.full(shape, 255, np.uint8)
            context = load_mask(row['context_mask'], shape, {0, 1, 2, 255}) if row.get('context_mask') else None
            return {**row, **decision, 'readonly': bool(row.get('reference_mask')),
                    'mask_base64': encode_array(mask), 'context_base64': encode_array(context) if context is not None else None}

    def save(self, key, payload):
        with self.lock:
            row = self.records[key]
            self.verify(row)
            if row.get('reference_mask'):
                raise ValueError('Original 29 approved references are read-only')
            old, _ = self.decision(row)
            if payload.get('revision') != old['revision']:
                raise ConflictError('Stale revision; reload before saving')
            mask = decode_array(payload['mask_base64'], row['width'], row['height'], 'u1')
            context = load_mask(row['context_mask'], mask.shape, {0, 1, 2, 255}) if row.get('context_mask') else None
            result = validate_reference(row, payload, mask, context)
            revision = old['revision'] + 1
            folder = Path('revisions') / hashlib.sha256(key.encode()).hexdigest()[:20] / f'{revision:05d}_{uuid.uuid4().hex}'
            (self.directory / folder).mkdir(parents=True)
            path = self.directory / folder / 'mask.png'
            Image.fromarray(mask).save(path)
            result.update(revision=revision, mask=str(folder / 'mask.png'), mask_sha256=sha256(path),
                          image_sha256=row['image_sha256'], saved_utc=datetime.now(timezone.utc).isoformat())
            atomic_json(self.directory / folder / 'decision.json', result)
            state = {**self.state, 'decisions': {**self.state['decisions'], key: result}}
            atomic_json(self.state_path, state)
            self.state = state
            self.export()
            return {'revision': revision, 'status': result['status']}

    def export(self):
        records = []
        for row in self.records.values():
            decision, path = self.decision(row)
            records.append({**row, **decision, 'mask': str(path.resolve()) if path else None,
                            'mask_sha256': sha256(path) if path else None,
                            'eligible': decision['status'] == 'approved'})
        atomic_json(self.directory / 'references.json', {
            'schema_version': 1, 'manifest': str(self.manifest_path), **self.state['inputs'],
            'role': 'development_validation_only_never_training', 'records': records,
            'tip_label_schema': {'0': 'confirmed_non_fiber_in_tip_roi', '1': 'visible_tip',
                                 '2': 'visible_shaft_excluded_from_tip_precision', '255': 'unknown_not_background'},
            'semantic_label_schema': {'0': 'background', '1': 'stone', '2': 'visible_fiber', '255': 'unknown'},
            'predictions_used_as_ground_truth': False})


def prepare(output=DEFAULT):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    config_path = ROOT / 'configs/fiber_reference_d1_20261029.json'
    config = read(config_path)
    baseline = read(ROOT / 'runs/baselines/a0_20261027/config.json')
    video_manifest = ROOT / config['video_reference_manifest']
    references = {(r['video_id'], r['frame_index']): r for r in read(video_manifest)['val']}
    pilot = read(ROOT / config['pilot_template'])['records']
    pilots = {(r['video_id'], r['frame_index']): r for r in pilot}
    output.mkdir(parents=True)
    inputs = {str(config_path): sha256(config_path), str(video_manifest): sha256(video_manifest)}
    for path in [ROOT / config['pilot_template'], ROOT / 'runs/baselines/a0_20261027/config.json']:
        inputs[str(path)] = sha256(path)
    records, videos = [], []
    sampling = config['sampling']
    for video in baseline['inputs']:
        vid = video['video_id']
        if sha256(video['path']) != video['sha256']:
            raise ValueError('Video changed')
        log_path = ROOT / f'runs/a1_fiber_tip_d0_20261029/full_video/{vid}/frames.jsonl'
        logs = [json.loads(line) for line in log_path.read_text(encoding='utf-8').splitlines()]
        if len(logs) != video['frames']:
            raise ValueError('D0 identity log incomplete')
        inputs[str(log_path)] = sha256(log_path)
        rng = np.random.default_rng(sampling['seed_by_video'][vid])
        strata = np.array_split(np.arange(video['frames']), sampling['strata_per_video'])
        frame_strata = {}
        for h, indices in enumerate(strata):
            selected = set(map(int, rng.choice(indices, size=sampling['frames_per_stratum'], replace=False)))
            for index in indices:
                frame_strata[int(index)] = {'stratum_id': f'{vid}:{h:02d}', 'stratum_N': len(indices),
                    'stratum_n': len(selected), 'sampling_selected': int(index) in selected,
                    'sampling_weight': len(indices) / len(selected)}
        folder = output / 'images' / vid
        folder.mkdir(parents=True)
        cap = cv2.VideoCapture(video['path'])
        count = 0
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                raw_hash = hashlib.sha256(rgb.tobytes()).hexdigest()
                if count >= len(logs) or logs[count]['frame_index'] != count or logs[count]['rgb_raw_bytes_sha256'] != raw_hash:
                    raise ValueError(f'Sequential frame identity differs: {vid}:{count}')
                image = folder / f'{count:06d}.png'
                Image.fromarray(rgb).save(image)
                base = {'id': f'semantic:{vid}:{count}', 'kind': 'semantic', 'video_id': vid,
                        'patient_id': 'case_' + vid, 'frame_index': count, 'one_based_frame': count + 1,
                        'width': rgb.shape[1], 'height': rgb.shape[0], 'image': str(image),
                        'image_sha256': sha256(image), 'rgb_raw_bytes_sha256': raw_hash,
                        'coordinate_system': 'raw_source_pixels', 'training_allowed': False, **frame_strata[count]}
                reference = references.get((vid, count))
                if reference:
                    if sha256(reference['image']) != reference['image_sha256'] or sha256(reference['mask']) != reference['mask_sha256']:
                        raise ValueError('Original video29 reference changed')
                    with Image.open(reference['image']) as source:
                        if not np.array_equal(np.asarray(source.convert('RGB')), rgb):
                            raise ValueError('Original reference RGB is misaligned')
                    base.update(reference_mask=reference['mask'], reference_mask_sha256=reference['mask_sha256'],
                                reference_reviewer=reference['reviewer'])
                records.append(base)
                if (vid, count) in pilots:
                    if not reference:
                        raise ValueError('Tip pilot missing original whole-fiber context')
                    records.append({**{k: v for k, v in base.items() if not k.startswith('reference_')},
                                    'id': f'tip:{vid}:{count}', 'kind': 'tip',
                                    'context_mask': reference['mask'], 'context_mask_sha256': reference['mask_sha256']})
                count += 1
        finally:
            cap.release()
        if count != video['frames']:
            raise ValueError('Incomplete video decoding')
        videos.append({**video, 'decoded_frames': count, 'approved_semantic_frames': sum(v == vid for v, _ in references)})
        print(f'{vid}: {count} sequential frame identities verified', flush=True)
    if sum(r['kind'] == 'tip' for r in records) != 12 or len(references) != 29:
        raise ValueError('Reference scope changed')
    selected = [r for r in records if r['kind'] == 'semantic' and r['sampling_selected']]
    strata_report = []
    for stratum_id in sorted({r['stratum_id'] for r in selected}):
        rows = [r for r in records if r['kind'] == 'semantic' and r['stratum_id'] == stratum_id]
        strata_report.append({'stratum_id': stratum_id, 'population_frames': len(rows),
            'first_frame_zero_based': min(r['frame_index'] for r in rows),
            'last_frame_zero_based': max(r['frame_index'] for r in rows),
            'selected_frames_zero_based': [r['frame_index'] for r in rows if r['sampling_selected']],
            'selected_frames_one_based': [r['one_based_frame'] for r in rows if r['sampling_selected']],
            'weight': rows[0]['sampling_weight']})
    atomic_json(output / 'sampling_plan.json', {'design': sampling, 'population_frames': 2694,
                'sample_frames': len(selected), 'strata': strata_report, 'image_features_used_for_sampling': False})
    with (output / 'sampling_frames.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        fields = ['id', 'video_id', 'frame_index', 'one_based_frame', 'stratum_id', 'stratum_N', 'stratum_n', 'sampling_weight', 'image']
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(selected)
    source_paths = [Path(__file__), ROOT / 'tools/fiber_reference_evaluation.py',
                    ROOT / 'tests/test_fiber_reference_workspace.py', ROOT / 'tools/review_store.py',
                    ROOT / 'ureter_semis/experiment.py', ROOT / 'tools/evaluate_spatial_detail.py']
    source_paths += sorted((ROOT / 'tools/fiber_reference_ui').glob('*'))
    sources = {str(p.relative_to(ROOT)): sha256(p) for p in source_paths}
    atomic_json(output / 'manifest.json', {'schema_version': 1, 'role': 'development_validation_only',
                'frame_decoding': 'sequential_from_frame_zero_no_random_seek', 'videos': videos, 'records': records,
                'inputs': inputs, 'sources': sources, 'sampling': sampling, 'models_loaded': 0, 'gradient_updates': 0,
                'train_pixels_read': 0, 'reserved_test_pixels_read': 0,
                'original_video29_masks_changed': False, 'predictions_prefilled': False})
    with zipfile.ZipFile(output / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in sources:
            archive.write(ROOT / name, name.replace('\\', '/'))
        archive.write(config_path, str(config_path.relative_to(ROOT)))
    store = ReferenceStore(output / 'manifest.json', output / 'session')
    atomic_json(output / 'preparation.json', {'phase': 'awaiting_human_tip_reference',
                'tip_pilot_pending': 12, 'semantic_frames': 2694, 'semantic_approved': 29,
                'semantic_pending': 2665, 'sampled_frames': 120,
                'sampled_approved': sum(bool(r.get('reference_mask')) for r in records if r['kind'] == 'semantic' and r['sampling_selected']),
                'dense_accuracy_available': False, 'estimated_whole_video_accuracy_available': False, 'tip_metrics': None,
                'manifest_sha256': sha256(output / 'manifest.json'), 'source_snapshot_sha256': sha256(output / 'source_snapshot.zip')})
    return store


def create_server(store, port=8768):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, status, data, content_type='application/json; charset=utf-8'):
            if not isinstance(data, bytes):
                data = json.dumps(data, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            for key, value in {'Content-Type': content_type, 'Content-Length': str(len(data)),
                               'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff',
                               'Content-Security-Policy': "default-src 'self'; img-src 'self' blob:; style-src 'self'; script-src 'self'; frame-ancestors 'none'"}.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(data)

        def local(self):
            host = f'127.0.0.1:{self.server.server_port}'
            if self.headers.get('Host') != host or self.headers.get('Origin', 'http://' + host) != 'http://' + host:
                self.send(403, {'error': 'Local workspace requests only'})
                return False
            return True

        def do_GET(self):
            if not self.local():
                return
            request = urlparse(self.path)
            q = parse_qs(request.query)
            try:
                if request.path in {'/', '/app.js', '/style.css'}:
                    path = ROOT / 'tools/fiber_reference_ui' / ('index.html' if request.path == '/' else request.path[1:])
                    self.send(200, path.read_bytes(), mimetypes.guess_type(path)[0] + '; charset=utf-8')
                elif request.path == '/api/items':
                    self.send(200, {'items': store.items(q.get('kind', ['tip'])[0], q.get('scope', ['sampled'])[0])})
                elif request.path == '/api/item':
                    self.send(200, store.item(q['id'][0]))
                elif request.path == '/image':
                    self.send(200, store.image(q['id'][0]).read_bytes(), 'image/png')
                else:
                    self.send(404, {'error': 'Not found'})
            except (KeyError, ValueError, OSError) as error:
                self.send(400, {'error': str(error)})

        def do_POST(self):
            if not self.local():
                return
            try:
                if self.path != '/api/save':
                    return self.send(404, {'error': 'Not found'})
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length <= 4 * 1024 * 1024 or self.headers.get('Content-Type') != 'application/json':
                    raise ValueError('Invalid request format or size')
                payload = json.loads(self.rfile.read(length))
                self.send(200, store.save(payload['id'], payload))
            except ConflictError as error:
                self.send(409, {'error': str(error)})
            except (KeyError, TypeError, ValueError, OSError) as error:
                self.send(400, {'error': str(error)})
    return ThreadingHTTPServer(('127.0.0.1', port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'serve', 'export'])
    parser.add_argument('--workspace', type=Path, default=DEFAULT)
    parser.add_argument('--port', type=int, default=8768)
    args = parser.parse_args()
    store = prepare(args.workspace) if args.command == 'prepare' else ReferenceStore(args.workspace / 'manifest.json', args.workspace / 'session')
    if args.command == 'serve':
        server = create_server(store, args.port)
        print(f'Reference workspace: http://127.0.0.1:{server.server_port}/', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == '__main__':
    main()
