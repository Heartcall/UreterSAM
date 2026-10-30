"""Read-only scientific diagnostics; run after isolated timing workers finish."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tools.fiber_reference_workspace import ROOT, read, load_mask, sha256
from tools.fiber_blue_soft_prior import blue_evidence
from tools.run_fiber_blue_aplus import cached_paths, frame_read
from ureter_semis.experiment import atomic_json

out = ROOT / 'runs/a1_fiber_blue_aplus_20261029'
if read(out / 'job.json')['phase'] != 'completed': raise RuntimeError('Wait until all timed replays complete')
config = read(ROOT / 'configs/fiber_blue_aplus_20261029.json')
rows = read(ROOT / config['video_manifest'])['val']
old = read(ROOT / 'datasets/ureter_semis_kidney_joint_20261028/video_validation_manifest.json')['val']
indices = {(r['video_id'], r['frame_index']): i for i, r in enumerate(old)}
lookup = {(r['video_id'], r['frame_index']): i for i, r in enumerate(rows)}
font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 18)
small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 15)

def label_overlay(rgb, mask):
    result = rgb.copy()
    for c, color in [(1, (240, 70, 70)), (2, (30, 220, 245)), (255, (170, 110, 180))]:
        a = mask == c; result[a] = (.45 * rgb[a] + .55 * np.asarray(color)).astype(np.uint8)
    return result

titles = ['原始RGB', '人工参考（紫=未知）', '原网络光纤概率', '蓝色软证据（未标定）', '原classwise', 'A+新增：绿TP/红FP/紫未知']
outputs = {}
for vid, frames in {'video_1': [32, 72, 329, 525], 'video_2': [1000, 1208, 1223, 1846, 1881, 1965]}.items():
    cell, rowheight, top = 240, 270, 70
    board = Image.new('RGB', (6 * cell, top + len(frames) * rowheight + 35), (18, 25, 34))
    draw = ImageDraw.Draw(board)
    draw.text((10, 8), f'{vid} | 固定失败/新增帧诊断，非尖端真值 | 网络与蓝色分别检查；A+没有几何填充', font=font, fill='white')
    for j, title in enumerate(titles): draw.text((j * cell + 5, 42), title, font=small, fill='white')
    for k, index in enumerate(frames):
        i = lookup[(vid, index)]; row = rows[i]; bgr = frame_read(row['image']); rgb = bgr[:, :, ::-1].copy()
        truth = load_mask(row['mask'], bgr.shape[:2], {0, 1, 2, 255})
        cache, _ = cached_paths(row, i, True, indices)
        if cache is None: cache = out / f'video_native/new_probabilities/{i:04d}.npz'
        with np.load(cache) as data: p = data['a0_native'][2].copy()
        blue = blue_evidence(bgr, config['refinement'])['blue_score_uncalibrated']
        raw = load_mask(out / f'video_native/raw/{i:04d}.png', truth.shape, {0, 1, 2})
        pred = load_mask(out / f'video_native/blue_aplus/{i:04d}.png', truth.shape, {0, 1, 2})
        extra = (pred == 2) & (raw != 2); last = label_overlay(rgb, raw)
        last[extra & (truth == 2)] = (20, 255, 20); last[extra & (truth != 2) & (truth != 255)] = (255, 0, 80)
        last[extra & (truth == 255)] = (180, 80, 220)
        heat = lambda a: cv2.cvtColor(cv2.applyColorMap(np.uint8(np.clip(a, 0, 1) * 255), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
        panels = [rgb, label_overlay(rgb, truth), heat(p), heat(blue), label_overlay(rgb, raw), last]
        for j, array in enumerate(panels): board.paste(Image.fromarray(array).resize((cell, cell), Image.Resampling.BILINEAR), (j * cell, top + k * rowheight))
        tp = int((extra & (truth == 2)).sum()); fp = int((extra & (truth != 2) & (truth != 255)).sum())
        draw.text((5, top + k * rowheight + cell + 3), f'第{index+1}帧 / 零基{index} | 新增TP {tp}，FP {fp}，原光纤删除0', font=small, fill='white')
    draw.text((10, board.height - 28), '所有静态202样本均计入指标；图为指定诊断帧。热图0低→1高；蓝色证据不是光纤身份或解剖尖端真值。', font=small, fill=(190, 200, 210))
    path = out / f'{vid}_diagnostic_board.jpg'; board.save(path, quality=95)
    outputs[str(path)] = {'frames_zero_based': frames, 'sha256': sha256(path)}
atomic_json(out / 'visualization_identity.json', {'source_sha256': sha256(Path(__file__)), 'outputs': outputs})
print(json.dumps({'boards': list(outputs)}))
