"""TRAIN-only pooled fixed shape and color calibration, before any dev scores."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from ureter_semis.experiment import atomic_json,sha256
from .direct_fiber_template import train_geometry,appearance_hist

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/a1_direct_fiber_template_20261029'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def calibrate():
    OUT.mkdir(exist_ok=False);config_path=ROOT/'configs/direct_fiber_template_20261029.json';config=read(config_path);manifest_path=ROOT/config['manifest'];m=read(manifest_path)
    if len(m['train']['labeled'])!=521:raise ValueError('Unexpected TRAIN split')
    records=[];eligible=[];files={};fg=[];bg=[];bins=config['calibration']['histogram_bins']
    for row in m['train']['labeled']:
        if 'stone_new' in str(row).lower() or row.get('video_id'):raise ValueError('Excluded source in calibration')
        for key in ('image','mask'):
            path=Path(row[key]);files[str(path)]=sha256(path)
        rgb=np.asarray(Image.open(row['image']).convert('RGB'));frame=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR);mask=np.asarray(Image.open(row['mask']))
        fit=train_geometry(frame,mask,config['calibration']) if (mask==2).any() else None
        record={'sample_id':row['sample_id'],'group_id':row['group_id'],'source_dataset':row['source_dataset'],'eligible':fit is not None}
        if fit is not None:
            selected=fit.pop('selected');record.update(fit);eligible.append(record)
            kernel=max(3,round(min(mask.shape)*.025));kernel+=1-kernel%2
            outside=(cv2.dilate(selected.astype(np.uint8),np.ones((kernel,kernel),np.uint8))>0)&(mask!=2)&(mask!=255)
            if outside.any():fg.append(appearance_hist(frame,selected,bins));bg.append(appearance_hist(frame,outside,bins))
        records.append(record)
    if len(eligible)<config['calibration']['minimum_eligible_images']:raise ValueError(f'Only{len(eligible)}eligible training shapes; no validation execution')
    slopes=np.array([r['halfwidth_slope'] for r in eligible]);radii=np.array([r['cap_radius_fraction'] for r in eligible])
    smoothing=config['calibration']['histogram_smoothing'];a=np.mean(fg,0)+smoothing;b=np.mean(bg,0)+smoothing;a/=a.sum();b/=b.sum()
    prior={'halfwidth_slope':float(np.median(slopes)),'cap_radius_fraction':float(np.median(radii)),'cap_axis_ratio_fixed':1,
        'histogram_bins':bins,'appearance_probability':(a/(a+b)).tolist(),'eligible_train_images':len(eligible),
        'full_taper_angle_degrees':float(2*np.rad2deg(np.arctan(np.median(slopes)))),
        'training_geometry_spread':{'slope_q25_q50_q75':np.percentile(slopes,[25,50,75]).tolist(),'radius_fraction_q25_q50_q75':np.percentile(radii,[25,50,75]).tolist()},
        'scope':'Frozen pooled TRAIN-only provisional2D shape/appearance, not device/channel calibration; no validation labels or videos used'}
    atomic_json(OUT/'template.json',prior);atomic_json(OUT/'calibration.json',{'config_sha256':sha256(config_path),'manifest_sha256':sha256(manifest_path),'files':files,'records':records,'train_images_read':521,'eligible':len(eligible),'validation_images_read':0,'reserved_test_pixels_read':False,'gradient_updates':0})
    print(json.dumps({k:v for k,v in prior.items() if k!='appearance_probability'},indent=2))


if __name__=='__main__':calibrate()
