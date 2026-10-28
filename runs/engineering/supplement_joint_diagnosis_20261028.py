"""Record source augmentation evidence and qualified active-label comparisons."""
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from ureter_semis.experiment import atomic_json,sha256


if __name__=='__main__':
    out=ROOT/'runs/a1_sup_kidney_joint_diagnosis_20261028'
    conflicts=json.loads((out/'conflicts.json').read_text(encoding='utf-8'))['groups']
    source=json.loads((out/'source_exposure.json').read_text(encoding='utf-8'))
    active=[r for r in source['source_records'] if r['active_presence_conflict']]
    aligned=[]
    for group in conflicts:
        if group['pixel_analysis']!='verified_D4_source_disagreement_only':continue
        folder=out/'conflicts'/group['group_id']
        a=np.asarray(Image.open(folder/'canonical_mask.png'));b=np.asarray(Image.open(folder/'member_aligned_mask.png'))
        union=int(((a==1)|(b==1)).sum());intersection=int(((a==1)&(b==1)).sum())
        aligned.append({'group_id':group['group_id'],'canonical_family':group['canonical_family'],
            'stone_label_iou_on256_alignment':intersection/union if union else None,
            'fiber_disagreement_pixels_256':group['fiber_disagreement_pixels_256'],
            'fiber_vs_other_source_laser_overlap_pixels_256':group['fiber_union_overlap_with_other_source_laser_pixels_256']})
    readmes={version:ROOT/f'data/Kidney stones-fiber-.{version}i.yolov8/README.roboflow.txt' for version in ('v1','v2')}
    result={'readme_sha256':{version:sha256(path) for version,path in readmes.items()},
        'documented_v1_augmentation':['horizontal/vertical flips','90degree rotations','random crop0-20percent','rotation-15to15degrees',
            'horizontal/vertical shear-13to13degrees','brightness/exposure','Gaussian blur0-3.6pixels','salt-and-pepper noise1.45percent pixels'],
        'documented_v2_augmentation':['horizontal/vertical flips','90degree rotations','brightness/exposure','Gaussian blur0-3pixels'],
        'alignment_limit':'D4-only verifier deliberately does not fit crop/shear/arbitrary rotation; do not equate failed direct alignment or object-count disagreement with annotation error',
        'active_conflict_training_groups':sum(r['split']=='train' for r in active),
        'active_conflict_imageval_groups':sum(r['split']=='image_val' for r in active),
        'all_active_canonicals_have_fiber_annotation':all(r['fiber']['target_pixels']>0 for r in active),
        'active_conflict_train_exposures':sum(r['exposures'] or 0 for r in active),
        'active_conflict_train_exposure_fraction':sum(r['exposures'] or 0 for r in active)/2850,
        'strict_aligned_source_label_comparisons':aligned,
        'interpretation':'Source-label differences only, not accuracy against verified truth; metadata/geometry/canonicalization/exposure factors are not isolated causal experiments'}
    atomic_json(out/'supplemental_source_evidence.json',result)
    with zipfile.ZipFile(out/'source_snapshot.zip','a',zipfile.ZIP_DEFLATED) as z:z.write(Path(__file__).resolve(),str(Path(__file__).resolve().relative_to(ROOT)))
    print(json.dumps(result,indent=2))
