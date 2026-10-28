"""New joint source version, conservative dedup/group split and frozen video29."""
from __future__ import annotations

from collections import Counter,defaultdict
import copy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import yaml

from ureter_semis.experiment import atomic_json,sha256
from ureter_semis.source_policy import excluded_training_source,require_training_sources
from ureter_semis.supervision import audit_supervision
from .audit_kidney_v2 import feature,variants,verified_pairs,read_image
from .convert_dataset import read_yolo_sample

ROOT=Path(__file__).resolve().parents[1]


def grouped_components(records,pairs):
    parent=list(range(len(records)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(a,b):parent[find(b)]=find(a)
    families={};hashes={}
    for i,row in enumerate(records):
        for key,seen in (((row['source_dataset'],row['family']),families),(row['image_sha256'],hashes)):
            if key in seen:union(i,seen[key])
            else:seen[key]=i
    for pair in pairs:union(pair['query_index'],pair['reference_index'])
    groups=defaultdict(list)
    for i in range(len(records)):groups[find(i)].append(i)
    return list(groups.values())


def assign_component(members,records,seed,train_fraction):
    identities=sorted({r['source_dataset']+':'+r['family'] for r in (records[i] for i in members)})
    digest=hashlib.sha256((str(seed)+':'+','.join(identities)).encode()).hexdigest()
    split='train' if int(digest[:16],16)/2**64<train_fraction else 'image_val'
    if any(records[i]['source_split']=='test' for i in members):split='reserved_test'
    if any(records[i].get('excluded') for i in members):split='excluded'
    canonical=min(members,key=lambda i:(records[i]['source_dataset']!='kidney_stones_fiber_v2',records[i]['source_split']=='train',records[i]['image_sha256']))
    return digest[:24],split,canonical


def main():
    config_path=ROOT/'configs/prepare_kidney_joint_20261028.json';config=json.loads(config_path.read_text(encoding='utf-8'))
    output=ROOT/config['output']
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);(output/'masks').mkdir();(output/'auxiliary').mkdir()
    atomic_json(output/'job.json',{'phase':'indexing_sources','training_updates':0})
    oldpath=ROOT/config['previous_manifest'];previous=json.loads(oldpath.read_text(encoding='utf-8'))
    session=json.loads((ROOT/config['review_session']).read_text(encoding='utf-8'))
    rejected={key.removeprefix('dataset:') for key,value in session['decisions'].items() if key.startswith('dataset:') and value.get('review_status')=='rejected'}
    draft=json.loads((ROOT/config['draft_manifest']).read_text(encoding='utf-8'))
    rejected_paths={str(Path(r['source_image']).resolve()) for r in draft['draft_samples'] if r['sample_id'] in rejected}
    registry=json.loads((ROOT/'annotations/excluded_training_sources_20261027.json').read_text(encoding='utf-8'))
    blocked=set(registry['excluded_image_sha256'])
    records=[];features=[];source_names={};inputs={}
    for source in config['sources']:
        directory=ROOT/source['path'];data=yaml.safe_load((directory/'data.yaml').read_text(encoding='utf-8'))
        assert data['names']==source['expected_names'];names=dict(enumerate(data['names']));source_names[source['id']]=names
        inputs[str(directory/'data.yaml')]=sha256(directory/'data.yaml')
        for split in ('train','valid','test'):
            for path in sorted((directory/split/'images').glob('*.jpg')):
                label=directory/split/'labels'/(path.stem+'.txt')
                mask,meta,_=read_yolo_sample(path,label,names)
                if not meta['label_file_present']:raise ValueError('Missing source label')
                digest=sha256(path);inputs[str(path)]=digest;inputs[str(label)]=sha256(label)
                row={'index':len(records),'source_dataset':source['id'],'source_split':split,'image':str(path),
                    'source_label':str(label),'source_label_sha256':inputs[str(label)],'image_sha256':digest,'family':path.stem.split('.rf.')[0],
                    'width':meta['width'],'height':meta['height'],'annotation_counts':meta['annotation_counts'],'review_flags':meta['review_flags']}
                row['excluded']=str(path.resolve()) in rejected_paths or excluded_training_source(row,blocked)
                records.append(row);features.append([feature(v) for v in variants(read_image(path))])
    atomic_json(output/'raw_inventory.json',{'config':config,'config_sha256':sha256(config_path),'records':records,'input_hashes':inputs})
    atomic_json(output/'job.json',{'phase':'joint_near_copy_screen','raw_images':len(records),'training_updates':0})
    pairs=verified_pairs(features,[f[0] for f in features],exclude_self=True)
    atomic_json(output/'near_copy_pairs.json',pairs)
    atomic_json(output/'job.json',{'phase':'video_exclusion_screen','training_updates':0})
    video_features=[];video_frames=[];video_hashes={}
    for name in ('video_1','video_2'):
        path=ROOT/'data/videos'/(name+'.mp4');video_hashes[str(path)]=sha256(path)
        cap=cv2.VideoCapture(str(path));index=0
        if not cap.isOpened():raise ValueError(path)
        while True:
            ok,bgr=cap.read()
            if not ok:break
            video_features.append(feature(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)));video_frames.append({'video_id':name,'frame_index':index});index+=1
        cap.release()
    video_pairs=verified_pairs(features,video_features)
    for pair in video_pairs:
        records[pair['query_index']]['excluded']=True
        records[pair['query_index']]['video_duplicate_candidate']=video_frames[pair['reference_index']]
    atomic_json(output/'video_overlap.json',{'video_sha256':video_hashes,'frames_compared':len(video_frames),'pairs':video_pairs,'case_independence_verified':False})
    components=grouped_components(records,pairs)
    groups=[];rows_by_split=defaultdict(list)
    # Label conflicts are recorded, never resolved by merging masks across geometries.
    for members in components:
        group_id,split,canonical=assign_component(members,records,config['split']['seed'],config['split']['train_fraction'])
        r=records[canonical];mask,metadata,aux=read_yolo_sample(Path(r['image']),Path(r['source_label']),source_names[r['source_dataset']])
        sid=r['source_dataset']+'__joint__'+group_id
        maskpath=output/'masks'/(sid+'.png');Image.fromarray(mask).save(maskpath)
        auxiliary={}
        for name,array in aux.items():
            path=output/'auxiliary'/(sid+'__'+name+'.png');Image.fromarray(array).save(path);auxiliary[name]=str(path)
        signatures={tuple(sorted(records[i]['annotation_counts'].items())) for i in members}
        group={'group_id':group_id,'split':split,'canonical_index':canonical,'member_indices':members,
            'source_families':sorted({records[i]['source_dataset']+':'+records[i]['family'] for i in members}),
            'annotation_counts_disagree':len(signatures)>1,'contains_original_test':any(records[i]['source_split']=='test' for i in members)}
        groups.append(group)
        rows_by_split[split].append({'sample_id':sid,'image':r['image'],'mask':str(maskpath),'source_image':r['image'],
            'source_label':r['source_label'],'source_label_sha256':r['source_label_sha256'],'image_sha256':r['image_sha256'],
            'mask_sha256':sha256(maskpath),'source_dataset':r['source_dataset'],'source_split':r['source_split'],'group_id':group_id,
            'source_group_verified':False,'patient_id':None,'video_id':None,'review_status':'pending','stone_presence_review':'unknown',
            'annotation_basis':'provided_polygons_and_provisional_background_unreviewed','background_basis':'source_label_background_provisional_not_reviewed',
            'fiber_annotation_completeness':'unknown','review_flags':metadata['review_flags']+(['duplicate_component_annotation_counts_disagree'] if len(signatures)>1 else []),
            'auxiliary_masks':auxiliary,'dedup_members':len(members),'width':metadata['width'],'height':metadata['height']})
    assert len(rows_by_split['train'])>=190 and rows_by_split['image_val']
    video_rows=copy.deepcopy(previous['val'])
    for row in video_rows:
        assert sha256(Path(row['image']))==row['image_sha256'] and sha256(Path(row['mask']))==row['mask_sha256']
    manifest={'schema_version':3,'root':str(output),'classes':['background','stone','fiber'],'ignore_index':255,
        'role':'unreviewed_development_experiment','training_ready':False,'experiment_ready':True,
        'review_waiver':{'user_authorized':True,'instruction':config['user_authorization'],'review_status_not_promoted':True},
        'train':{'labeled':sorted(rows_by_split['train'],key=lambda r:r['sample_id']),'unlabeled':[]},
        'val':sorted(rows_by_split['image_val'],key=lambda r:r['sample_id']),'test':[],
        'reserved_test':sorted(rows_by_split['reserved_test'],key=lambda r:r['sample_id']),'excluded':rows_by_split['excluded'],
        'reserved_unlabeled':[],'video_validation':video_rows,'independent_test_available':False,
        'split_policy':config['split'],'provenance':{'previous_manifest_sha256':sha256(oldpath),'preparation_config_sha256':sha256(config_path)},
        'evaluation_limitations':['Source labels and background provisional; no new manual review','Groups are near-copy/augmentation groups, not verified cases',
            'Video29 is separate regression only; no training/selection on video frames','Original test-associated groups retained; no test-mask evaluation',
            'Changed data protocol; old190 comparisons on video29 only, new image validation may overlap historical training'],
        'counts':{'raw_images':len(records),'canonical_components':len(groups),'train_labeled':len(rows_by_split['train']),
            'image_validation':len(rows_by_split['image_val']),'video_validation':29,'reserved_test':len(rows_by_split['reserved_test']),
            'excluded':len(rows_by_split['excluded']),'reviewed_train':0,'unlabeled_used':0}}
    require_training_sources(manifest);audit=audit_supervision(output,manifest)
    video_manifest=copy.deepcopy(manifest);video_manifest.update(train={'labeled':[],'unlabeled':[]},val=video_rows,role='video_validation_only',experiment_ready=False)
    video_manifest.pop('reserved_test',None);video_manifest.pop('video_validation',None)
    assert len(video_rows)==29 and {r['video_id'] for r in video_rows}=={'video_1','video_2'}
    assert not {r['image_sha256'] for r in manifest['train']['labeled']} & {r['image_sha256'] for r in manifest['val']+video_rows}
    atomic_json(output/'manifest.json',manifest);atomic_json(output/'video_validation_manifest.json',video_manifest)
    atomic_json(output/'groups.json',groups);atomic_json(output/'supervision_audit.json',audit)
    atomic_json(output/'summary.json',{'counts':manifest['counts'],'source_by_split':{s:dict(Counter(r['source_dataset'] for r in rows)) for s,rows in rows_by_split.items()},
        'annotation_count_conflict_components':sum(g['annotation_counts_disagree'] for g in groups),'near_copy_pairs':len(pairs),
        'video_duplicate_candidates':len(video_pairs),'case_independence_verified':False,'preparation_config':config})
    assert all(sha256(Path(path))==digest for path,digest in inputs.items())
    assert sha256(oldpath)==manifest['provenance']['previous_manifest_sha256']
    atomic_json(output/'job.json',{'phase':'completed','training_updates':0,**manifest['counts']})
    print(json.dumps(manifest['counts'],indent=2))


if __name__=='__main__':main()
