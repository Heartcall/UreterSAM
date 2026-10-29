"""Verify complete exports, frame-source hashes and frozen prior identities."""
import hashlib
import json
from pathlib import Path
import cv2

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/prior_method_comparison_20261029'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def paths(vid,i):
    f=f'{i:06d}.png'
    return {'classwise_raw':ROOT/f'runs/a1_classwise_replay_20261028/{vid}/classwise'/f,
        'old190_raw':ROOT/f'runs/a1_classwise_replay_20261028/{vid}/old190'/f,
        'edge_geometry':ROOT/f'runs/a1_fiber_geometry_20261028/{vid}/geometry'/f,
        'edge_geometry_causal':ROOT/f'runs/a1_fiber_geometry_20261028/{vid}/geometry_causal'/f,
        'component_gate_3net':ROOT/f'runs/a1_fiber_component_gate_20261029/full_replay/classwise_component_gate_3net/{vid}/gated'/f,
        'fixed_color':ROOT/f'runs/a1_direct_fiber_template_20261029/{vid}/image_template'/f,
        'fixed_color_soft':ROOT/f'runs/a1_direct_fiber_template_20261029/{vid}/soft_template'/f,
        'fixed_color_causal':ROOT/f'runs/a1_direct_fiber_template_20261029/{vid}/causal_template'/f,
        **{n:OUT/vid/n/f for n in ('component_gate_1net','fixed_neural')}}

def main():
    p=read(OUT/'preflight.json');s=read(OUT/'summary.json');job=read(OUT/'job.json');checks={};details={}
    checks['completed_all2694_frames']=job['phase']=='completed' and job['frames_completed']==2694 and set(s['exports'])=={'video_1','video_2'}
    checks['all_frozen_sources_metadata_videos_checkpoints_unchanged']=all(digest(ROOT/n)==h for n,h in p['sources'].items()) and all(digest(ROOT/n)==h for n,h in p['frozen_metadata'].items()) and all(digest(v['path'])==v['sha256'] for v in p['videos']+list(p['checkpoints'].values()))
    checks['no_training_default_acceptance_or_memory_change']=s['training_updates']==0 and all(s[n] is False for n in ('default_model_changed','acceptance_decision_changed','s2_accepted','stone_memory_started','reserved_test_pixels_read'))
    checks['all_eight_variants_and_two_references_labeled']=len(s['variants'])==len(s['quality_video29_only'])==10 and p['layout'].startswith('4 columns x3 rows')
    for vid,e in s['exports'].items():
        count=e['frames'];folder=OUT/vid;logs=[json.loads(line) for line in (folder/'supplemental_inference.jsonl').read_text(encoding='utf-8').splitlines()]
        checks[vid+'_supplemental_predictions_complete_consecutive']=len(logs)==count and [r['frame_index'] for r in logs]==list(range(count)) and all(len(list((folder/n).glob('*.png')))==count for n in ('component_gate_1net','fixed_neural'))
        checks[vid+'_static_predictions_and_raw_reference_reproduce_exact']=not any(s['supplemental_inference'][vid]['parity'].values()) and s['supplemental_inference'][vid]['counts']['reviewed_rgb_exact']==(20 if vid=='video_1' else 9)
        chain=hashlib.sha256();valid=True;lines=0
        for i,line in enumerate((folder/'render_sources.jsonl').read_text(encoding='utf-8').splitlines(keepends=True)):
            row=json.loads(line);source=paths(vid,i)
            valid &= row['frame_index']==i and set(row['source_sha256'])==set(source) and all(digest(path)==row['source_sha256'][n] for n,path in source.items())
            chain.update(line.encode('utf-8'));lines+=1
        checks[vid+'_every_frame_ten_mask_sources_and_hash_chain_exact']=bool(valid) and lines==count and chain.hexdigest()==e['source_mask_identity_chain']
        cap=cv2.VideoCapture(e['path']);codec=int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4,'little').decode('ascii');fps=float(cap.get(cv2.CAP_PROP_FPS))
        valid=cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==count and (int(cap.get(3)),int(cap.get(4)))==(1600,1436) and codec in ('h264','avc1') and abs(fps-e['source_fps'])<.005
        sample=[]
        for i in (0,count//2,count-1):
            cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,frame=cap.read();reference=cv2.imread(str(folder/f'comparison_frame_{i+1:06d}.jpg'))
            psnr=float(cv2.PSNR(frame,reference)) if ok else 0.;valid &= psnr>=25;sample.append({'one_based_frame':i+1,'psnr':psnr})
        cap.release()
        checks[vid+'_h264_native_panels_fps_full_duration_and_three_encoded_samples']=bool(valid) and e['decoded_frames']==count and digest(e['path'])==e['sha256']
        details[vid]={'frames':count,'duration_seconds':count/fps,'codec':codec,'bytes':Path(e['path']).stat().st_size,'samples':sample}
    result={'all_passed':bool(all(checks.values())),'checks':{n:bool(v) for n,v in checks.items()},'details':details,'scope':'Artifact/source/reproduction validation only; no dense-ground-truth quality or model acceptance claim'}
    (OUT/'export_checks.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    assert result['all_passed'],result
    print(json.dumps({'all_passed':True,'checks':len(checks),'details':details},indent=2))

if __name__=='__main__':main()
