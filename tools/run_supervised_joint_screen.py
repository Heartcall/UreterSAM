"""Fresh joint-data fixed-update baseline with separate image and video evaluation."""
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile

import torch

from ureter_semis.data import load_manifest
from ureter_semis.experiment import atomic_json,input_identity,require_training_access,sha256,source_identity
from ureter_semis.supervised_trainer import assess,train_supervised
from .evaluate_native_checkpoints import evaluate
from .run_development_training import validation_annotations

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    config_path=ROOT/'configs/train_a1_sup_kidney_joint_20261028.json';config=read(config_path);out=ROOT/config['output']
    if out.exists():raise FileExistsError(out)
    manifest_path=ROOT/config['manifest'];root,manifest=load_manifest(manifest_path)
    video_path=ROOT/config['video_manifest'];video_manifest=read(video_path)
    require_training_access(manifest,True)
    annotations=validation_annotations(ROOT/'runs/review_snapshots/video_keyframes_20261027_v2/pixel_reviews.json',video_manifest)
    identity=input_identity(manifest_path,root,manifest)
    video_identity=input_identity(video_path,root,video_manifest)
    encoder=Path(torch.hub.get_dir())/'checkpoints/resnet34-333f7ec4.pth'
    related=[config_path,ROOT/'configs/prepare_kidney_joint_20261028.json',root/'groups.json',root/'raw_inventory.json',root/'video_overlap.json',root/'video_validation_manifest.json']
    preflight={'config':config,'config_sha256':sha256(config_path),'inputs':identity,'video_inputs':video_identity,
        'source_hashes':source_identity(),'related_inputs':{str(p):sha256(p) for p in related},'encoder_sha256':sha256(encoder),
        'planned_effective_updates':1425,'planned_image_exposures':2850,'new_train_images':len(manifest['train']['labeled']),
        'image_validation':len(manifest['val']),'video_validation':29,'training_video_frames':0,'unlabeled_images_used':0,
        'protocol_frozen_before_training':True}
    out.mkdir(parents=True)
    # Legacy ImageNet tar checkpoints extract via tempfile; sandbox system TMP may be read-only.
    runtime_tmp=out/'runtime_tmp';runtime_tmp.mkdir()
    tempfile.tempdir=str(runtime_tmp)
    preflight['runtime_temporary_directory']=str(runtime_tmp)
    atomic_json(out/'preflight.json',preflight);atomic_json(out/'video_annotations.json',annotations)
    with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for folder in ('ureter_semis','tools','third_party/Endo-SemiS/src/models'):
            for path in sorted((ROOT/folder).rglob('*.py')):archive.write(path,str(path.relative_to(ROOT)))
        for path in [config_path,ROOT/'configs/prepare_kidney_joint_20261028.json']:archive.write(path,str(path.relative_to(ROOT)))
    job={'pid':os.getpid(),'experiment':config['experiment'],'training_video_frames':0,'unlabeled_images_used':0}
    def phase(value,**more):job.update(phase=value,**more);atomic_json(out/'job.json',job)
    try:
        phase('freeze_video_reference')
        baseline=evaluate(video_path,{'A0':ROOT/'runs/endosemis/best.pt',
            'kidney190':ROOT/'runs/a1_sup_kidney_resolution_20261028/unet256/last.pt'},out/'baseline_video_native')
        frozen=read(ROOT/'runs/a1_sup_kidney_resolution_20261028/native_comparison/comparison.json')['models']
        assert baseline['models']['A0']['metrics']==frozen['A0']['metrics']
        assert baseline['models']['kidney190']['metrics']==frozen['unet256']['metrics']
        m=baseline['models']['A0']['metrics']
        gates={'stone_dice_min':m['stone_dice'],'fiber_dice_min':m['fiber_dice']-.01,'negative_fp_fraction_max':m['stone_negative_fp_pixel_fraction']}
        preflight.update(native_video_gates=gates,gates_frozen_before_training_model_construction=True)
        atomic_json(out/'preflight.json',preflight)
        phase('training_fixed1425_updates',planned_epochs=15)
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        train_supervised(manifest_path=str(manifest_path),output_dir=str(out/'training'),gates=gates,**config['training'])
        elapsed=time.perf_counter()-start;memory=torch.cuda.max_memory_allocated()/2**20
        history=read(out/'training/history.json');status=read(out/'training/status.json')
        if len(history)!=15 or any(r['optimizer_update_steps']!=95 or r['sampled_training_images']!=190 for r in history):
            raise ValueError('Bounded run did not complete matched1425 effective updates; keep diagnostic, do not extend')
        sampled=[sid for r in history for sid in r['sampled_training_sample_ids']]
        cost={'training_validation_io_wall_s':elapsed,'peak_cuda_allocated_mib':memory,'effective_updates':sum(r['optimizer_update_steps'] for r in history),
            'image_exposures':len(sampled),'unique_training_images_seen':len(set(sampled)),'training_pool_images':len(manifest['train']['labeled'])}
        atomic_json(out/'training_cost.json',cost)
        phase('separate_final_native_evaluation')
        checkpoint=out/'training/last.pt'
        image_result=evaluate(manifest_path,{'joint_final':checkpoint},out/'image_native')
        video_result=evaluate(video_path,{'joint_final':checkpoint},out/'video_native')
        video_metrics=video_result['models']['joint_final']['metrics']
        screening=assess(video_metrics,gates)
        phase('fresh_process_cost')
        with (out/'benchmark.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-m','tools.benchmark_native','--manifest',str(video_path),'--checkpoint',str(checkpoint),
                '--output',str(out/'performance.json')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        assert input_identity(manifest_path,root,manifest)==identity and input_identity(video_path,root,video_manifest)==video_identity
        assert all(sha256(Path(p))==digest for p,digest in preflight['related_inputs'].items())
        assert source_identity()==preflight['source_hashes'] and sha256(encoder)==preflight['encoder_sha256']
        summary={'config':config,'fixed_final_epoch':15,'training_cost':cost,'image_validation':image_result['models']['joint_final'],
            'video_validation':video_result['models']['joint_final'],'video_baselines':baseline['models'],
            'video_screening':screening,'image_best_epoch_diagnostic_only':status['best_epoch'],
            'performance':read(out/'performance.json'),'adopted':False,'s2_accepted':False,
            'scope':'Single-seed new-data matched-update development baseline; no independent case or temporal/geometry acceptance'}
        atomic_json(out/'summary.json',summary)
        phase('completed',completed_epochs=15,effective_updates=1425,screening_passed=screening['passed'],adopted=False,s2_accepted=False)
        print(json.dumps({'phase':'completed','video_metrics':video_metrics,'video_screening':screening,'training_cost':cost},indent=2))
    except Exception as error:
        phase('failed',error=str(error));(out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise


if __name__=='__main__':main()
