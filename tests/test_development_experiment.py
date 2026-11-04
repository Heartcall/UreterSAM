from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from tools.prepare_development_experiment import build
from tools.run_development_training import run as run_pipeline, validation_annotations
from ureter_semis.experiment import require_training_access, sha256
from ureter_semis.labels import read_class_mask


class DevelopmentExperimentTests(unittest.TestCase):
    def test_retired_video_training_experiment_cannot_be_resumed(self):
        parent=Path("runs/test_tmp").resolve();parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as name:
            root=Path(name);output=root/"retired";output.mkdir()
            (output/"retired_run.json").write_text('{"state":"retired"}')
            config=root/"config.json";config.write_text('{}')
            with self.assertRaisesRegex(ValueError,'retired'):
                run_pipeline(config,output,resume=True)
            self.assertEqual({p.name for p in output.iterdir()},{'retired_run.json'})

    def test_review_exception_requires_explicit_flag_and_separate_authorized_role(self):
        draft={"training_ready":False}
        for flag in (False,True):
            with self.assertRaisesRegex(ValueError,"unreviewed draft"):require_training_access(draft,flag)
        development={**draft,"role":"unreviewed_development_experiment","experiment_ready":True,
                     "review_waiver":{"user_authorized":True}}
        with self.assertRaises(ValueError):require_training_access(development)
        require_training_access(development,True)
        development["review_waiver"]["user_authorized"]=False
        with self.assertRaises(ValueError):require_training_access(development,True)

    def test_partial_labels_and_both_validation_only_videos_are_preserved(self):
        parent=Path("runs/test_tmp").resolve();parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as name:
            root=Path(name);rows=[]
            for i in range(12):
                image=root/f"raw_{i}.png";mask=root/f"mask_{i}.png"
                Image.new("RGB",(32,32),(i,30,70)).save(image)
                labels=np.zeros((32,32),np.uint8);labels[2:8,2:8]=1;labels[14:16]=2
                Image.fromarray(labels).save(mask)
                rows.append({"sample_id":f"raw{i}","image":image.name,"mask":mask.name,"image_sha256":sha256(image),
                    "mask_sha256":sha256(mask),"proposed_split":"train" if i<10 else "val",
                    "source_dataset":"synthetic","group_id":f"group{i}"})
            draft=root/"draft.json";draft.write_text(json.dumps({"schema_version":2,"root":str(root),"draft_samples":rows}))
            session=root/"session.json";session.write_text(json.dumps({"decisions":{"dataset:raw0":{"review_status":"rejected"}}}))
            reviewed=[]
            for vid in ("video_1","video_2"):
                for index,presence in enumerate(("present","absent")):
                    image=root/f"{vid}_{index}.png";mask=root/f"{vid}_{index}_mask.png"
                    Image.new("RGB",(32,32),(100+index,30,120 if vid=="video_1" else 150)).save(image)
                    labels=np.zeros((32,32),np.uint8)
                    if presence=="present":labels[3:9,3:9]=1
                    labels[-2:]=2;Image.fromarray(labels).save(mask)
                    reviewed.append({"video_id":vid,"patient_id":f"case_{vid}","frame_index":index,
                        "image":image.name,"mask":mask.name,"image_sha256":sha256(image),"mask_sha256":sha256(mask),
                        "review_status":"approved","reviewer":"synthetic_fixture","stone_presence":presence})
            snapshot=root/"reviewed.json";snapshot.write_text(json.dumps({"training_ready":False,"frames":reviewed}))
            output=root/"experiment"
            manifest_path=build(draft,snapshot,session,output,labeled_fraction=.4)
            manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
            self.assertFalse(manifest["training_ready"]);self.assertTrue(manifest["experiment_ready"])
            self.assertEqual(manifest["counts"]["excluded"]["explicitly_rejected"],1)
            self.assertEqual({r['video_id'] for r in manifest['val']},{'video_1','video_2'})
            self.assertEqual(len(manifest['val']),4)
            self.assertEqual(manifest['counts']['reviewed_train'],0)
            training=manifest['train']['labeled']+manifest['train']['unlabeled']
            self.assertNotIn('raw0',{r['sample_id'] for r in training})
            self.assertTrue(all(r.get('video_id') is None for r in training))
            self.assertTrue(all(r['review_status']=='pending' for r in training))
            for row in manifest['train']['labeled']:
                mask=read_class_mask(Path(row['mask']))
                if row['review_status']=='pending':
                    self.assertNotIn(0,np.unique(mask));self.assertIn(255,np.unique(mask))
                    self.assertFalse(row['source_group_verified'])
            for row in manifest['val']:
                if row['stone_presence_review']=='absent':
                    mask=read_class_mask(Path(row['mask']))
                    self.assertIn(0,np.unique(mask));self.assertNotIn(1,np.unique(mask))
            self.assertAlmostEqual(sum(r['sampling_weight'] for r in manifest['train']['labeled']),1)
            self.assertTrue(all(abs(r['sampling_weight']-1/len(manifest['train']['labeled']))<1e-12
                                for r in manifest['train']['labeled']))
            annotations=validation_annotations(snapshot,manifest)
            self.assertEqual(len(annotations['frames']),4)
            self.assertEqual({r['video_id'] for r in annotations['frames']},{'video_1','video_2'})
            bad=json.loads(json.dumps(manifest))
            bad['train']['unlabeled'].append(bad['val'][0])
            with self.assertRaisesRegex(ValueError,'validation only'):
                validation_annotations(snapshot,bad)
            bad=json.loads(json.dumps(manifest))
            bad['train']['unlabeled'][0]['image_sha256']=bad['val'][0]['image_sha256']
            with self.assertRaisesRegex(ValueError,'appears in training'):
                validation_annotations(snapshot,bad)
            bad=json.loads(json.dumps(manifest))
            bad['val'][0]['mask_sha256']='changed'
            with self.assertRaisesRegex(ValueError,'differs from frozen'):
                validation_annotations(snapshot,bad)
            self.assertEqual(manifest['test'],[])
            with self.assertRaises(FileExistsError):build(draft,snapshot,session,output)


if __name__=='__main__':unittest.main()
