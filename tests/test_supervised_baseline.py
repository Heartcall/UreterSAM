from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from tools.prepare_supervised_experiment import build
from ureter_semis.experiment import sha256
from ureter_semis.inference import SegmentationPredictor
from ureter_semis.supervised_trainer import assess, collapsed, labeled_sampler, train_supervised
from ureter_semis.supervision import audit_supervision
from ureter_semis.losses import supervised_multiclass_loss
from test_training_reproducibility import TinyModel, make_fixture


class SupervisedBaselineTests(unittest.TestCase):
    def setUp(self):
        self.old_threads = torch.get_num_threads(); torch.set_num_threads(1)
        root = Path("runs/test_tmp").resolve(); root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=root); self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup(); torch.set_num_threads(self.old_threads)

    def assert_nested_equal(self, a, b):
        if isinstance(a, torch.Tensor): self.assertTrue(torch.equal(a, b))
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a: self.assert_nested_equal(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b): self.assert_nested_equal(x, y)
        else: self.assertEqual(a, b)

    def test_foreground_only_and_validation_leakage_fail_before_training(self):
        path = make_fixture(self.root); m = json.loads(path.read_text())
        for row in m['train']['labeled']:
            p = self.root / row['mask']; mask = np.asarray(Image.open(p)).copy()
            mask[mask == 0] = 255; Image.fromarray(mask).save(p)
        with self.assertRaisesRegex(ValueError, 'background'):
            audit_supervision(self.root, m)
        m['train']['labeled'][0]['video_id'] = 'video_1'
        with self.assertRaisesRegex(ValueError, 'validation only'):
            audit_supervision(self.root, m)
        m['train']['labeled'][0].pop('video_id')
        m['train']['labeled'][0]['image_sha256'] = 'same'
        m['val'][1]['image_sha256'] = 'same'
        with self.assertRaisesRegex(ValueError, 'appears in training'):
            audit_supervision(self.root, m)

    def test_builder_restores_only_source_background_and_preserves_split_and_unknown(self):
        path = make_fixture(self.root); m = json.loads(path.read_text())
        draft_rows=[]
        for i,row in enumerate(m['train']['labeled']+m['train']['unlabeled']):
            row.update(sample_id=f's{i}', image_sha256=sha256(self.root/row['image']),
                       source_dataset='fixture', video_id=None, review_status='pending', source_group_verified=False)
            draft_rows.append({**row, 'mask_sha256':sha256(self.root/row['mask']), 'proposed_split':'train',
                               'review_flags':['fiber_completeness_unknown']})
        original_masks=[np.asarray(Image.open(self.root/r['mask'])).copy() for r in m['train']['labeled']]
        draft=self.root/'draft.json'; draft.write_text(json.dumps({'root':str(self.root),'draft_samples':draft_rows}))
        previous=self.root/'previous.json'
        m.update(review_waiver={'user_authorized':True},provenance={},counts={})
        previous.write_text(json.dumps(m))
        session=self.root/'session.json';session.write_text('{"decisions":{}}')
        # Avoid synthetic overlap: use distinct content identifiers for validation.
        for i,row in enumerate(m['val']): row['image_sha256']=f'validation_{i}'
        previous.write_text(json.dumps(m))
        result=build(previous,draft,session,self.root/'new')
        new=json.loads(result.read_text(encoding='utf-8'))
        self.assertEqual(new['train']['unlabeled'],[])
        self.assertEqual(len(new['reserved_unlabeled']),6)
        self.assertEqual(new['val'],m['val'])
        for expected,row in zip(original_masks,new['train']['labeled']):
            self.assertTrue(np.array_equal(expected,np.asarray(Image.open(row['mask']))))
            self.assertEqual(row['review_status'],'pending');self.assertFalse(row['source_group_verified'])
            self.assertIn(255,np.unique(expected));self.assertIn(0,np.unique(expected))
        with self.assertRaises(FileExistsError):build(previous,draft,session,self.root/'new')

    def test_single_supervised_resume_is_exact_and_reserved_images_are_unused(self):
        manifest=make_fixture(self.root); m=json.loads(manifest.read_text())
        m['reserved_unlabeled']=m['train']['unlabeled']+[{'image':'must_not_be_read.png'}]
        m['train']['unlabeled']=[];manifest.write_text(json.dumps(m))
        args=dict(manifest_path=str(manifest),epochs=2,size=32,batch_size=2,
                  pretrained=False,device_name='cpu',seed=17)
        with patch('ureter_semis.supervised_trainer.build_model',TinyModel):
            train_supervised(**args,output_dir=str(self.root/'full'))
            train_supervised(**args,output_dir=str(self.root/'staged'),stop_after_epoch=1)
            train_supervised(**args,output_dir=str(self.root/'staged'),resume=str(self.root/'staged/last.pt'))
        a=torch.load(self.root/'full/last.pt',weights_only=True)
        b=torch.load(self.root/'staged/last.pt',weights_only=True)
        self.assert_nested_equal(a,b)
        self.assertNotIn('model2',a);self.assertEqual(a['model_count'],1)
        self.assertEqual(a['history'][0]['optimizer_update_steps'],2)
        self.assertFalse(any('must_not_be_read' in p for p in a['run_identity']['inputs']['files']))
        with patch('ureter_semis.inference.build_model',TinyModel):
            predictor=SegmentationPredictor(self.root/'full/best.pt',device='cpu')
            prediction=predictor.predict(np.zeros((32,32,3),np.uint8))
            self.assertEqual(len(predictor.models),1);self.assertEqual(prediction.mask.shape,(32,32))

    def test_fiber_score_cannot_hide_stone_failure_and_collapse_stops(self):
        gates={'stone_dice_min':.48,'fiber_dice_min':.96,'negative_fp_fraction_max':.08}
        metrics={'stone_dice':.04,'fiber_dice':.999,'stone_negative_fp_pixel_fraction':.92,'stone_precision':.02}
        self.assertFalse(assess(metrics,gates)['passed']);self.assertTrue(collapsed(metrics))
        self.assertTrue(collapsed({'confusion_matrix':[[0,0,900],[0,100,0],[0,0,10]],
                                   'stone_precision':1.,'fiber_precision':.01}))
        manifest=make_fixture(self.root);m=json.loads(manifest.read_text());m['train']['unlabeled']=[]
        manifest.write_text(json.dumps(m))
        with patch('ureter_semis.supervised_trainer.build_model',TinyModel), patch(
                'ureter_semis.supervised_trainer.validate',return_value=metrics):
            train_supervised(str(manifest),str(self.root/'collapse'),epochs=5,size=32,pretrained=False,device_name='cpu',gates=gates)
        status=json.loads((self.root/'collapse/status.json').read_text())
        self.assertEqual(status['state'],'stopped_early_collapse');self.assertEqual(status['completed_epoch'],3)
        self.assertFalse(status['best_passed_screening'])

    def test_fiber_sampler_has_declared_mass_and_reproducible_state(self):
        audit={'samples':[{'pixels':{0:100,2:5}} for _ in range(2)]+[{'pixels':{0:100}} for _ in range(3)]}
        generator=torch.Generator().manual_seed(7)
        sampler=labeled_sampler(audit,'fiber_balanced',.7,generator)
        self.assertAlmostEqual(float(sampler.weights[:2].sum()),.7)
        self.assertAlmostEqual(float(sampler.weights[2:].sum()),.3)
        state=generator.get_state();first=list(sampler);generator.set_state(state)
        self.assertEqual(first,list(sampler))
        with self.assertRaises(ValueError):labeled_sampler(audit,'fiber_balanced',1.,generator)

    def test_fiber_balanced_full_training_matches_epoch_resume(self):
        manifest=make_fixture(self.root);m=json.loads(manifest.read_text());m['train']['unlabeled']=[]
        for row in m['train']['labeled'][:2]:
            p=self.root/row['mask'];a=np.asarray(Image.open(p)).copy();a[a==2]=0;Image.fromarray(a).save(p)
        manifest.write_text(json.dumps(m))
        args=dict(manifest_path=str(manifest),epochs=2,size=32,batch_size=2,pretrained=False,
                  device_name='cpu',seed=17,sampling_strategy='fiber_balanced',fiber_image_fraction=.7)
        with patch('ureter_semis.supervised_trainer.build_model',TinyModel):
            train_supervised(**args,output_dir=str(self.root/'balanced_full'))
            train_supervised(**args,output_dir=str(self.root/'balanced_staged'),stop_after_epoch=1)
            train_supervised(**args,output_dir=str(self.root/'balanced_staged'),resume=str(self.root/'balanced_staged/last.pt'))
        a=torch.load(self.root/'balanced_full/last.pt',weights_only=True)
        b=torch.load(self.root/'balanced_staged/last.pt',weights_only=True)
        self.assert_nested_equal(a,b)
        self.assertEqual(a['config']['sampling_strategy'],'fiber_balanced')

    def test_classwise_partial_background_keeps_stone_negative_without_penalizing_fiber(self):
        target=torch.zeros((1,1,1),dtype=torch.long);unknown=torch.ones_like(target,dtype=torch.bool)
        weights=torch.tensor([1.,2.,4.])
        a=torch.tensor([.5,.1,.4]).log().view(1,3,1,1).requires_grad_()
        b=torch.tensor([.4,.1,.5]).log().view(1,3,1,1)
        loss=supervised_multiclass_loss(a,target,weights,dice_weight=0,fiber_unknown_background=unknown)
        same=supervised_multiclass_loss(b,target,weights,dice_weight=0,fiber_unknown_background=unknown)
        self.assertTrue(torch.allclose(loss,same))
        grad=torch.autograd.grad(loss,a)[0]
        self.assertGreater(float(grad[0,1,0,0]),0)
        self.assertLess(float(grad[0,2,0,0]),0)
        with self.assertRaisesRegex(ValueError,'Only provisional background'):
            supervised_multiclass_loss(a,torch.ones_like(target),weights,fiber_unknown_background=unknown)
        full=supervised_multiclass_loss(a,target,weights,fiber_unknown_background=torch.zeros_like(unknown))
        self.assertTrue(torch.equal(full,supervised_multiclass_loss(a,target,weights)))


if __name__=='__main__':unittest.main()
