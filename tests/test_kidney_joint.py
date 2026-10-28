import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from tools.prepare_kidney_joint import grouped_components,assign_component
from ureter_semis.supervised_trainer import train_supervised
from test_training_reproducibility import TinyModel,make_fixture


class KidneyJointTests(unittest.TestCase):
    def test_transitive_family_duplicates_test_and_rejection_propagate(self):
        rows=[{'source_dataset':'v1','family':'a','image_sha256':'a','source_split':'train'},
              {'source_dataset':'v1','family':'a','image_sha256':'b','source_split':'train'},
              {'source_dataset':'kidney_stones_fiber_v2','family':'c','image_sha256':'c','source_split':'valid'},
              {'source_dataset':'kidney_stones_fiber_v2','family':'c','image_sha256':'d','source_split':'test'},
              {'source_dataset':'v1','family':'other','image_sha256':'e','source_split':'train'}]
        groups=grouped_components(rows,[{'query_index':1,'reference_index':2}])
        self.assertEqual(sorted(map(sorted,groups)),[[0,1,2,3],[4]])
        group_id,split,index=assign_component(groups[0],rows,42,.8)
        self.assertEqual(split,'reserved_test');self.assertEqual(index,2)
        self.assertEqual(assign_component(list(reversed(groups[0])),rows,42,.8),(group_id,split,index))
        rows[0]['excluded']=True
        self.assertEqual(assign_component(groups[0],rows,42,.8)[1],'excluded')

    def test_fixed_subset_update_budget_and_resume_rng_are_exact(self):
        threads=torch.get_num_threads();torch.set_num_threads(1)
        parent=Path('runs/test_tmp').resolve();parent.mkdir(parents=True,exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(dir=parent) as directory:
                root=Path(directory);manifest=make_fixture(root,size=40)
                data=json.loads(manifest.read_text());data['train']['unlabeled']=[];manifest.write_text(json.dumps(data))
                args=dict(manifest_path=str(manifest),epochs=2,size=32,batch_size=2,images_per_epoch=2,
                    pretrained=False,device_name='cpu',seed=17,validation_grid='native_probability')
                with patch('ureter_semis.supervised_trainer.build_model',TinyModel):
                    train_supervised(**args,output_dir=str(root/'full'))
                    train_supervised(**args,output_dir=str(root/'staged'),stop_after_epoch=1)
                    train_supervised(**args,output_dir=str(root/'staged'),resume=str(root/'staged/last.pt'))
                a=torch.load(root/'full/last.pt',weights_only=True);b=torch.load(root/'staged/last.pt',weights_only=True)
                def equal(x,y):
                    if isinstance(x,torch.Tensor):self.assertTrue(torch.equal(x,y))
                    elif isinstance(x,dict):
                        self.assertEqual(x.keys(),y.keys())
                        for key in x:equal(x[key],y[key])
                    elif isinstance(x,(list,tuple)):
                        self.assertEqual(len(x),len(y))
                        for u,v in zip(x,y):equal(u,v)
                    else:self.assertEqual(x,y)
                equal(a,b)
                self.assertEqual([r['optimizer_update_steps'] for r in a['history']],[1,1])
                self.assertEqual([r['sampled_training_images'] for r in a['history']],[2,2])
        finally:torch.set_num_threads(threads)

    def test_invalid_fixed_budget_rejected_before_model_construction(self):
        parent=Path('runs/test_tmp').resolve();parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            root=Path(directory);manifest=make_fixture(root);data=json.loads(manifest.read_text())
            data['train']['unlabeled']=[];manifest.write_text(json.dumps(data))
            for count in (1,3,6):
                with self.assertRaisesRegex(ValueError,'Fixed image budget'),patch('ureter_semis.supervised_trainer.build_model') as model:
                    train_supervised(str(manifest),str(root/'never'),epochs=1,size=32,batch_size=2,
                        images_per_epoch=count,pretrained=False,device_name='cpu')
                    model.assert_not_called()


if __name__=='__main__':unittest.main()
