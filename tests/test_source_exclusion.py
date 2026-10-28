from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from tools.exclude_stone_new import build
from ureter_semis.experiment import require_training_access, sha256
from ureter_semis.source_policy import require_training_sources


class SourceExclusionTests(unittest.TestCase):
    def test_excluded_source_rejected_even_if_reviewed_or_development_authorized(self):
        for name in ['stone_new_v4','STONE-NEW','stone_new.v4i.yolov8']:
            for split in ['labeled','unlabeled','reserved_unlabeled']:
                m={'training_ready':True,'train':{'labeled':[],'unlabeled':[]},'reserved_unlabeled':[]}
                row={'source_dataset':name,'sample_id':'sample'}
                (m['reserved_unlabeled'] if split=='reserved_unlabeled' else m['train'][split]).append(row)
                with self.assertRaisesRegex(ValueError,'stone_new is excluded'):
                    require_training_access(m,True)
        require_training_sources({'train':{'labeled':[{'source_dataset':'kidney_stones_fiber_v1'}]},
                                  'val':[{'source_dataset':'stone_new_v4'}]})

    def test_renamed_known_excluded_image_is_rejected_by_hash(self):
        registry=json.loads(Path('annotations/excluded_training_sources_20261027.json').read_text(encoding='utf-8'))
        m={'train':{'labeled':[{'source_dataset':'renamed','image':'renamed.png',
                               'image_sha256':registry['excluded_image_sha256'][0]}]}}
        with self.assertRaisesRegex(ValueError,'stone_new is excluded'):
            require_training_sources(m)

    def test_filter_preserves_remaining_bytes_validation_and_pending_status(self):
        parent=Path('runs/test_tmp').resolve();parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temp:
            root=Path(temp);rows=[]
            for i,source in enumerate(['kidney_stones_fiber_v1','stone_new_v4']*2):
                image=root/f'image{i}.png';mask=root/f'mask{i}.png'
                Image.new('RGB',(16,16),(30+i,40,50)).save(image)
                a=np.zeros((16,16),np.uint8);a[:4]=1;a[6:8]=2;Image.fromarray(a).save(mask)
                rows.append({'sample_id':f'sample{i}','source_dataset':source,'image':str(image),
                    'mask':str(mask),'image_sha256':sha256(image),'mask_sha256':sha256(mask),
                    'review_status':'pending','source_group_verified':False})
            val=[{'image':'validation.png','mask':'validation_mask.png','video_id':'video_1'}]
            manifest={'root':str(root),'training_ready':False,'train':{'labeled':rows[:2],'unlabeled':[]},
                      'reserved_unlabeled':rows[2:],'val':val,'counts':{'excluded':{'historical':5}},
                      'provenance':{},'evaluation_limitations':[]}
            previous=root/'previous.json';previous.write_text(json.dumps(manifest))
            result=build(previous,root/'new');new=json.loads(result.read_text(encoding='utf-8'))
            self.assertEqual(len(new['train']['labeled']),1);self.assertEqual(len(new['reserved_unlabeled']),1)
            self.assertEqual(new['counts']['raw_train_pool'],2)
            self.assertEqual(new['val'][0]['video_id'],'video_1')
            self.assertEqual(new['val'][0]['image'],str(root/'validation.png'))
            self.assertFalse(new['training_ready']);self.assertEqual(new['train']['labeled'][0],rows[0])
            self.assertEqual(new['counts']['removed_stone_new']['labeled'],1)
            self.assertEqual(json.loads(previous.read_text()),manifest)
            require_training_sources(new)
            with self.assertRaises(FileExistsError):build(previous,root/'new')


if __name__=='__main__':unittest.main()
