import unittest
import cv2,numpy as np
from tools.fiber_reference_workspace import ROOT,read
from tools.fiber_weak_stage1 import GrowthPipeline,QualifiedSeedObserver
from tools.fiber_weak_blue_evidence import GrowthPipeline as Old
from tests.test_fiber_blue_outline import scene

class Stage1Tests(unittest.TestCase):
    def setUp(self):self.cfg=read(ROOT/'configs/weak_blue_evidence_control_20261103.json')
    def test_clean_exact_old(self):
        im,raw,p,_=scene();a=GrowthPipeline(self.cfg);b=Old(self.cfg,'weak_blue_extension_tip');a.run(im,raw,p);b.run(im,raw,p);np.testing.assert_array_equal(a.final_mask,b.final_mask)
    def test_two_stray_components_do_not_disable_subject(self):
        im,raw,p,_=scene()
        for x in [300,350]:raw[10:14,x:x+4]=2;p[2,10:14,x:x+4]=.99
        a=GrowthPipeline(self.cfg);b=Old(self.cfg,'weak_blue_extension_tip');a.run(im,raw,p);b.run(im,raw,p);self.assertTrue(a.outline['eligible']);self.assertFalse(b.outline['eligible']);self.assertEqual(sum(x['reason']=='qualified' for x in a.inner.inner.inner.inner.growth.observer.seeds(im,raw,p,None,99)[1]),1)
    def test_original_label_kept(self):
        im,raw,p,_=scene();raw[10:14,300:304]=2;p[2,10:14,300:304]=.99;g=GrowthPipeline(self.cfg);g.run(im,raw,p);q=next(x for x in g.inner.inner.inner.inner.growth.observer.seeds(im,raw,p,None,99)[1] if x['reason']=='qualified');self.assertEqual(q['label'],2)
    def test_raw_stone_and_input_protected(self):
        im,raw,p,_=scene();raw[:,220:225]=1;before=raw.copy();g=GrowthPipeline(self.cfg);g.run(im,raw,p);np.testing.assert_array_equal(raw,before);self.assertTrue(np.all(g.final_mask[raw==1]==1))
    def test_ambiguous_qualified_never_choose_largest(self):
        from tools.fiber_connected_blue_union import ConnectedBlue
        cfg=read(ROOT/self.cfg['reference_union_config']);q=QualifiedSeedObserver(read(ROOT/cfg['reference_aggressive_config']),'aggressive');im,raw,p,_=scene();im[300:320,:135]=(210,100,15);raw[300:320,:135]=2;p[2,300:320,:135]=.99;seeds,records=q.seeds(im,raw,p,None,99)
        if len(seeds)>1:self.assertEqual(q.chosen,0)
    def test_reset_clears_history(self):g=GrowthPipeline(self.cfg);g.reset();self.assertIsNone(g.tracker)
if __name__=='__main__':unittest.main()
