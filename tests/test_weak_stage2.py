import json,unittest
import cv2,numpy as np
from tools.fiber_reference_workspace import ROOT,read
from tools.fiber_weak_stage2 import GrowthPipeline,native_gradient,evidence
from tools.fiber_weak_stage1 import GrowthPipeline as Previous
from tools.fiber_fast_observation import sample
from tests.test_fiber_blue_outline import scene

class Stage2Tests(unittest.TestCase):
    def setUp(self):self.cfg=read(ROOT/'configs/weak_blue_evidence_control_20261103.json')
    def test_full_image_evidence_bitexact(self):
        from tools.fiber_weak_blue_evidence import evidence as old
        im,raw,p,_=scene();q=dict(mu=np.array([.05,.3,.65]),v=np.array([.05,.3,.65])-1/3,bg=np.array([.5,.3,.2]),gap=.4)
        for a,b in zip(old(im,q,self.cfg['weak_blue']['evidence']),evidence(im,q,self.cfg['weak_blue']['evidence'],native_gradient(im))):np.testing.assert_array_equal(a,b)
    def test_body_and_protected_union_exact(self):
        for curved in [False,True]:
            im,raw,p,_=scene(curved);a=GrowthPipeline(self.cfg,'weak_blue_extension_tip');b=Previous(self.cfg);a.run(im,raw,p);b.run(im,raw,p);np.testing.assert_array_equal(a.layers['reachable_blue'],b.layers['reachable_blue']);self.assertTrue(np.all(a.final_mask[a.layers['union_fiber']>0]==2))
    def test_native_probe_reproduces_saved_diagnosis(self):
        folder=ROOT/'tests/fixtures/weak_blue_stage2';probes=read(folder/'terminal_sample_probe.json')['samples'];seeds=read(folder/'qualified_seeds.json')
        for key,probe in probes.items():
            vid,idx=key.split(':');im=cv2.imread(str(folder/f'{vid}_frame{int(idx):04d}_RGB.png'));pts=np.asarray(probe['points']);rgb,valid=sample(im,pts);grad,gv=sample(native_gradient(im),pts);self.assertTrue(valid.all() and gv.all());np.testing.assert_allclose(grad.ravel(),probe['native_image_gradient'],atol=1e-7)
            z=seeds[key];mu=np.asarray(z['mu'],np.float32);seed=dict(mu=mu,v=mu-1/3,bg=np.asarray(z['background_chroma'],np.float32),gap=z['relative_blue_advantage']);a=evidence(rgb.reshape(-1,1,3),seed,self.cfg['weak_blue']['evidence'],grad.reshape(-1,1));self.assertEqual(int((a[1]|a[2]).sum()),6);np.testing.assert_allclose(a[0].ravel(),probe['score'],atol=1e-6)
    def test_sample_order_is_not_spatial_gradient(self):
        im=np.zeros((30,30,3),np.uint8);im[:,15:]=200;pts=np.array([[14.,5.],[14.,10.],[14.,20.]]);rgb,_=sample(im,pts);edge,_=sample(native_gradient(im),pts);self.assertTrue(np.all(edge>0));self.assertFalse(native_gradient(rgb[:,None,:]).any())
    def test_black_gap_and_stone_not_filled(self):
        im,raw,p,_=scene();im[:,190:196]=0;raw[:,250:255]=1;g=GrowthPipeline(self.cfg,'weak_blue_extension_tip');g.run(im,raw,p);self.assertFalse(g.layers['reachable_blue'][:,196:].any());self.assertTrue(np.all(g.final_mask[raw==1]==1))
    def test_shape_mismatch_rejected(self):
        im,_,_,_=scene()
        with self.assertRaises(ValueError):evidence(im,dict(v=np.ones(3),mu=np.ones(3),bg=np.ones(3),gap=.1),self.cfg['weak_blue']['evidence'],np.zeros((3,1)))
    def test_reset(self):
        im,raw,p,_=scene();g=GrowthPipeline(self.cfg,'weak_blue_extension_tip');g.run(im,raw,p);g.reset();self.assertIsNone(g.gradient)
if __name__=='__main__':unittest.main()
