import unittest,numpy as np
from tools.fiber_reference_workspace import ROOT,read
from tools.fiber_blue_outline import Outline
from tools.fiber_blue_outline_repaired import RepairedOutline

class OutlineEvidenceRepairTests(unittest.TestCase):
    def test_diagonal_neural_core_fragment_does_not_clear_RGB_body(self):
        cfg=read(ROOT/'configs/blue_outline_control_20261102_v2.json');body=np.zeros((400,400),np.uint8);body[100:120,100:180]=1;core=body>0;core[99,99:101]=True;image=np.full((400,400,3),(45,50,80),np.uint8);image[body>0]=(210,100,15);image[99,99]=(210,100,15);raw=np.where(core,2,0).astype(np.uint8);meta=dict(seeds=[dict(reason='qualified',mu=(np.array([15,100,210])/325).tolist(),background_chroma=(np.array([80,50,45])/175).tolist(),relative_blue_advantage=.5)])
        before,_=Outline(cfg).evidence(image,raw,body,core,meta,1.);after,_=RepairedOutline(cfg).evidence(image,raw,body,core,meta,1.);self.assertFalse(before.any());self.assertTrue(after[body>0].all());self.assertEqual(after[99,99],0)
    def test_repair_changes_no_frozen_geometry_or_evaluation_parameter(self):
        old=read(ROOT/'configs/blue_outline_control_20261102.json');new=read(ROOT/'configs/blue_outline_control_20261102_v2.json')
        for k in ['outline','geometry_quality','connected_blue','extent','aggressive_parameters','execution','data']:self.assertEqual(old[k],new[k])

if __name__=='__main__':unittest.main()
