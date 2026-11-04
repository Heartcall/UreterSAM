import copy,json
from pathlib import Path
import unittest
import cv2
import numpy as np
from tools.fiber_connected_blue_growth import ConnectedBlueGrowth,GrowthPipeline,color_ok,chroma

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'configs/fiber_connected_blue_growth_20261101.json').read_text(encoding='utf-8'))
T1=json.loads((ROOT/CFG['t1']['config']).read_text(encoding='utf-8'))


def scene(y=200):
    f=np.full((400,400,3),70,np.uint8);truth=np.zeros((400,400),np.uint8)
    cv2.fillPoly(truth,[np.array([[0,y-20],[210,y-14],[210,y+14],[0,y+20]])],1)
    f[truth>0]=[160,70,20];f[(truth>0)&(np.indices(truth.shape)[1]>=130)]=[64,28,8]
    m=np.zeros_like(truth);m[(truth>0)&(np.indices(truth.shape)[1]<=110)]=2
    p=np.zeros((3,400,400),np.float32);p[0]=.99;p[1]=.005;p[2]=.005;p[:,m==2]=np.array([.005,.005,.99])[:,None]
    return f,m,p,truth


class ConnectedGrowthTests(unittest.TestCase):
    def test_dark_blue_lowP_continuous_region_is_candidate(self):
        f,m,p,t=scene();a,b,c,r=ConnectedBlueGrowth(CFG).grow(f,m,p)
        self.assertTrue(((a>0)&(p[2]<.1)&(np.indices(m.shape)[1]>135)).any(),r['reason'])
        self.assertTrue(np.all(m[a>0]==0));self.assertFalse(r['apply_to_mask'])

    def test_stone_is_barrier_before_connectivity(self):
        f,m,p,t=scene();m[:,140:150]=1;a,proposal,conflict,r=ConnectedBlueGrowth(CFG).grow(f,m,p)
        self.assertFalse(a[:,140:].any());self.assertTrue(conflict[:,140:150].any());self.assertFalse(np.any(a[m==1]))

    def test_gap_and_detached_blue_do_not_connect(self):
        f,m,p,t=scene();f[:,145:155]=70;a,_,_,_=ConnectedBlueGrowth(CFG).grow(f,m,p)
        self.assertFalse(a[:,155:].any())

    def test_two_qualified_seeds_are_ambiguous(self):
        f,m,p,t=scene();f2,m2,p2,_=scene(y=80);f[55:105]=f2[55:105];m[55:105]=m2[55:105];p[:,55:105]=p2[:,55:105]
        a,_,_,r=ConnectedBlueGrowth(CFG).grow(f,m,p);self.assertFalse(a.any());self.assertEqual(r['reason'],'ambiguous_trusted_seeds')

    def test_neutral_high_confidence_seed_cannot_establish_blue_identity(self):
        f,m,p,t=scene();f[t>0]=120;a,_,_,r=ConnectedBlueGrowth(CFG).grow(f,m,p)
        self.assertFalse(a.any())

    def test_immutable_seed_chroma_and_noise_unknown(self):
        mu=np.array([.08,.28,.64]);sigma=np.array([.015]*3)
        rgb=np.array([[160,70,20],[64,28,8],[70,70,70],[0,0,0]],np.uint8)
        q=color_ok(rgb,mu,sigma,CFG['appearance']);self.assertEqual(q.tolist(),[True,True,False,False])

    def test_no_seed_t1_prediction_cannot_create_candidate(self):
        f,m,p,t=scene();m[:]=0;p[2]=.0001;pipeline=GrowthPipeline(CFG,'connected_rgb_t1',T1,True)
        pipeline.tracker.groups['shaft']=np.array([0,200,0,40.]);pipeline.tracker.groups['endpoint']=np.array([200.,20.]);pipeline.tracker.velocity={k:np.zeros_like(v) for k,v in pipeline.tracker.groups.items()}
        a,_,_,r=pipeline.run(f,m,p,0,'video');self.assertFalse(a.any());self.assertEqual(r['tips'],[])

    def test_t1_advice_cannot_change_current_growth(self):
        f,m,p,t=scene();a=GrowthPipeline(CFG,'connected_rgb',T1,True).run(f,m,p,0,'v')[0]
        pipeline=GrowthPipeline(CFG,'connected_rgb_t1',T1,True);b= pipeline.run(f,m,p,0,'v')[0]
        np.testing.assert_array_equal(a,b);pipeline.reset();self.assertIsNone(pipeline.tracker.source)

    def test_internal_brightness_transition_is_not_forced_tip(self):
        f,m,p,t=scene();g=ConnectedBlueGrowth(CFG);seed={'mu':np.array([.08,.28,.64]),'sigma':np.array([.015]*3)}
        branch={'sign':1,'sections':[{'center':np.array([128.,200.]),'width':36.}], 'final_u':np.array([1.,0.]),'stop_reason':'current_RGB_evidence_missing','abnormal':None}
        r=g.tip(f,m,seed,branch);self.assertNotEqual(r['state'],'visible_end_candidate');self.assertIsNone(r['xy'])

    def test_search_limit_cannot_become_tip(self):
        f,m,p,t=scene();g=ConnectedBlueGrowth(CFG);branch={'sign':1,'sections':[{'center':np.array([200.,200.]),'width':30.}], 'final_u':np.array([1.,0.]),'stop_reason':'search_limit','abnormal':None}
        self.assertIsNone(g.tip(f,m,{},branch)['xy'])

    def test_stateless_no_mutation_and_actual_pixels_preserved(self):
        f,m,p,t=scene();before=[x.copy() for x in [f,m,p]];a,_,_,r=ConnectedBlueGrowth(CFG).grow(f,m,p)
        for x,y in zip([f,m,p],before):np.testing.assert_array_equal(x,y)
        virtual=m.copy();virtual[(a>0)&(m==0)]=2
        self.assertTrue((virtual[m==2]==2).all());json.dumps(r,allow_nan=False)

    def test_raw_arm_no_growth_or_state(self):
        f,m,p,t=scene();a,_,_,r=GrowthPipeline(CFG,'raw2',T1,True).run(f,m,p)
        self.assertFalse(a.any());self.assertEqual(r['reason'],'raw_control')


if __name__=='__main__':unittest.main()
