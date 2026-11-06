import copy,unittest
from unittest.mock import patch
import cv2,numpy as np
from tests.test_tapered_tangent_cap import scene
from tools.fiber_weak_stage1 import QualifiedSeedObserver
from experiments.g0_realtime_stage7_20261106.runtime import Runtime as V7
from experiments.g0_realtime_20261105.benchmark import geometry_record
from .runtime import Runtime


def exact(a,b):
    if isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b);return
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:exact(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):exact(x,y)
    else:assert a==b,(a,b)


class SeedReuseTests(unittest.TestCase):
    def compare(self,frame,raw,p,configure=None,prior=None,age=99):
        reference=V7();candidates=[Runtime(False),Runtime()]
        if configure:
            for r in (reference,*candidates):configure(r.observer)
        before=[v.copy() for v in (frame,raw,p)]
        expected=reference.observer.seeds(frame,raw,p,prior,age)
        for r in candidates:
            actual=r.observer.seeds(frame,raw,p,prior,age);exact(expected,actual)
            self.assertEqual((reference.observer.chosen,reference.observer.orientation),(r.observer.chosen,r.observer.orientation))
            self.assertIs(r._captured[0],frame)
        for a,b in zip(before,(frame,raw,p)):np.testing.assert_array_equal(a,b)
        return expected,candidates

    def test_actual_qualified_seed_and_private_no_global_rebinding(self):
        frame,raw,p,_=scene();original=QualifiedSeedObserver.seeds;cv_original=cv2.connectedComponentsWithStats
        expected,candidates=self.compare(frame,raw,p)
        self.assertEqual(len(expected[0]),1)
        self.assertIs(QualifiedSeedObserver.seeds,original);self.assertIs(cv2.connectedComponentsWithStats,cv_original)
        self.assertLess(candidates[1].seed_counts['coordinate_evaluated_pixels'],candidates[0].seed_counts['coordinate_evaluated_pixels'])

    def test_component_holes_border_diagonals_probability_and_stone(self):
        rng=np.random.default_rng(110)
        for shape in ((31,43),(83,97)):
            frame=rng.integers(0,256,(*shape,3),np.uint8);raw=np.zeros(shape,np.uint8)
            raw[:6,:7]=2;raw[1:3,1:3]=0;raw[-6:,-7:]=2
            for i in range(8,15):raw[i,i]=2
            raw[20:23,12:20]=1
            p=rng.random((3,*shape),dtype=np.float32)
            self.compare(frame,raw,p,lambda o:o.cfg['seed'].update(min_pixels_at400=1))
        frame,raw,p,_=scene();raw[:]=0;self.compare(frame,raw,p)

    def test_area_ties_and_32_33_component_budget(self):
        for count in (2,32,33):
            frame=np.zeros((100,100,3),np.uint8);raw=np.zeros((100,100),np.uint8);p=np.zeros((3,100,100),np.float32)
            for k in range(count):y,x=divmod(k,8);raw[y*10,x*10]=2;p[2,y*10,x*10]=1
            result,_=self.compare(frame,raw,p,lambda o:o.cfg['seed'].update(min_pixels_at400=1))
            if count==33:self.assertEqual(result[1],[dict(reason='seed_computation_budget',components=33,limit=32)])
            else:self.assertEqual([r['label'] for r in result[1]],list(range(1,count+1)))

    def test_inner_component_limit_and_weak_history_age(self):
        frame,raw,p,_=scene()
        self.compare(frame,raw,p,lambda o:o.cfg['seed'].update(max_components=0))
        def weak(o):o.cfg['seed'].update(normal_min_sections=99,weak_min_sections=1,weak_min_paired=0)
        for prior,age in ((None,0),(np.array([0.,200.,0.,200.,40.,40.]),0),(np.array([0.,200.,0.,200.,40.,40.]),4)):
            self.compare(frame,raw,p,weak,prior,age)

    def test_runtime_mask_geometry_and_seed_reuse_reset(self):
        frame,raw,p,_=scene();reference,candidate=V7(),Runtime()
        for i in range(3):
            reference.run(frame,raw,p,i/30,'fixture');candidate.run(frame,raw,p,i/30,'fixture')
            np.testing.assert_array_equal(reference.stage.final_mask,candidate.stage.final_mask)
            np.testing.assert_array_equal(reference.g0.final_mask,candidate.g0.final_mask)
            exact(geometry_record(reference.g0.info),geometry_record(candidate.g0.info))
            self.assertEqual(reference.counters,candidate.counters)
        self.assertEqual(candidate.seed_counts['outer_component_passes'],3)
        candidate.reset();self.assertFalse(any(candidate.seed_counts.values()))


if __name__=='__main__':unittest.main()
