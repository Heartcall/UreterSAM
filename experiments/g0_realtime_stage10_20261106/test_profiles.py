import unittest
import numpy as np
from tests.test_tapered_tangent_cap import scene
from experiments.g0_realtime_stage2_20261105.runtime import grouped_profiles as reference
from experiments.g0_realtime_stage7_20261106.runtime import Runtime as V7
from experiments.g0_realtime_20261105.benchmark import geometry_record
from .profiles import grouped_profiles
from .profile_runtime import ProfileOnlyRuntime,Runtime
from .test_seeds import exact


class ProfileTests(unittest.TestCase):
    def test_exact_random_sparse_tied_singleton_profiles(self):
        rng=np.random.default_rng(111)
        for shape in ((1,1),(7,9),(53,71),(400,400)):
            for density in (.005,.2,1):
                body=(rng.random(shape)<density).astype(np.uint8);body[0,0]=1
                for u in (np.array([1.,0.]),np.array([0.,-1.]),np.array([.6,.8]),rng.normal(size=2)):
                    origin=rng.normal(size=2)*100
                    for step,margin in ((.5,0),(1.,1.5),(3.2,4.)):
                        exact(reference(body,origin,u,step,margin),grouped_profiles(body,origin,u,step,margin))

    def test_private_runtime_bindings_and_end_to_end_equality(self):
        baseline,one,two=V7(),ProfileOnlyRuntime(),Runtime()
        self.assertIsNot(baseline.stage.fit_cap.__func__.__globals__['profiles'],grouped_profiles)
        self.assertIs(one.stage.fit_cap.__func__.__globals__['profiles'],grouped_profiles)
        self.assertIs(two.stage.side.run.__func__.__globals__['profiles'],grouped_profiles)
        frame,raw,p,_=scene()
        for i in range(3):
            for r in (baseline,one,two):r.run(frame,raw,p,i/30,'fixture')
            for r in (one,two):
                np.testing.assert_array_equal(baseline.stage.final_mask,r.stage.final_mask)
                np.testing.assert_array_equal(baseline.g0.final_mask,r.g0.final_mask)
                exact(geometry_record(baseline.g0.info),geometry_record(r.g0.info))
                self.assertEqual(baseline.counters,r.counters)

    def test_empty_profile_keeps_reference_exception(self):
        for fn in (reference,grouped_profiles):
            with self.assertRaises(ValueError):fn(np.zeros((5,7),np.uint8),np.zeros(2),np.array([1.,0.]),1.)


if __name__=='__main__':unittest.main()
