import unittest
import numpy as np
from experiments.g0_realtime_stage7_20261106.color import from_chroma
from experiments.g0_realtime_stage7_20261106.runtime import Runtime as V7
from .runtime import indexed_color,Runtime
from tests.test_tapered_tangent_cap import scene


class IndexedRegionTests(unittest.TestCase):
    def test_indices_preserve_pixel_order_dtype_and_strided_inputs(self):
        runtime=V7();frame,raw,p,_=scene()
        seed=runtime.observer.seeds(frame,raw,p,None,99)[0][0]
        rng=np.random.default_rng(109)
        for dtype in (np.float32,np.float64):
            color=rng.random((51,73,3)).astype(dtype);noise=rng.random((51,73)).astype(dtype)
            for stride in (1,2,-1):
                c=color[::stride,::stride];n=noise[::stride,::stride]
                for density in (0,.01,.3,1):
                    region=rng.random(n.shape)<density
                    np.testing.assert_array_equal(indexed_color(runtime.observer,c,n,region,seed),
                                                  from_chroma(runtime.observer,c[region],n[region],seed))

    def test_runtime_current_frame_threshold_and_fallbacks(self):
        base,candidate=V7(),Runtime();frame,raw,p,_=scene()
        seed=base.observer.seeds(frame,raw,p,None,99)[0][0]
        for same,dtype,pixels in ((True,np.uint8,4095),(True,np.uint8,4096),
                                  (True,np.uint8,frame.shape[0]*frame.shape[1]),
                                  (False,np.uint8,8000),(True,np.float32,8000)):
            image=frame.astype(dtype);region=np.zeros(image.shape[:2],bool);region.flat[:pixels]=True
            for r in (base,candidate):r.frame=image if same else image.copy();r._chroma=None
            np.testing.assert_array_equal(base.region_color(image,region,seed),candidate.region_color(image,region,seed))
        self.assertEqual(base.extra_counts,candidate.extra_counts)


if __name__=='__main__':unittest.main()
