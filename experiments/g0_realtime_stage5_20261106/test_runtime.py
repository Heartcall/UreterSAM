import copy
import json
import unittest

import numpy as np

from tools.fiber_reference_workspace import ROOT,sha256
from tests.test_tapered_tangent_cap import scene
from experiments.g0_realtime_20261105.compare import close
from experiments.g0_realtime_20261105.benchmark import geometry_record
from experiments.g0_realtime_stage3_20261105.pixels import native_gradient
from experiments.g0_realtime_stage4_20261105.runtime import Runtime as V4
from .native import Native
from .runtime import Runtime,rectangle


class NativeTests(unittest.TestCase):
    def test_binary_identity_matches_final_source_and_build_flags(self):
        old=V4();native=Native(old.chroma)
        identity=json.loads((native.path.parent/'build_identity.json').read_text(encoding='utf-8-sig'))
        self.assertEqual(identity['dll_sha256'].lower(),sha256(native.path))
        self.assertEqual(identity['source_sha256'].lower(),sha256(ROOT/'experiments/g0_realtime_stage5_20261106/kernels.cpp'))
        self.assertIn('/fp:strict',identity['flags']);self.assertIn('/NODEFAULTLIB',identity['flags'])

    def test_full_evidence_scores_and_color_predicates_are_exact(self):
        old=V4();backend=Native(old.chroma)
        frame,raw,p,_=scene();seed=old.observer.seeds(frame,raw,p,None,99)[0][0]
        rng=np.random.default_rng(41)
        for index in range(10):
            image=rng.integers(0,256,(400,400,3),np.uint8)
            if index==0:image=frame
            if index==1:image[:]=0
            if index==2:image[:]=255
            if index==3:image[:]=(210,100,15)
            gradient=native_gradient(image)
            a=old.colors.fast_evidence(image,seed,old.stage.q['evidence'],gradient)
            b=backend.evidence(image,seed,old.stage.q['evidence'],gradient)
            for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
            for weak in (False,True):
                np.testing.assert_array_equal(old.observer.color(image.reshape(-1,3),seed,weak),
                                              backend.color(old.observer,image.reshape(-1,3),seed,weak))
        values=np.arange(0,256,17,dtype=np.uint8)
        colors=np.stack(np.meshgrid(values,values,values,indexing='ij'),-1).reshape(-1,3)
        for weak in (False,True):
            np.testing.assert_array_equal(old.observer.color(colors,seed,weak),backend.color(old.observer,colors,seed,weak))

    def test_floating_probes_and_float64_seed_use_reference_fallback(self):
        old=V4();backend=Native(old.chroma)
        frame,raw,p,_=scene();seed=old.observer.seeds(frame,raw,p,None,99)[0][0]
        rng=np.random.default_rng(43)
        for image in (rng.random((61,1,3))*255,frame):
            altered=copy.deepcopy(seed)
            if image.shape[1]>1:
                for key in ('mu','bg','v'):altered[key]=altered[key].astype(np.float64)
            gradient=native_gradient(image)
            for x,y in zip(old.colors.fast_evidence(image,altered,old.stage.q['evidence'],gradient),
                           backend.evidence(image,altered,old.stage.q['evidence'],gradient)):
                np.testing.assert_array_equal(x,y)
            np.testing.assert_array_equal(old.observer.color(image.reshape(-1,3),altered),
                                          backend.color(old.observer,image.reshape(-1,3),altered))
        self.assertEqual(backend.counts['evidence_calls'],0)
        with self.assertRaises(ValueError):backend.color(old.observer,np.zeros((5000,4),np.uint8),seed)

    def test_projection_domain_and_roi_cover_original_pixel_rules(self):
        backend=Native(V4().chroma);rng=np.random.default_rng(47);shape=(71,93)
        for _ in range(80):
            angle=rng.uniform(-np.pi,np.pi);u=np.array([np.cos(angle),np.sin(angle)])
            n=np.array([-u[1],u[0]]);o=rng.uniform(-30,120,2);offset=(-2,3)
            yy,xx=np.indices(shape);yy=yy+offset[0];xx=xx+offset[1]
            t=(xx-o[0])*u[0]+(yy-o[1])*u[1];v=(xx-o[0])*n[0]+(yy-o[1])*n[1]
            nt,nv=backend.project(shape,o,u,n,offset)
            np.testing.assert_array_equal(t,nt);np.testing.assert_array_equal(v,nv)
            lo,hi=-20.,80.;center=12.;lateral=9.
            domain=(t>=lo)&(t<=hi)&(abs(v-center)<=lateral)
            np.testing.assert_array_equal(domain,backend.domain(shape,o,u,n,lo,hi,center,lateral,offset))
            # Native ROI uses actual absolute coordinates, without shifting o.
            yy,xx=np.indices(shape);t=(xx-o[0])*u[0]+(yy-o[1])*u[1];v=(xx-o[0])*n[0]+(yy-o[1])*n[1]
            expected=(t>=lo)&(t<=hi)&(abs(v-center)<=lateral)
            y0,y1,x0,x1=rectangle(shape,o,u,n,lo,hi,center,lateral)
            actual=np.zeros(shape,bool)
            if y0<y1 and x0<x1:actual[y0:y1,x0:x1]=backend.domain((y1-y0,x1-x0),o,u,n,lo,hi,center,lateral,(y0,x0))
            np.testing.assert_array_equal(expected,actual)

    def test_stateful_domain_caps_masks_inputs_and_area_fallback_match(self):
        old,new=V4(),Runtime()
        original=old.expand
        def capture(frame,raw,probability,meta,B,core,u):
            o=np.asarray(old.stage.outline['context']['origin']);n=np.array([-u[1],u[0]])
            yy,xx=np.indices(raw.shape);t=(xx-o[0])*u[0]+(yy-o[1])*u[1];v=(xx-o[0])*n[0]+(yy-o[1])*n[1]
            bt,bv=t[B>0],v[B>0];cfg=old.stage.q['search'];scale=min(raw.shape)/400
            limit=min(cfg['maximum_forward_at400']*scale,cfg['maximum_forward_diagonal_fraction']*np.hypot(*raw.shape))
            lateral=(np.quantile(bv,.98)-np.quantile(bv,.02))*cfg['lateral_width_multiplier']/2+cfg['lateral_margin_at400']*scale
            old.test_domain=(t>=bt.min()-2*scale)&(t<=float(bt.max())+limit)&(abs(v-float(np.median(bv)))<=lateral)
            original(frame,raw,probability,meta,B,core,u)
        old.expand=capture
        for index,mode in enumerate(('plain','gap','stone','none','budget','plain')):
            frame,raw,p,_=scene();old.test_domain=None
            if mode=='gap':frame[198:202,200:204]=0
            if mode=='stone':raw[198:202,220:224]=1
            if mode=='none':raw[:]=0
            for runtime in (old,new):runtime.stage.q['search']['maximum_component_fraction']=0. if mode=='budget' else .25
            before=[x.copy() for x in (frame,raw,p)]
            old.run(frame,raw,p,index/30,'fixture');new.run(frame,raw,p,index/30,'fixture')
            if old.test_domain is None:self.assertIsNone(new.last_domain)
            else:np.testing.assert_array_equal(old.test_domain,new.last_domain)
            np.testing.assert_array_equal(old.stage.final_mask,new.stage.final_mask)
            np.testing.assert_array_equal(old.g0.final_mask,new.g0.final_mask)
            self.assertTrue(close(geometry_record(old.g0.info),geometry_record(new.g0.info)))
            self.assertTrue(close(old.stage.outline['cap'],new.stage.outline['cap']))
            for x,y in zip(before,(frame,raw,p)):np.testing.assert_array_equal(x,y)
        self.assertGreater(new.backend.counts['evidence_calls'],0)
        self.assertLess(new.s2_spatial['domain_roi_pixels'],new.s2_spatial['domain_full_pixels'])
        new.reset();self.assertFalse(any(new.backend.counts.values()))


if __name__=='__main__':unittest.main()
