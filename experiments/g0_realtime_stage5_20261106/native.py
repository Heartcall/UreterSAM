"""ctypes adapter; small/floating probes retain the frozen Python functions."""
import ctypes
import math
from pathlib import Path

import numpy as np

from tools.fiber_reference_workspace import ROOT, sha256
from experiments.g0_realtime_stage3_20261105.pixels import PixelMath


class Native:
    def __init__(self, get_chroma, path=None):
        self.path=Path(path or ROOT/'runs/g0_realtime_stage5_20261106/native_final/g0_kernels.dll')
        if not self.path.is_file():
            raise RuntimeError('Build the independent native experiment with its build.ps1 first')
        self.sha256=sha256(self.path)
        self.dll=ctypes.CDLL(str(self.path)); self.chroma=get_chroma; self.fallback=PixelMath(get_chroma)
        ptr=ctypes.c_void_p; integer=ctypes.c_int
        self.dll.s2_evidence.argtypes=[ptr,ptr,ptr,integer,ptr,ptr,ptr,ptr,ptr,ptr]
        self.dll.color_predicate.argtypes=[ptr,integer,ptr,ptr,ptr]
        self.dll.project.argtypes=[integer,integer,integer,integer,ptr,ptr,ptr]
        self.dll.domain.argtypes=[integer,integer,integer,integer,ptr,ptr,ptr]
        for name in ('s2_evidence','color_predicate','project','domain'):
            getattr(self.dll,name).restype=None
        self.counts=dict(evidence_calls=0,color_calls=0,projection_pixels=0,domain_pixels=0)

    def reset(self):
        for key in self.counts:self.counts[key]=0

    @staticmethod
    def pointer(a): return a.ctypes.data

    @staticmethod
    def seed(seed):
        return np.asarray([*seed['mu'],*seed['bg'],*seed['v'],np.linalg.norm(seed['v'])],np.float32)

    @staticmethod
    def supported(seed):
        return all(np.asarray(seed[k]).dtype==np.float32 and np.asarray(seed[k]).shape==(3,)
                   for k in ('mu','bg','v'))

    def evidence(self,frame,seed,cfg,gradient=None):
        if (frame.shape[1]<32 or frame.shape[0]*frame.shape[1]<4096 or gradient is None
                or not self.supported(seed)):
            return self.fallback.evidence(frame,seed,cfg,gradient)
        c,noise=self.chroma(frame); c=np.ascontiguousarray(c);noise=np.ascontiguousarray(noise)
        edge=np.asarray(gradient,np.float32)
        if edge.shape!=frame.shape[:-1]:raise ValueError('Gradient/sample shape mismatch')
        edge=np.ascontiguousarray(edge); params=self.seed(seed)
        if not np.isfinite(params).all():return self.fallback.evidence(frame,seed,cfg,gradient)
        q=np.asarray([cfg['cosine_floor'],1-cfg['cosine_floor'],cfg['minimum_blue_excess'],
                      max(cfg['minimum_blue_excess'],cfg['blue_seed_gap_fraction']*seed['gap']),
                      cfg['distance_margin'],2*cfg['distance_margin'],cfg['max_chroma_noise'],
                      cfg['minimum_chroma_norm'],cfg['minimum_gradient'],cfg['weak_score'],
                      cfg['strong_score'],max(seed['gap'],cfg['minimum_blue_excess'])],np.float32)
        score=np.empty(frame.shape[:2],np.float32)
        strong=np.empty(frame.shape[:2],bool);weak=np.empty_like(strong);uncertain=np.empty_like(strong)
        self.dll.s2_evidence(self.pointer(c),self.pointer(noise),self.pointer(edge),score.size,
                             self.pointer(params),self.pointer(q),self.pointer(score),
                             self.pointer(strong),self.pointer(weak),self.pointer(uncertain))
        self.counts['evidence_calls']+=1
        return score,strong,weak,uncertain,edge,noise

    def color(self,observer,bgr,seed,weak=False):
        if bgr.shape[-1]!=3:raise ValueError('Expected three BGR channels')
        if bgr.dtype!=np.uint8 or int(np.prod(bgr.shape[:-1]))<4096 or not self.supported(seed):
            return self.fallback.color(observer,bgr,seed,weak)
        image=np.ascontiguousarray(bgr);p=self.seed(seed)
        if not np.isfinite(p).all():return self.fallback.color(observer,bgr,seed,weak)
        cfg=observer.cfg['color'];suffix='weak' if weak else 'normal'
        q=np.asarray([cfg['max_noise'],cfg['blue_gap_'+suffix+'_fraction']*seed['gap'],
                      math.cos(math.radians(cfg['angle_'+suffix+'_degrees'])),
                      cfg['background_distance_margin_'+suffix]],np.float32)
        result=np.empty(bgr.shape[:-1],bool)
        self.dll.color_predicate(self.pointer(image),result.size,self.pointer(p),self.pointer(q),self.pointer(result))
        self.counts['color_calls']+=1
        return result

    def project(self,shape,origin,axis,normal,offset=(0,0)):
        t=np.empty(shape,np.float64);v=np.empty(shape,np.float64)
        p=np.asarray([*origin,*axis,*normal],np.float64)
        self.dll.project(*shape,*offset,self.pointer(p),self.pointer(t),self.pointer(v))
        self.counts['projection_pixels']+=t.size
        return t,v

    def domain(self,shape,origin,axis,normal,lo,hi,center,lateral,offset=(0,0)):
        result=np.empty(shape,bool);p=np.asarray([*origin,*axis,*normal],np.float64)
        bounds=np.asarray([lo,hi,center,lateral],np.float64)
        self.dll.domain(*shape,*offset,self.pointer(p),self.pointer(bounds),self.pointer(result))
        self.counts['domain_pixels']+=result.size
        return result
