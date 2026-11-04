"""Engineering correction only: boundary support rooted in qualified RGB body."""
import numpy as np,cv2
from .fiber_blue_outline import Outline,GrowthPipeline as OriginalPipeline,ARMS,LAYERS
from .fiber_connected_blue_union import rooted

class RepairedOutline(Outline):
    def evidence(self,frame,raw,body,core,meta,s):
        q=next(z for z in meta['seeds'] if z['reason']=='qualified');mu=np.asarray(q['mu'],np.float32);seed=dict(mu=mu,v=mu-1/3,bg=np.asarray(q['background_chroma'],np.float32),gap=q['relative_blue_advantage']);distance=cv2.distanceTransform((body==0).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE);band=(distance<=self.cfg['evidence']['band_distance_at400']*s)&(body==0)&(raw!=1)
        image=frame.astype(np.float32)/255.;gx=cv2.Sobel(image,cv2.CV_32F,1,0,ksize=3)/8;gy=cv2.Sobel(image,cv2.CV_32F,0,1,ksize=3)/8;gradient=np.sqrt((gx*gx+gy*gy).sum(-1)/3);eligible=body.astype(bool).copy();eligible[band]=self.relaxed.color(frame[band],seed)&(gradient[band]>=self.cfg['evidence']['minimum_RGB_gradient']);eligible[raw==1]=False
        result,_=rooted(eligible,(body>0)&core,1)
        if not np.all(result[body>0]):raise ValueError('Qualified current blue body lost from evidence domain')
        return result,seed

class GrowthPipeline(OriginalPipeline):
    def __init__(self,cfg,arm,t1_config=None,sequential=False):
        super().__init__(cfg,arm,t1_config,sequential);self.fitter=RepairedOutline(cfg)
