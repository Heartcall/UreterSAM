"""Post-hoc failure decomposition; reference-derived poses are NOT automatic results."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from tools.direct_fiber_template import observation,score_poses,fine_poses,visible

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/a1_direct_fiber_template_20261029'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def main():
    config=read(ROOT/'configs/direct_fiber_template_20261029.json');prior=read(OUT/'template.json');vm=read(ROOT/'datasets/ureter_semis_kidney_joint_20261028/video_validation_manifest.json')
    output={'scope':'Post-hoc diagnosis only. GT-PCA pose is reference-conditioned, not a certified optimum or automatic segmentation; no parameter changes','frames':[]}
    for row in vm['val']:
        rgb=np.asarray(Image.open(row['image']).convert('RGB'));frame=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR);target=np.asarray(Image.open(row['mask']));truth=target==2
        obs=observation(frame,prior,config);coarse=score_poses(obs,obs['poses'],prior,config);poses=fine_poses(obs['poses'],coarse['score'],config,obs['shape'],obs['boundary']);scores=score_poses(obs,poses,prior,config)
        a=config['acceptance'];valid=(scores['inside']>=a['inside_appearance_min'])&(scores['contrast']>=a['region_contrast_min'])&(scores['boundary']>=a['boundary_contrast_min'])&(scores['rgb']>=a['rgb_edge_min'])&(scores['support']>=a['fov_support_min'])
        best=int(np.argmax(scores['score']))
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV);bins=prior['histogram_bins'];h=(hsv[:,:,0].astype(int)*bins//180).clip(0,bins-1);s=(hsv[:,:,1].astype(int)*bins//256).clip(0,bins-1);evidence=np.asarray(prior['appearance_probability'])[h,s]>=a['visible_appearance_min']
        yy,xx=np.where(truth);points=np.column_stack([xx,yy]).astype(np.float32);origin=points.mean(0);_,axis=cv2.PCACompute(points,mean=None,maxComponents=1);u=axis[0];projection=(points-origin)@u
        low,high=np.percentile(projection,[.5,99.5]);ends=[origin+u*t for t in (low,high)];edge=lambda p:min(p[0],p[1],frame.shape[1]-1-p[0],frame.shape[0]-1-p[1])
        if edge(ends[0])>edge(ends[1]):u=-u;projection=(points-origin)@u;low,high=np.percentile(projection,[.5,99.5])
        entry=origin+u*low;pose=[float(entry[0]),float(entry[1]),float(np.arctan2(u[1],u[0])),float(high-low)]
        conditional,template=visible(frame,pose,True,prior,config);dice=lambda m:2*int((m&truth).sum())/max(1,int(m.sum()+truth.sum()))
        output['frames'].append({'sample_id':row['sample_id'],'video_id':row['video_id'],'frame_index':row['frame_index'],
            'valid_fine_image_candidates':int(valid.sum()),'selected_image_candidate_valid':bool(valid[best]),
            'appearance_recall_on_reference_fiber':float(evidence[truth].mean()),'reference_fiber_pixels':int(truth.sum()),
            'reference_fiber_with_appearance_support':int((evidence&truth).sum()),
            'reference_PCA_pose':pose,'reference_PCA_template_dice':dice(template),'reference_PCA_visible_dice':dice(conditional)})
    output['summary']={'frames_where_selected_rejected_but_valid_alternative_exists':sum(not r['selected_image_candidate_valid'] and r['valid_fine_image_candidates']>0 for r in output['frames']),
        'frames_no_valid_fine_image_candidate':sum(r['valid_fine_image_candidates']==0 for r in output['frames']),
        'pooled_reference_fiber_appearance_recall':sum(r['reference_fiber_with_appearance_support'] for r in output['frames'])/sum(r['reference_fiber_pixels'] for r in output['frames']),
        'reference_PCA_template_dice_mean':float(np.mean([r['reference_PCA_template_dice'] for r in output['frames']])),
        'reference_PCA_visible_dice_mean':float(np.mean([r['reference_PCA_visible_dice'] for r in output['frames']]))}
    (OUT/'failure_diagnosis.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    # Fixed representative indices show both cases and the known missed small-stone frame.
    choices=[('video_1',2),('video_1',721),('video_2',149),('video_2',1822)];names=['reference','raw_classwise','image_template','soft_template','causal_template']
    canvas=Image.new('RGB',(1300,4*286+30),'#20252b');draw=ImageDraw.Draw(canvas);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',13)
    draw.text((5,5),'Reference / raw classwise / direct image / neural soft / causal. Red=stone, cyan=fiber.',font=font,fill='white')
    for j,(vid,index) in enumerate(choices):
        row=next(r for r in vm['val'] if r['video_id']==vid and r['frame_index']==index);rgb=np.asarray(Image.open(row['image']).convert('RGB'))
        for col,n in enumerate(names):
            path=Path(row['mask']) if n=='reference' else OUT/vid/n/f'{index:06d}.png'
            if not path.exists():continue
            mask=np.asarray(Image.open(path));overlay=rgb.copy()
            for k,color in [(1,[235,40,40]),(2,[20,210,235])]:
                selected=mask==k;overlay[selected]=(.48*rgb[selected]+.52*np.asarray(color)).astype(np.uint8)
            canvas.paste(Image.fromarray(overlay).resize((260,260)),(col*260,30+j*286))
        draw.text((5,291+j*286),f'{vid} display frame{index+1}',font=font,fill='white')
    canvas.save(OUT/'reviewed_examples.jpg',quality=94)
    print(json.dumps(output['summary'],indent=2))


if __name__=='__main__':main()
