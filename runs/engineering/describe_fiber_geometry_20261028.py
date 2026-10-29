"""Post-hoc error location and representative overlays; never change fit rules."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/a1_fiber_geometry_20261028'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def load(path):
    with Image.open(path) as im:return np.asarray(im).copy()


def counts(target,raw,other,where):
    valid=where&(target!=255);truth=target==2
    return {'lost_fiber_tp':int(((raw==2)&(other!=2)&truth&valid).sum()),
        'gained_fiber_tp':int(((raw!=2)&(other==2)&truth&valid).sum()),
        'added_fiber_fp':int(((raw!=2)&(other==2)&~truth&valid).sum()),
        'removed_fiber_fp':int(((raw==2)&(other!=2)&~truth&valid).sum())}


def main():
    vm=read(ROOT/'datasets/ureter_semis_kidney_joint_20261028/video_validation_manifest.json');rows=vm['val']
    records={vid:{r['frame_index']:r for r in [json.loads(x) for x in (OUT/vid/'frames.jsonl').read_text(encoding='utf-8').splitlines()]} for vid in ('video_1','video_2')}
    result={'scope':'Post-hoc error localization only; longitudinal thirds are not anatomical tip/shaft truth and do not tune parameters', 'frames':[], 'totals':{}}
    for row in rows:
        vid=row['video_id'];index=row['frame_index'];record=records[vid][index];target=load(row['mask'])
        masks={n:load(OUT/vid/n/f'{index:06d}.png') for n in ('raw','geometry','geometry_causal')}
        item={'sample_id':row['sample_id'],'video_id':vid,'frame_index':index,'fit_accepted':record['fit']['accepted'],'comparisons':{}}
        for name,base in [('geometry','raw'),('geometry_causal','raw'),('causal_vs_geometry','geometry')]:
            other='geometry_causal' if name=='causal_vs_geometry' else name
            item['comparisons'][name]={'all':counts(target,masks[base],masks[other],np.ones(target.shape,bool))}
            state=record['fit']['state']
            if state is not None:
                x,y,angle,length,*_=state;yy,xx=np.indices(target.shape);along=((xx-x)*np.cos(angle)+(yy-y)*np.sin(angle))/length
                zones={'proximal_third':along<1/3,'middle_third':(along>=1/3)&(along<2/3),'distal_third':along>=2/3}
                for zone,where in zones.items():item['comparisons'][name][zone]=counts(target,masks[base],masks[other],where)
        result['frames'].append(item)
    for name in ('geometry','geometry_causal','causal_vs_geometry'):
        zones=set().union(*(r['comparisons'][name] for r in result['frames']))
        result['totals'][name]={z:{k:sum(r['comparisons'][name].get(z,{}).get(k,0) for r in result['frames']) for k in ('lost_fiber_tp','gained_fiber_tp','added_fiber_fp','removed_fiber_fp')} for z in sorted(zones)}
    (OUT/'error_breakdown.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    # Predefined representative rules: two largest geometric TP losses, largest FP removal,
    # largest causal additional TP loss, plus the known stone-miss frame.
    keys=[]
    for rule,compare,metric,number in [('geometric_TP_loss','geometry','lost_fiber_tp',2),('geometric_FP_removal','geometry','removed_fiber_fp',1),('causal_TP_loss','causal_vs_geometry','lost_fiber_tp',1)]:
        for item in sorted(result['frames'],key=lambda r:r['comparisons'][compare]['all'][metric],reverse=True)[:number]:
            key=(item['video_id'],item['frame_index'])
            if key not in [k[:2] for k in keys]:keys.append((*key,rule))
    key=('video_2',1822)
    if key not in [k[:2] for k in keys]:keys.append((*key,'previous_small_stone_miss'))
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
    canvas=Image.new('RGB',(1200,len(keys)*328+28),'#20252b');draw=ImageDraw.Draw(canvas)
    draw.text((5,5),'Reviewed29 overlays: red=stone, cyan=fiber. Columns: reference / raw / geometry / causal.',font=font,fill='white')
    index_rows=[]
    for j,(vid,index,rule) in enumerate(keys):
        row=next(r for r in rows if r['video_id']==vid and r['frame_index']==index);rgb=np.asarray(Image.open(row['image']).convert('RGB'))
        for col,name in enumerate(('reference','raw','geometry','geometry_causal')):
            mask=load(row['mask']) if name=='reference' else load(OUT/vid/name/f'{index:06d}.png')
            overlay=rgb.copy()
            for k,color in [(1,[235,40,40]),(2,[20,210,235])]:
                selected=mask==k;overlay[selected]=(.48*rgb[selected]+.52*np.asarray(color)).astype(np.uint8)
            canvas.paste(Image.fromarray(overlay).resize((300,300)),(col*300,28+j*328))
        draw.text((5,328+j*328),f'{vid} display frame{index+1} | {rule}',font=font,fill='white')
        index_rows.append({'video_id':vid,'frame_index':index,'one_based_frame':index+1,'selection_rule':rule})
    canvas.save(OUT/'reviewed_examples.jpg',quality=94)
    (OUT/'reviewed_examples.json').write_text(json.dumps(index_rows,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'totals':result['totals'],'examples':index_rows},indent=2))


if __name__=='__main__':main()
