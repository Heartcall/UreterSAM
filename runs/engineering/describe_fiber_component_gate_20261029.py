"""Transparent representative selection from frozen primary comparisons."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/a1_fiber_component_gate_20261029'
ARMS=('classwise_raw_3net','classwise_component_gate_3net','old190_raw_1net','old190_component_gate_1net')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def load(path):
    with Image.open(path) as im:return np.asarray(im).copy()


def main():
    s=read(OUT/'accuracy_summary.json');base=ROOT/'datasets/ureter_semis_kidney_joint_20261028'
    rows={'video':read(base/'video_validation_manifest.json')['val'],'image':read(base/'manifest.json')['val']}
    choices=[]
    for scope,arm,metric,label in [('video',ARMS[3],'lost_tp','largest_single_net_true_fiber_loss'),('video',ARMS[3],'removed_fp','largest_single_net_FP_removal'),('video',ARMS[1],'lost_tp','largest_three_net_true_fiber_loss'),('image',ARMS[1],'removed_fp','largest_image_FP_removal_pending_label')]:
        selected=max(s[scope]['pairs'][arm]['changes_per_frame'],key=lambda r:r['class_changes']['2'][metric])
        i=next(i for i,r in enumerate(rows[scope]) if r['sample_id']==selected['sample_id'])
        choices.append({'scope':scope,'index':i,'sample_id':selected['sample_id'],'selection_rule':label,'fiber_changes':selected['class_changes']['2']})
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',13);canvas=Image.new('RGB',(1300,len(choices)*290+28),'#20252b');draw=ImageDraw.Draw(canvas)
    draw.text((5,5),'Columns: reference / raw3net / gated3net / raw1net / gated1net. Red=stone, cyan=fiber.',font=font,fill='white')
    for j,item in enumerate(choices):
        row=rows[item['scope']][item['index']];rgb=np.asarray(Image.open(row['image']).convert('RGB'))
        for col,name in enumerate(('reference',*ARMS)):
            mask=load(row['mask']) if name=='reference' else load(OUT/f'{item["scope"]}_native'/name/f'{item["index"]:04d}.png')
            image=rgb.copy()
            for k,color in [(1,[235,40,40]),(2,[20,210,235])]:
                selected=mask==k;image[selected]=(.48*rgb[selected]+.52*np.asarray(color)).astype(np.uint8)
            canvas.paste(Image.fromarray(image).resize((260,260)),(col*260,28+j*290))
        text=f"{row.get('video_id') or 'image-dev'} frame{row.get('frame_index',item['index'])+1} | {item['selection_rule']}"
        draw.text((5,290+j*290),text,font=font,fill='white')
    canvas.save(OUT/'representative_examples.jpg',quality=94)
    (OUT/'representative_examples.json').write_text(json.dumps(choices,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(choices,indent=2))


if __name__=='__main__':main()
