"""Read-only semantic-reference coverage and failure addendum, no thresholds changed."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from tools.fiber_reference_workspace import read,sha256
from tools.run_fiber_blue_aplus import panel,frame_read
from ureter_semis.experiment import atomic_json
OUT=ROOT/'runs/a1_fiber_temporal_observation_search_20261030'


def main():
    cfg=read(OUT/'preflight.json')['config'];stats={};chosen=[]
    for scope,key in [('video_native','video_manifest'),('image_native','image_manifest')]:
        d=read(OUT/f'{scope}.json');rows=read(ROOT/cfg['data'][key])['val'];records=[]
        for row,r,m in zip(rows,d['records'],d['summary']['frames']):
            gt=sum(m['confusion_matrix'][2]);obs=r['observations']['light_current'];valid=obs['tracked']['groups']['shaft']['value'] is not None
            records.append({'sample_id':row['sample_id'],'gt_fiber_pixels':gt,'shaft_state_available':valid,
                            'front_state_available':obs['tracked']['groups']['endpoint']['value'] is not None,
                            'reasons':[c['reason'] for c in obs['current']['candidates']],'current_axis_candidates':len(obs['current']['candidates'])})
        stats[scope]={'all_frames':len(records),'gt_fiber_positive_frames':sum(r['gt_fiber_pixels']>0 for r in records),
                      'available_shaft_frames':sum(r['shaft_state_available'] for r in records),
                      'available_shaft_in_gt_positive_frames':sum(r['shaft_state_available'] and r['gt_fiber_pixels']>0 for r in records),
                      'available_shaft_in_gt_absent_frames':sum(r['shaft_state_available'] and r['gt_fiber_pixels']==0 for r in records),
                      'interpretation':'Pose availability conditioned on semantic reference presence, NOT geometric accuracy or tip recall; GT-absent can be FP or missing annotation', 'records':records}
        if scope=='image_native':
            zero=next(i for i,r in enumerate(records) if r['gt_fiber_pixels']==445 and r['current_axis_candidates']==0)
            absent=next(i for i,r in enumerate(records) if r['gt_fiber_pixels']==0 and r['shaft_state_available'])
            rejected=next(i for i,r in enumerate(records) if r['gt_fiber_pixels']>0 and not r['shaft_state_available'] and 'width_profile' in r['reasons'])
            chosen=[(zero,'GT fiber445 pixels; no neural seed'),(absent,'Pose accepted; semantic reference has no fiber'),(rejected,'GT fiber present; candidate rejected by width family')]
            board=Image.new('RGB',(1600,3*510),(10,10,10));draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
            for j,(i,reason) in enumerate(chosen):
                row=rows[i];r=d['records'][i];frame=frame_read(row['image']);target=np.array(Image.open(row['mask']));pred=np.array(Image.open(OUT/f'image_native/shared_raw2/{i:04d}.png'))
                base=frame.copy();obs=r['observations']['light_current'];pic=panel(frame,'Current observation',pred,i)
                c=next((c for c in obs['current']['candidates'] if c['label']==obs['current']['winner_label']),None)
                if c and c['shaft_observed']:
                    for a,b in c['shaft_lines']:
                        cv2.line(pic,tuple(np.rint(np.array(a)+[0,34]).astype(int)),tuple(np.rint(np.array(b)+[0,34]).astype(int)),(255,255,0),1,cv2.LINE_AA)
                    if c['front_observed']:cv2.circle(pic,tuple(np.rint(np.array(c['visible_front_candidate'])+[0,34]).astype(int)),4,(255,255,0),1)
                pics=[panel(base,'Input',None,i),panel(frame,'Semantic reference (not tip GT)',target,i),panel(frame,'Frozen raw model1',pred,i),pic]
                for k,p in enumerate(pics):
                    p=cv2.resize(p,(400,434),interpolation=cv2.INTER_AREA);board.paste(Image.fromarray(cv2.cvtColor(p,cv2.COLOR_BGR2RGB)),(400*k,510*j))
                draw.text((4,510*j+441),reason,font=font,fill='white')
                draw.text((4,510*j+464),f"sample={row['sample_id']} | reasons={records[i]['reasons']}",font=font,fill='white')
                draw.text((4,510*j+487),'Native metrics use original pixels; board scaled for display. Annotation completeness may be uncertain.',font=font,fill='white')
            board.save(OUT/'image_development_failure_board.jpg',quality=94)
    atomic_json(OUT/'static_geometry_diagnostics.json',stats)
    text='''# T1静态覆盖与可视化失败补充

此补充在固定四臂全部结束后只读生成，没有调整阈值、重训或额外网络前向。

70张视频静态参考全部含光纤，68张产生可用杆体与端部状态。132张图像开发参考中50张标有光纤，仅11张有可用几何状态；另82张未标光纤的参考中有1张被接受，总计12/132。几何可用性不是位置正确率，也不能把这1张自动判为真实误检：原图像标注完整性仍可能有限，需区分漏标。三个静态几何臂相同，独立图像无历史增益。

这说明本轮耗时通过不能推广成通用几何定位达标。入口/FOV方向、小目标尺寸、双边缘支持、宽度族与误检均可能限制静态图像；接受样本也没有独立尖端真值，不能保证前沿正确。

视频可视化保留了负例：video_2显示1266帧原RGB仍可见光纤，当前候选只3/8对双边缘支持，按冻结5对门槛拒绝，历史已过期，状态不可用。显示311帧当前种子候选只有4/8对支持而拒绝，历史候选用当前RGB获得5/8对、平均光纤概率约0.99994并被弱更新；这是搜索路径改变的证据，不是已验证的尖端恢复。

image_development_failure_board.jpg列出全部445像素无种子样本、唯一参考未标光纤但接受状态的样本，以及按清单顺序第一个参考含光纤却因宽度族拒绝的样本；不是成功帧筛选。全部132样本分母、原因和状态在static_geometry_diagnostics.json。

本轮结论限定为轻量计算和当前证据优先机制可行。几何准确性和适用范围仍未通过，history_search、model1均不默认采用。下一步先只读诊断这些失败及观测与语义参考的关系，再形成独立的像素修正候选协议；不自动实施T2、训练或石头记忆。
'''
    (OUT/'static_geometry_limitations.md').write_text(text,encoding='utf-8')
    report=OUT/'report.md';initial=OUT/'report_v1_before_visual_review.md'
    if initial.exists():raise FileExistsError(initial)
    initial.write_bytes(report.read_bytes());old_art=OUT/'report_artifacts.json';(OUT/'report_artifacts_v1_before_visual_review.json').write_bytes(old_art.read_bytes())
    report.write_text(report.read_text(encoding='utf-8')+'\n## 静态几何与负例补充\n\n132张图像中50张参考含光纤，仅11张有可用几何状态；另1张参考未标光纤却被接受。video_2显示1266帧光纤可见，但仅3/8对边缘支持而拒绝。**耗时通过不能视为通用几何定位达标。** 详见[补充分析](static_geometry_limitations.md)、[逐样本统计](static_geometry_diagnostics.json)、[图像失败板](image_development_failure_board.jpg)。\n',encoding='utf-8')
    art=read(old_art);art['supplement_generator_sha256']=sha256(Path(__file__))
    for p in [report,OUT/'static_geometry_limitations.md',OUT/'static_geometry_diagnostics.json',OUT/'image_development_failure_board.jpg']:
        art['artifacts'][p.name]={'path':str(p),'sha256':sha256(p)}
    atomic_json(old_art,art);print(json.dumps({'static_video':68,'static_image':12,'image_gt_positive':50,'available_in_gt_positive':11,'new_inference':0}))


if __name__=='__main__':main()
