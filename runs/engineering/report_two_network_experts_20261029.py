"""Post-run report and diagnostic boards; no inference or parameter selection."""
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from tools.fiber_reference_workspace import read,sha256
from ureter_semis.experiment import atomic_json
OUT=ROOT/'runs/a1_two_network_experts_20261029'
NAMES=['classwise3','member1_2net','member2_2net']

def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])+'\n'
def f(v):return f'{v:.6f}' if isinstance(v,float) else str(v)
def main():
    s=read(OUT/'summary.json'); cfg=read(OUT/'preflight.json')['config'];audit=read(OUT/'engineering_checks.json')
    data={scope:{name:read(OUT/f'{scope}/{name}.json') for name in NAMES} for scope in ['video_native','image_native']}
    rows=read(ROOT/cfg['video_manifest'])['val']; font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    selected={}
    for vid in ['video_1','video_2']:
        ids=[i for i,r in enumerate(rows) if r['video_id']==vid]; rec=data['video_native']['member1_2net']['records']
        picks=[]
        for reason,key in [('largest_lost_tp',lambda i:rec[i]['paired_fiber']['lost_tp']),('largest_added_fp',lambda i:rec[i]['paired_fiber']['added_fp']),('largest_net_tp_gain',lambda i:rec[i]['paired_fiber']['gained_tp']-rec[i]['paired_fiber']['lost_tp'])]:
            i=max(ids,key=key)
            if i not in [x[0] for x in picks]:picks.append((i,reason))
        i=max(ids,key=lambda i:rows[i]['frame_index'])
        if i not in [x[0] for x in picks]:picks.append((i,'latest_approved_frame'))
        selected[vid]=[{'sample_id':rows[i]['sample_id'],'display_frame_one_based':rows[i]['frame_index']+1,'selection':reason} for i,reason in picks]
        canvas=Image.new('RGB',(1200,len(picks)*286),(20,20,20)); draw=ImageDraw.Draw(canvas)
        for j,(i,reason) in enumerate(picks):
            rgb=np.array(Image.open(rows[i]['image']).convert('RGB'));target=np.array(Image.open(rows[i]['mask']))
            masks=[None,target]+[np.array(Image.open(OUT/f'video_native/{n}/{i:04d}.png')) for n in NAMES]
            labels=['Input','Reference']+NAMES
            for col,(mask,label) in enumerate(zip(masks,labels)):
                a=rgb.copy()
                if mask is not None:
                    for c,color in [(1,[40,255,80]),(2,[30,160,255]),(255,[150,150,150])]:
                        area=mask==c;a[area]=(0.52*a[area]+0.48*np.array(color)).astype(np.uint8)
                canvas.paste(Image.fromarray(a).resize((240,240)),(col*240,j*286+24))
                draw.text((col*240+4,j*286+3),label,fill='white',font=font)
            draw.text((5,j*286+265),f'{vid} frame {rows[i]["frame_index"]+1}: {reason}',fill='white',font=font)
        canvas.save(OUT/f'{vid}_diagnostic_board.jpg',quality=93)
    atomic_json(OUT/'diagnostic_board_selection.json',{'role':'diagnostic extremes and latest reference; all70 metrics reported separately; never used for selection','frames':selected})
    fields=['scope','scheme','sample_id','video_id','frame_index','class','dice','precision','recall','tp','fp','fn','boundary_f','lost_tp','gained_tp','added_fp','removed_fp']
    with (OUT/'per_frame_metrics.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for scope,arms in data.items():
            for name,d in arms.items():
                for frame,record in zip(d['summary']['frames'],d['records']):
                    matrix=np.array(frame['confusion_matrix'])
                    for c,label in [(1,'stone'),(2,'fiber')]:
                        tp=int(matrix[c,c]);fp=int(matrix[:,c].sum()-tp);fn=int(matrix[c,:].sum()-tp)
                        def ratio(a,b):return a/b if b else ''
                        b=frame['boundaries'][str(c)];bp=ratio(b['matched_predicted'],b['predicted_boundary']);br=ratio(b['matched_target'],b['target_boundary'])
                        bf=2*bp*br/(bp+br) if isinstance(bp,float) and isinstance(br,float) and bp+br else ''
                        writer.writerow(dict(scope=scope,scheme=name,sample_id=record['sample_id'],video_id=record['video_id'],frame_index=record['frame_index'],**{'class':label},dice=ratio(2*tp,2*tp+fp+fn),precision=ratio(tp,tp+fp),recall=ratio(tp,tp+fn),tp=tp,fp=fp,fn=fn,boundary_f=bf,**record['paired_'+label]))
    report=['# 固定A0单光纤成员两网质量与成本对照（2026-10-29）\n',
    '已完成两个预定义成员的有限对照。model1两网组合同时提高70帧和132图像的光纤Dice与边界F，降低计算成本，但有病例退化、视频Precision下降和图像Recall微降；保留为低成本候选，不直接采用。model2总体光纤质量较差。本轮结束成员搜索，不扫阈值、不加几何或颜色、不自动训练或进入S3。\n',
    '## 冻结范围与复现\n',
    '三网classwise为旧190结石网+A0两个光纤成员（73,804,185参数）；两网为同一旧190结石网+A0 model1或model2（49,202,790参数）。两网实际仅构建两张GPU网络，没有隐藏运行未选成员。所有方案PIL RGB256/ImageNet输入、FP32、batch1、原图双线性概率映射align_corners=False、各专家argmax和光纤覆盖顺序一致。没有阈值、组件、颜色、模板或时序修改。成员变化仍会通过光纤优先影响少量结石像素，因此结石并非严格不变。\n',
    '202张静态×3组均重新推理，原图连续概率与历史各成员缓存逐值相等（最大误差0）；缓存仅作复现核验。三网所有静态汇总与A+原始结果完全一致；全片2694帧三网掩膜与D0完全一致。输入/标签/权重/源码保持身份，源码与冻结配置另存source_snapshot.zip。测试集仍70帧（原29+新增41，病例43/27），本轮未发现新确认标注。132图像按68/64来源、25/107旧190曝光另报。\n',
    'TRAIN和144保留test像素读取0，训练更新0，stone_new排除不变。两个视频及派生物只作开发验证。42/120随机样本尚未完成，70便利/开发参考不能外推全片精度；无独立尖端真值，尖端Recall/定位及物理落点均不可计算。\n',
    '## 整体质量\n']
    qrows=[]
    for scope,label in [('video_native','视频70'),('image_native','图像132')]:
        for name,d in data[scope].items():
            m=d['summary']['metrics'];b=d['summary']['boundaries']
            for c,cls in [('1','stone'),('2','fiber')]:qrows.append([label,name,cls,*[f(m[cls+'_'+k]) for k in ['dice','precision','recall']],f(b[c]['f_score']),m[cls+'_tp'],m[cls+'_fp'],m[cls+'_fn']])
    report.append(table(['参考','方案','类别','Dice','Precision','Recall','边界F(3px)','TP','FP','FN'],qrows))
    report.append('边界与混淆计数排除未知像素；所有70帧/132图像逐项指标见per_frame_metrics.csv及各组JSON。结石可见面积分组、小目标零TP帧也保留在JSON，不用平均值掩盖漏检。\n')
    report.append('## 配对像素变化\n')
    report.append(table(['参考','方案','类别','原TP删除','TP恢复','FP新增','FP移除'],[[scope,name,c,*[d['paired_changes'][c][k] for k in ['lost_tp','gained_tp','added_fp','removed_fp']]] for scope,arms in data.items() for name,d in arms.items() if name!='classwise3' for c in ['stone','fiber']]))
    report.append('model1视频光纤净增6897 TP、2410 FP；图像净少93 TP、少1386 FP。不能把净变化当作“没有删正确像素”：分别删除2205和1025个原TP。图像结石TP变化还受光纤覆盖顺序影响。\n')
    report.append('## 来源、病例和历史参考分组\n')
    report.append(table(['参考','分组','数量','方案','结石Dice','光纤Dice','光纤Precision','光纤Recall'],[[scope,g,len(v['frames']),name,*[f(v['metrics'][k]) for k in ['stone_dice','fiber_dice','fiber_precision','fiber_recall']]] for scope,arms in data.items() for name,d in arms.items() for g,v in d['groups'].items()]))
    report.append('model1的主要收益在video_2（0.890088→0.906306），video_1从0.978289降到0.976570。旧29帧光纤Dice为0.966802/0.969235/0.959110（三网/model1/model2）。两候选均通过原29三项初筛；原门槛未改变。另行预冻结的70/132逐类Dice/P/R/边界均不退化检查，两候选均未通过。该检查比旧初筛更严格，不能混称“未通过原29门槛”，也不表示model1没有研究价值。无默认采用。\n')
    for name,q in s['static_quality'].items():report.append('- '+name+'不退化检查退化项：'+', '.join(q['rejected_checks'])+'。\n')
    center=read(OUT/'centerline_coverage_proxy.json')
    report.append('\n## 中心线补充代理\n')
    report.append(table(['参考','方案','参考中心线像素','覆盖','覆盖率'],[[scope,name,v['reference_pixels'],v['covered'],f(v['coverage'])] for scope,d in center['sets'].items() for name,v in d['models'].items()]))
    report.append('该指标从全部可见光纤参考组件细化得到，排除未知255的3px邻域和边缘，不连隐藏区域。它是事后补充覆盖代理，既不是解剖尖端真值，也不改变门槛。\n')
    report.append('## 完整视频性能\n')
    report.append('RTX4070Ti，torch2.11+cu128，12帧预热；每个方案/视频使用独立新进程，共6进程、8082帧。各组均传回融合掩膜和两个独立专家的原图三类概率，保持相同接口。该接口及运行环境时序与历史A+不同，仅作本轮匹配比较，不与历史延迟相减来归因加速。\n')
    report.append(table(['方案','视频','实际网络','帧数','管线P50/P95 ms','总周期P50/P95 ms','含导出fps','峰值CUDA MiB','改变帧数'],[[name,vid,v['networks'],v['frames'],f"{v['timings']['pipeline_ms']['p50']:.2f}/{v['timings']['pipeline_ms']['p95']:.2f}",f"{v['timings']['cycle_ms']['p50']:.2f}/{v['timings']['cycle_ms']['p95']:.2f}",f"{v['offline_fps_including_all_exports']:.2f}",f"{v['cuda_peak_allocated_mib']:.2f}",v['changed_frames_vs_classwise3']] for name,items in s['full_video'].items() for vid,v in items.items()]))
    report.append('管线包含预处理、全部所需网络、原图映射/融合/CPU传输。总周期另含解码、渲染编码、PNG/日志/审计，离线fps还含编码释放。三组p95均在约33ms源帧预算内，但单帧最大值仍可超预算；未测试实际采集、显示队列、丢帧和持续运行，不宣称实时系统验收。两网CUDA分配峰值约223.92MiB，三网319.82MiB；此为allocated而非设备总占用。模型没有拟合/颜色分支，接受/回退率不适用；每帧直接推理，无隐藏轮廓填补。两个成员均改变2694/2694帧，此仅表示预测变化。\n')
    report.append('六份timings.json保留全部逐帧分解、最大值。六份frames.jsonl保留RGB/掩膜哈希、面积与有无参考。全片没有稠密真值；未知帧不得以预测变化、面积或视频观感冒充准确率。\n')
    for vid in ['video_1','video_2']:
        report.append(f'\n### {vid}已知片段及输出行为\n')
        logs={name:[json.loads(line) for line in (OUT/f'full_video/{name}/{vid}/frames.jsonl').read_text().splitlines()] for name in NAMES}
        if vid=='video_1':
            ranges=[(33,37),(54,72),(77,78),(80,113),(330,722)]
            for start,end in ranges:
                text=[]
                for name,records in logs.items():
                    empty=[r['frame_index']+1 for r in records[start-1:end] if r['stone_pixels']==0]
                    text.append(name+': stone_empty='+str(empty))
                report.append(f'- 显示帧{start}–{end}：'+'；'.join(text)+'。无参考帧的空掩膜不自动判为真实漏检。\n')
        report.append(f'\n[完整四栏视频]({vid}_comparison.mp4)，[诊断板]({vid}_diagnostic_board.jpg)。诊断板按model1最大TP删除、最大FP新增、最大净TP恢复及最晚已审帧选择；包含失败和改善，不作为选型子集。索引见diagnostic_board_selection.json。\n')
    report.append('\n## 结论、停止与下一步\n')
    report.append('固定两个成员实验到此结束。model2不作为优先底座；model1保留低成本候选，三网保留冻结对照，默认仍原A0。两者都通过旧初筛，但不能按本轮不退化规则直接推荐替换，不继续搜索更多成员或调阈值。S2尚未验收，不自动进入结石时序记忆。\n')
    report.append('下一步按已有计划开展光纤专用轻量精修可行性，先做只读的可修正区域/错误分布和时间预算分析，重点是model1新增FP、丢失TP及132图像弱项，固定三网作为对照；先判断局部ROI能否同时覆盖错误与保持正确像素。原联合ROI失败仍保留，真值只作上界评价，不能用来生成推理ROI。冻结ROI/融合方案后再决定是否实验；任何训练、专门尖端头或光度增强须另提数据/初始化/预算/验证/回滚计划并批准。12帧尖端标注和120分层样本持续可补，不用预测生成真值。\n')
    report.append(f"工程：3项新增测试+3项classwise回归通过；{audit['passed']}项独立后验核验通过，重新检查606静态及8082全片掩膜、六个进程、全部视频解码/哈希、源码归档、冻结输入/权重、概率和时分位数。中心线另含3项辅助自检。训练/默认/划分/标签/历史输出均未修改。\n")
    report.append('\n复现：`python -m unittest tests.test_member_experts tests.test_classwise_experts`；实验`python -m tools.run_two_network_experts`（已存在结果目录时拒绝覆盖）。检查`preflight.json`、`model_identities.json`、`static_quality.json`、`engineering_checks.json`和`summary.json`。\n')
    (OUT/'report.md').write_text('\n'.join(report),encoding='utf-8')
    atomic_json(OUT/'report_artifacts.json',{'generator_sha256':sha256(Path(__file__)),'files':{p.name:sha256(p) for p in [OUT/'report.md',OUT/'per_frame_metrics.csv',OUT/'video_1_diagnostic_board.jpg',OUT/'video_2_diagnostic_board.jpg']}})
    print(json.dumps({'report':str(OUT/'report.md'),'checks':audit['passed']}))

if __name__=='__main__':main()
