"""T1 reports and literal metadata visualizations; no new model inference."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from tools.fiber_reference_workspace import read,sha256
from tools.run_fiber_blue_aplus import panel
from ureter_semis.experiment import atomic_json
OUT=ROOT/'runs/a1_fiber_temporal_observation_search_20261030'


def logs(arm,vid,root=OUT):return [json.loads(s) for s in (root/f'full_video/{arm}/{vid}/frames.jsonl').read_text().splitlines()]
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])+'\n'
def f(value):return '不可计算' if value is None else f'{value:.6f}' if isinstance(value,float) else str(value)


def literal_panel(frame,name,mask,record,index):
    image=panel(frame,name,mask,index);r=record['tracked'];color=(0,230,255) if r['front_is_prediction'] else (255,255,0)
    shaft=r['groups']['shaft'];endpoint=r['groups']['endpoint'];footer=np.zeros((28,frame.shape[1],3),np.uint8)
    if shaft['value'] is not None:
        ex,ey,theta,width=shaft['value'];e=np.array([ex,ey]);u=np.array([np.cos(theta),np.sin(theta)]);n=np.array([-u[1],u[0]])
        if endpoint['value'] is not None:
            length,distal=endpoint['value']
            for sign in (-1,1):
                a=e+sign*n*width/2;b=e+u*length+sign*n*distal/2
                cv2.line(image,tuple(np.rint(a+[0,34]).astype(int)),tuple(np.rint(b+[0,34]).astype(int)),color,1,cv2.LINE_AA)
            cv2.circle(image,tuple(np.rint(e+u*length+[0,34]).astype(int)),4,color,1,cv2.LINE_AA)
        else:
            # Direction glyph only; no invented visible shaft extent or endpoint.
            color=(200,200,200)
            cv2.arrowedLine(image,tuple(np.rint(e+[0,34]).astype(int)),tuple(np.rint(e+20*u+[0,34]).astype(int)),color,1,cv2.LINE_AA,tipLength=.2)
            cv2.line(image,tuple(np.rint(e+n*width/2+[0,34]).astype(int)),tuple(np.rint(e-n*width/2+[0,34]).astype(int)),color,1,cv2.LINE_AA)
    cv2.putText(footer,'S:'+shaft['status']+' E:'+endpoint['status'],(2,11),cv2.FONT_HERSHEY_SIMPLEX,.29,(255,255,255),1,cv2.LINE_AA)
    candidates=record['current']['candidates'];winner=next((c for c in candidates if c['label']==record['current']['winner_label']),None)
    if winner: text=f"current {winner['level']} qN={winner['qN']:.2f} edges={winner['edge_pairs']}/8"
    elif record['current']['ambiguous']:text='ambiguous current hypotheses; no forced identity'
    else:text='no qualified current winner'
    if endpoint['value'] is None and shaft['value'] is not None:text='shaft direction glyph ONLY; front unknown'
    cv2.putText(footer,text,(2,24),cv2.FONT_HERSHEY_SIMPLEX,.30,(230,230,230),1,cv2.LINE_AA)
    return np.vstack([image,footer])


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=read(OUT/'summary.json');cfg=read(OUT/'preflight.json')['config'];audit=read(OUT/'engineering_checks.json')
    g=read(OUT/'geometry_diagnostics.json');cl=read(OUT/'visible_centerline_diagnostics.json');artifacts={};selections={};known={}
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
    for vid in ['video_1','video_2']:
        data={a:logs(a,vid) for a in cfg['arms']};n=len(data['raw2']);info=summary['full_video']['raw2'][vid]
        old=logs('gated_state_only',vid,ROOT/'runs/a1_fiber_temporal_state_gate_20261030')
        picks=[0,n//4,n//2,3*n//4] if vid=='video_1' else [438,439,440,621,642]
        history_used=[i for i,r in enumerate(data['history_search']) if r['tracked']['used_provenance']=='history_conditioned']
        if history_used:picks.append(history_used[0])
        missing=[i for i,r in enumerate(data['history_search']) if r['tracked']['groups']['shaft']['value'] is None]
        if missing:picks.append(missing[len(missing)//2])
        picks=list(dict.fromkeys(picks));selections[vid]=[i+1 for i in picks]
        board=Image.new('RGB',(2000,len(picks)*530),(10,10,10));draw=ImageDraw.Draw(board)
        for row,i in enumerate(picks):
            rgb=np.array(Image.open(ROOT/f'runs/fiber_reference_d1_20261029/images/{vid}/{i:06d}.png').convert('RGB'))
            assert __import__('hashlib').sha256(rgb.tobytes()).hexdigest()==data['raw2'][i]['source_rgb_raw_sha256']
            frame=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR);mask=np.array(Image.open(data['raw2'][i]['mask_reference']))
            input_pic=np.vstack([panel(frame,'Input',None,i),np.zeros((28,400,3),np.uint8)])
            t0=panel(frame,'T0 gated (historical)',mask,i);geo=old[i]['tracked']['geometry']
            if geo:
                color=(0,230,255) if old[i]['tracked']['prediction_only'] else (255,255,0)
                for segment in geo['side_segments']:
                    a,b=[tuple(np.rint(p+[0,34]).astype(int)) for p in np.array(segment)];cv2.line(t0,a,b,color,1,cv2.LINE_AA)
                cv2.circle(t0,tuple(np.rint(np.array(geo['tip_xy'])+[0,34]).astype(int)),4,color,1)
            tail=np.zeros((28,400,3),np.uint8);cv2.putText(tail,old[i]['tracked']['status'],(3,18),cv2.FONT_HERSHEY_SIMPLEX,.36,(255,255,255),1)
            pics=[input_pic,np.vstack([t0,tail])]+[literal_panel(frame,a,mask,data[a][i],i) for a in cfg['arms'][1:]]
            for col,pic in enumerate(pics):board.paste(Image.fromarray(cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)),(col*400,row*530))
            label='fixed T0 regression frame' if vid=='video_2' and i in [438,439,440,621,642] else 'first history acceptance / missing context' if vid=='video_2' else 'uniform timeline context'
            draw.text((5,row*530+469),f'{vid} frame{i+1}: {label}. Geometry is NOT anatomical-tip truth.',font=font,fill='white')
            draw.text((5,row*530+494),'Cyan=current-supported state; yellow=prediction; gray arrow=shaft direction ONLY (front unavailable).',font=font,fill='white')
            if vid=='video_2' and i==438:board.crop((0,row*530,2000,(row+1)*530)).save(OUT/'video_2_frame439_comparison.jpg',quality=95)
        board.save(OUT/f'{vid}_evidence_diagnostic_board.jpg',quality=94)
        # Literal full videos preserve initial exports and fix shaft-only display semantics.
        cap=cv2.VideoCapture(str(ROOT/f'data/videos/{vid}.mp4'));path=OUT/f'{vid}_evidence_comparison.mp4'
        writer=cv2.VideoWriter(str(path),cv2.CAP_MSMF,cv2.VideoWriter_fourcc(*'H264'),info['source_fps'],(2000,462));i=0
        if not writer.isOpened():raise RuntimeError('Encoder unavailable')
        try:
            while True:
                ok,frame=cap.read()
                if not ok:break
                mask=np.array(Image.open(data['raw2'][i]['mask_reference']))
                pics=[np.vstack([panel(frame,'Input',None,i),np.zeros((28,400,3),np.uint8)])]
                pics.extend(literal_panel(frame,a,mask,data[a][i],i) for a in cfg['arms'])
                writer.write(np.hstack(pics));i+=1
        finally:cap.release();writer.release()
        cap=cv2.VideoCapture(str(path));count=0
        while cap.read()[0]:count+=1
        cap.release();assert count==i==n
        artifacts[path.name]={'frames':count,'sha256':sha256(path),'path':str(path)}
        fig,axes=plt.subplots(6,1,figsize=(14,12),sharex=True)
        for arm,color in [('light_current','0.55'),('evidence_state','#2873ad'),('history_search','#e08421')]:
            rows=data[arm];ts=[r['timestamp_seconds'] for r in rows];values=np.full((n,6),np.nan)
            for i,r in enumerate(rows):
                z=r['tracked'];shaft=z['groups']['shaft']['value'];end=z['groups']['endpoint']['value'];front=z['visible_front_candidate']
                if front is not None:values[i,:2]=front
                if shaft is not None:values[i,3]=np.rad2deg(shaft[2]);values[i,4]=shaft[3]
                if end is not None:values[i,2]=end[0];values[i,5]=end[1]
            for k,ax in enumerate(axes):ax.plot(ts,values[:,k],lw=.8,label=arm,color=color)
        for ax,label in zip(axes,['Visible-front candidate X (px)','Visible-front candidate Y (px)','Length (px, observed/predicted)','Axis angle (deg, wrapped)','Entry line width (px)','Front line width (px)']):ax.set_ylabel(label);ax.grid(alpha=.2)
        axes[0].legend(ncol=3);axes[0].set_title(vid+': T1 partial-state trajectories, NOT anatomical-tip accuracy');axes[-1].set_xlabel('Video source time (s)')
        fig.tight_layout();fig.savefig(OUT/f'{vid}_parameter_traces.png',dpi=150);plt.close(fig)
        known[vid]=[]
        for i in picks:
            r=data['history_search'][i];cur=r['current'];known[vid].append({'display_frame':i+1,'current_candidates':cur['candidates'],
                'ambiguous':cur['ambiguous'],'strong_label':cur['strong_label'],'T0_tracked':old[i]['tracked'],'T1_current':data['light_current'][i]['tracked'],
                'T1_evidence':data['evidence_state'][i]['tracked'],'T1_history':r['tracked'],'tip_truth':None})
    atomic_json(OUT/'known_frame_diagnostics.json',{'selection':selections,'frames':known,'policy':'fixed T0 regression frames, first history-use event, missing/uniform context; all frames retained in logs'})
    report=['# T1轻量当前观测与历史搜索完整对照\n',
            '2026-10-30。**四臂、202张静态复现及8独立进程10776帧完整回放已完成。轻量观测和两种状态方案均通过本轮耗时门槛；历史搜索增加了有状态覆盖，但没有独立几何真值，不能据此认定更准。所有分割像素不变，没有分割准确率提升、训练或默认替换。**\n',
            '## 设计、数据与身份\n',
            '原model1两网49,202,790参数，RGB/PIL256、ImageNet归一化、FP32/batch1、原图双线性概率映射和光纤优先保持。raw2不观测；light_current每帧最多2轴候选；evidence_state加入强当前优先的部分状态；history_search只在没有独立强当前胜者时增加最多1个历史候选。当前提案160网格、概率0.2、组件最少4点；8截面×每侧17位置，3×25端部采样，两次鲁棒直线拟合，无旧全图Sobel/密集端帽、裁剪重网、光流、新颜色模板或额外网络。\n',
            '候选仅元数据；没有最大组件删除、模板相交、隐藏像素填补或跨视频身份继承。RGB差只验证边界，不是蓝色身份硬门控；浅白端部可提供证据。可见前沿可能只是遮挡边界，不能当真实圆钝尖端。无端部时只输出杆体，端部组最多预测3帧；无向杆体不初始化入口组，用历史消歧只能算弱观测。\n',
            '强当前证据先于历史门控，明显越界本帧重获、清零相关速度，没有T0固定三帧等待。兼容强/弱观测沿用alpha0.65/beta0.10；弱观测只更新已有兼容组。历史搜索用同一当前RGB/网络检验，永不升级独立强证据；没有支持则拒绝/过期，全图当前提案始终运行。六参数联合图像状态没有端帽半径、真实设备标定或校准置信区间。\n',
            '70视频参考、132图像及原29/新增41、43/27病例、68/64来源、25/107曝光分组冻结；70是重复开发验证。42/120随机抽样未完成，不能外推全片分割精度；12帧尖端专项未补，无尖端Recall、物理尖端误差或落点准确率。视频与派生物均不训练，stone_new排除，521训练及144保留test像素读取0。冻结配置中的“unlabelled/zero-seed static445”措辞在preflight澄清：该样本有语义参考，只是网络无种子，不是没有真值。未改配置或任何阈值。\n',
            '## 完整视频性能（同轮匹配）\n']
    rows=[]
    for arm,vs in summary['full_video'].items():
        for vid,v in vs.items():
            t=v['timings'];rows.append([arm,vid,f"{t['pipeline_total_ms']['p50']:.2f}/{t['pipeline_total_ms']['p95']:.2f}/{t['pipeline_total_ms']['max']:.2f}",
                f"{t['geometry_total_ms']['p50']:.2f}/{t['geometry_total_ms']['p95']:.2f}",f"{t['cycle_ms']['p50']:.2f}/{t['cycle_ms']['p95']:.2f}",
                f"{v['offline_fps_including_render_audit_encode']:.2f}",f"{v['cpu_process_seconds_including_render']:.2f}",f"{v['cuda_peak_allocated_mib']:.2f}",v['pipeline_p95_within_source_interval'],v['geometry_p95_within_design_budget']])
    report.append(table(['臂','视频','总管线p50/p95/max ms','几何p50/p95 ms','含导出周期p50/p95 ms','含导出fps','进程CPU秒','CUDA MiB','源帧预算','几何≤5ms'],rows))
    report.append('RTX4070Ti，两片各臂独立进程；各12帧临时状态预热后清空，共96次预热前向不计入10776正式帧。另202次静态前向，共11074次网络调用，不能将四臂共享静态掩膜冒充808次前向。无额外神经网络/裁剪，几何主要为CPU。进程CPU秒含解码/编码与审计，不是几何纯CPU占用；CUDA峰值是allocated，不是设备总内存。\n')
    report.append('总管线直接计时包含两网、原图掩膜及两份三类概率传CPU、全部观测/状态/公开元数据转换；几何内层计时含grid/FOV、采样/拟合、质量和状态，不含返回时的公开元数据转换，差额仍在总管线中。逐帧全部分阶段分布在timings.json；不相加或相减分位数，不用历史T0推算加速。周期/fps还含解码、编码、哈希、选中帧概率核验；最终大JSON写盘、模型加载在计时外，后续重新渲染的说明视频也不计入原性能。通过p95门槛不等于已验证实时相机采集/显示队列。\n')
    report.append('源时间均预扫描验证单调PTS，视频各自独立状态，静态逐张清空；完整70参考状态从真实视频帧索引取，不把稀疏参考当相邻帧。所有四臂即使失败仍测两片，没有跳过。\n')
    report.append('## 观测、状态与历史的实际增量\n')
    rows=[]
    for arm,vs in summary['full_video'].items():
        for vid,v in vs.items():
            counts={g:sum(n for status,n in d.items() if status not in ['unavailable','expired','invalid_prediction_reset','invalid_update_reset']) for g,d in v['statuses'].items()}
            rows.append([arm,vid,v['qualified_shaft_frames'],v['qualified_front_frames'],v['current_strong_frames'],v['ambiguous_frames'],counts['shaft'],counts['endpoint'],v['history_attempted_frames'],v['history_used_frames'],v['strong_override_frames']])
    report.append(table(['臂','视频','合格杆体帧','合格前沿帧','强当前胜者帧','歧义帧','有杆体状态','有端部状态','历史搜索帧','历史接受帧','强当前覆盖帧'],rows))
    report.append('杆体/前沿“合格”统计所有当前候选，包含歧义候选；状态统计实际选中/预测后的可用性，分母各722/1972。video_2历史搜索105帧、接受72帧，杆体状态由1764增至1796、端部由1748增至1766；video_1始终有强当前证据，没有触发历史搜索，不能在此片宣称历史增益。video_2的72次历史接受中只有49次同时更新端部组；历史杆体可见不代表头部已找到。更多覆盖也可能维持错误，不能称72帧恢复正确或尖端召回提高。\n')
    for vid in ['video_1','video_2']:
        for arm in cfg['arms'][1:]:
            v=summary['full_video'][arm][vid]
            report.append(f"{vid}/{arm} 状态：`{json.dumps(v['statuses'],ensure_ascii=False)}`；当前候选原因：`{json.dumps(v['candidate_reasons'],ensure_ascii=False)}`。\n")
    report.append('全部接受/拒绝、歧义、重获、过期事件及全部候选均保留。geometry_diagnostics.json同时记录所有可用相邻对与三臂共同可用对的前沿速度/角速度/长度速度及分母，不将低抖动或拟合残差称真实准确率。没有光流或真实相机运动分离，过曝、组织双边缘、高置信误检、错误初始化、缺种子和长遮挡仍可能失败。\n')
    report.append('## 已知失败帧回归\n')
    report.append('video_2显示439帧：当前观测qN约1、双侧8/8支持，T1两状态臂本帧均strong_current_override，直接采用当前新位置；440、441走兼容更新，取消了T0在439–440仍保留旧位置的固定等待。622、643帧也逐项保留当前证据、原T0与T1状态；这验证了更新行为和定性跟随现象，不是有独立尖端GT的定位精度结论。\n')
    report.append('固定回归帧439/440/441/622/643、首个历史接受、无状态上下文和video_1均匀上下文均导出，选择规则在known_frame_diagnostics.json。全部帧都在日志与完整视频，没有只报告成功片段。需要结合RGB审核前沿是否落在真正可见端部；不能仅凭qN高就判定正确。\n')
    report.append('## 分割精度：所有四臂完全不变\n');rows=[]
    for scope in ['video_native','image_native']:
        d=read(OUT/(scope+'.json'));m=d['summary']['metrics']
        for cls,c in [('stone','1'),('fiber','2')]:rows.append([scope,cls,*[f(m[cls+'_'+k]) for k in ['dice','precision','recall']],f(d['summary']['boundaries'][c]['f_score']),m[cls+'_tp'],m[cls+'_fp'],m[cls+'_fn']])
    report.append(table(['参考','类别','Dice','Precision','Recall','边界F','TP','FP','FN'],rows))
    report.append('所有10776帧原图掩膜与原model1逐帧一致，202张静态概率和掩膜精确复现。全部四臂TP删除/恢复、FP移除/新增均0，不能把不变性称分割改善。逐帧70/132的混淆矩阵、边界与各原始分组完整保存，classwise冻结参照另列；model1先前不退化检查未通过的结论不变，不自动采用底座。原85/523个结石保护冲突仍未解决。\n')
    report.append(table(['参考','可见语义中心线Recall','中心线Precision','clDice'],[[scope,f(v['gt_centerline_recall']),f(v['pred_centerline_precision']),f(v['clDice'])] for scope,v in cl.items()]))
    report.append('中心线为对现有可见语义掩膜进行Zhang-Suen细化后的精确像素覆盖，逐样本分母已保存；此项为只读描述性补充，不是新门槛、真实解剖中心线或尖端真值。所有臂因掩膜相同而相同。未知255不计入前景，未用预测生成尖端参考。\n')
    report.append('## 核验、可视化与停止\n')
    report.append(f"19项新增机制测试+11项T0回归，共30项通过；{audit['passed']}项独立核验通过：源码/配置/权重/输入身份、202静态候选及指标重算、8独立进程完整10776帧、逐帧原掩膜和RGB重解码、相同当前观测、两状态全片及前缀重放、选中帧历史图像证据复算、阶段分位数与完整视频。审计没有额外网络前向，没有修改阈值或原结果。\n")
    report.append('原运行导出state_comparison/state_replay保留。其“仅杆体、无端部”面板用了固定20%画幅的示意线，不是已观测长度；最终evidence_comparison改成明确的短灰色方向箭头，不画未知端部，避免误读。该修正仅重新渲染已有元数据，不改变状态、掩膜、指标或性能记录。青色为当前支持状态、黄色为含预测组的几何；两侧直线及前沿圆点是未标定线模型示意，不代表真实端帽轮廓。状态条始终在原图之外。\n')
    for vid in ['video_1','video_2']:
        report.append(f'- [{vid}完整四方案对比]({vid}_evidence_comparison.mp4)、[T0/T1回归诊断板]({vid}_evidence_diagnostic_board.jpg)、[完整参数曲线]({vid}_parameter_traces.png)\n')
    report.append('T1有限四臂按协议收尾，运行成本和因果机制通过，真实几何准确性仍未验收。历史只证明能增加当前证据支持的状态更新，不能证明更准确；优先保留更简单的light_current/evidence_state作研究对照，不将history_search或model1接入默认。下一步先只读审核误接收/过期及观测-语义参考差异，形成单独的可见像素修正候选与协议；本轮不自动进入T2实现、训练、默认替换、结石S3或物理落点。任何训练仍须独立批准数据、初始化、实际增强、预算和回滚方案。\n')
    (OUT/'report.md').write_text('\n'.join(report),encoding='utf-8')
    artifacts.update({p.name:{'path':str(p),'sha256':sha256(p)} for p in [OUT/'report.md',OUT/'known_frame_diagnostics.json',OUT/'video_2_frame439_comparison.jpg']+list(OUT.glob('*_evidence_diagnostic_board.jpg'))+list(OUT.glob('*_parameter_traces.png'))})
    atomic_json(OUT/'report_artifacts.json',{'generator_sha256':sha256(Path(__file__)),'artifacts':artifacts,'new_inference':0,'video_display_semantics_clarified':True})
    print(json.dumps({'report':str(OUT/'report.md'),'artifacts':len(artifacts)}))


if __name__=='__main__':main()
