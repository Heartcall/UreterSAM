"""Post-run figures/report. No new inference or threshold selection."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from tools.fiber_reference_workspace import read,sha256
from tools.run_fiber_blue_aplus import panel
from ureter_semis.experiment import atomic_json
OUT=ROOT/'runs/a1_fiber_temporal_state_gate_20261030'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])+'\n'
def f(x):return '不可计算' if x is None else f'{x:.6f}' if isinstance(x,float) else str(x)
def logs(arm,vid):return [json.loads(x) for x in (OUT/f'full_video/{arm}/{vid}/frames.jsonl').read_text().splitlines()]
def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s=read(OUT/'summary.json');cfg=read(OUT/'preflight.json')['config'];a=read(OUT/'engineering_checks.json');g=read(OUT/'geometry_diagnostics.json')
    selections={};font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
    for vid in ['video_1','video_2']:
        data={arm:logs(arm,vid) for arm in cfg['arms']};records=data['gated_state_only'];picks=[]
        def add(i,reason):
            if 0<=i<len(records) and i not in [x[0] for x in picks]:picks.append((i,reason))
        rejected=[i for i,r in enumerate(records) if r['tracked']['status']=='rejected_jump_pending'];confirmed=[i for i,r in enumerate(records) if r['tracked']['status']=='confirmed_relocation']
        if rejected:add(rejected[0],'first rejected jump')
        if confirmed:add(confirmed[0]-1,'before first confirmation');add(confirmed[0],'first confirmation')
        distances=[]
        for i in range(len(records)):
            p=data['legacy_state_only'][i]['tracked']['geometry'];q=records[i]['tracked']['geometry']
            distances.append(float(np.linalg.norm(np.array(p['tip_xy'])-q['tip_xy'])) if p is not None and q is not None else -1)
        add(int(np.argmax(distances)),'largest legacy/gated endpoint separation (not GT error)')
        for i in [len(records)//4,len(records)//2,3*len(records)//4]:
            if len(picks)<4:add(i,'uniform timeline context')
        picks=picks[:4];selections[vid]=[{'frame_index':i,'display_frame':i+1,'reason':r} for i,r in picks]
        board=Image.new('RGB',(1600,len(picks)*478),(10,10,10));draw=ImageDraw.Draw(board)
        for j,(i,reason) in enumerate(picks):
            rgb=np.array(Image.open(ROOT/f'runs/fiber_reference_d1_20261029/images/{vid}/{i:06d}.png').convert('RGB'))
            import hashlib
            assert hashlib.sha256(rgb.tobytes()).hexdigest()==records[i]['source_rgb_raw_sha256']
            frame=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR);mask=np.array(Image.open(records[i]['mask_reference']))
            panels=[panel(frame,'Input',None,i)]
            for arm in cfg['arms'][1:]:
                state=data[arm][i]['tracked'];pic=panel(frame,arm,mask,i);geo=state['geometry'];color=(0,230,255) if state['prediction_only'] else (255,255,0)
                if geo:
                    for segment in geo['side_segments']:
                        p,q=[tuple(np.rint(x).astype(int)+[0,34]) for x in segment];cv2.line(pic,p,q,color,1,cv2.LINE_AA)
                    cv2.circle(pic,tuple(np.rint(geo['tip_xy']).astype(int)+[0,34]),4,color,1)
                panels.append(pic)
            for k,pic in enumerate(panels):
                board.paste(Image.fromarray(cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)),(400*k,j*478))
                if k:
                    state=data[cfg['arms'][k]][i]['tracked'];draw.text((400*k+3,j*478+436),f"{state['status']} age={state['age']} pending={state['pending_count']}",font=font,fill='white')
            draw.text((4,j*478+457),f'{vid} frame{i+1}: {reason}',font=font,fill='white')
        board.save(OUT/f'{vid}_state_diagnostic_board.jpg',quality=94)
        fig,axes=plt.subplots(6,1,figsize=(13,11),sharex=True)
        for arm,color in [('current_observation','0.5'),('legacy_state_only','#2873ad'),('gated_state_only','#dd7a16')]:
            t=np.array([r['timestamp_seconds'] for r in data[arm]]);values=np.full((len(t),6),np.nan)
            for i,r in enumerate(data[arm]):
                st=r['tracked'];z=st['state'];geo=st['geometry']
                if z is not None:
                    values[i]=[*geo['tip_xy'],z[3],np.rad2deg(z[2]),z[4],np.rad2deg(geo['signed_taper_angle_rad'])]
            for k,axis in enumerate(axes):axis.plot(t,values[:,k],lw=.8,color=color,label=arm)
        for axis,label in zip(axes,['Derived tip X (px)','Derived tip Y (px)','Length (px)','Axis (deg, wrapped)','Entry width (px)','Signed taper (deg)']):
            axis.set_ylabel(label);axis.grid(alpha=.2)
        axes[0].legend(loc='best',ncol=3);axes[0].set_title(vid+': uncalibrated geometry trajectories, NOT accuracy or physical tip truth');axes[-1].set_xlabel('Video time (s)')
        fig.tight_layout();fig.savefig(OUT/f'{vid}_parameter_traces.png',dpi=150);plt.close(fig)
    atomic_json(OUT/'diagnostic_selection.json',{'policy':'first jump/confirmation, largest state disagreement and uniform context; all frames reported in logs, no success-only scoring','videos':selections})
    report=['# T0光纤时序几何状态对照（2026-10-30）\n',
    '已完成冻结四臂、202张静态复现和8个独立进程的10776帧完整回放。新门控机制可运行，但存在当前观测被延迟接收的可视化失败，完整几何管线也超源帧预算，不推荐直接采用固定三帧确认作为更优方案。所有分割像素逐帧保持原model1，不把状态变化称为分割准确率改善。默认A0、模型权重、标签/划分与历史结果不变；没有训练、T1搜索或结石S3。\n',
    '## 固定设计与实际实现\n',
    'raw2仅两网；current_observation加入当前RGB/FOV/双侧边缘/端帽拟合；legacy_state_only使用原alpha-beta并立即接收越界观测重置；gated_state_only使用新异常暂存/确认规则。三几何臂的当前拟合逐帧完全相同。旧拟合器、旧跟踪器、alpha0.65/beta0.10、尖端0.08D/长度0.10D/方向20°/宽度比[0.5,2]门槛均保持。新入口与状态器独立，不调用visible_fiber/rasterize，不裁剪、填充、连通组件或额外运行网络。\n',
    '几何状态为入口x/y、轴角、长度、两端宽度、端帽轴比；末端及两侧方向从同一状态推导，不独立平滑。新状态器只在三个连续且相互一致的越界观测后，于第三次当前时刻重定位，前两帧不回改；最多3帧仅预测，第4帧清空活动状态。待确认与活动过期分开，不借重新初始化绕过确认。静态图每张清空，视频在正式帧0前清空预热状态；源/分辨率/异常时间变化清空两分支。\n',
    '所有状态只是二维图像几何，没有校准协方差/概率置信区间，尖端标记不是物理激光落点。第一观测可能本来就错，连续错误也能通过确认；此轮没有用历史重选组件或重新搜索图像。当前拟合器仍从最大合格入口邻近组件起拟合，其component_provenance是观测选择诊断，组件标签不是实例身份，原始片段没有被删除。\n',
    '## 完整回放：状态、拟合与失效\n']
    report.append(table(['臂','视频','帧数','当前有效拟合','有状态','仅预测','状态事件'],[[arm,vid,v['frames'],v['fit_accepted_frames'],v['state_available_frames'],v['prediction_only_frames'],json.dumps(v['states'],ensure_ascii=False)] for arm,items in s['full_video'].items() for vid,v in items.items()]))
    for vid in ['video_1','video_2']:
        report.append('\n'+vid+'拟合拒绝原因：'+json.dumps(s['full_video']['current_observation'][vid]['fit_reasons'],ensure_ascii=False)+'。\n')
        events=g['videos'][vid]['gated_events'];report.append(vid+'全部门控事件（显示帧）：'+', '.join(str(e['display_frame'])+':'+e['status'] for e in events)+'。\n')
    report.append('拒绝不是观测必错，确认也不是运动真实；没有独立几何标注，不能把更少跳动、更多可用状态或更小观测残差当作尖端准确率。当前观测缺失时，预测仍单独标记，不变成可见掩膜。真实快速移动可能带来两帧确认延迟，过期也可能丢失真实轨迹；解析合成轨迹的14项测试证明机制，不能给真实视频事件自动贴“真/假”标签。\n')
    report.append('**定性失败检查：video_2显示帧439–440，原RGB中的可见蓝色光纤已处于当前拟合所支持的新位置，而新门控黄色预测仍落在旧位置、明显偏离当前光纤；441才确认重定位。** 这支持“固定等待三次会造成可见跟随延迟”的定性结论，不能仅以较小速度p95称更好。显示帧643的最大新旧端点分离也保留在诊断板。没有独立点真值，不把状态到当前拟合的距离称为尖端定位误差，不据此给全部12次暂存判错；这是可视化审核记录，不是新增GT标注。\n')
    report.append('## 几何运动诊断（非真值误差）\n')
    report.append(table(['视频','臂','共同可用帧','共同有效相邻对','派生末端速度p50/p95 px/s','轴角速度p95 deg/s','长度速度p95 px/s'],[[vid,arm,v['common_available_frames'],m['tip_speed_px_per_s']['n'],f"{f(m['tip_speed_px_per_s']['p50'])}/{f(m['tip_speed_px_per_s']['p95'])}",f(m['axis_speed_deg_per_s']['p95']),f(m['length_speed_px_per_s']['p95'])] for vid,v in g['videos'].items() for arm,m in v['motion_common_available_pairs'].items()]))
    report.append('geometry_diagnostics.json同时保留各臂全部可用对、共同可用对、加速度及状态相对当前拟合的残差。条件集合的帧数明确列出，未将无拟合或无状态帧从全体结果剔除。轴角按角差wrap统计；图中角度跨±π的绘制折点不自动判为物理跳变。没有光流补偿或相机运动标定。\n')
    report.append('## 分割质量不变\n')
    q=[]
    for scope in ['video_native','image_native']:
        d=read(OUT/(scope+'.json'));m=d['summary']['metrics']
        for cls,c in [('stone','1'),('fiber','2')]:q.append([scope,'全部四臂',cls,*[f(m[cls+'_'+k]) for k in ['dice','precision','recall']],f(d['summary']['boundaries'][c]['f_score']),m[cls+'_tp'],m[cls+'_fp'],m[cls+'_fn']])
    report.append(table(['参考','臂','类别','Dice','Precision','Recall','边界F','TP','FP','FN'],q))
    report.append('202张静态实际新推理一次，四臂共享这些相同掩膜，不虚报为808次静态前向；原图两专家三类FP32概率、掩膜和汇总均与既有model1逐值一致。所有10776帧逐帧核验旧原图掩膜/RGB，保存不可变原生掩膜引用和哈希。各类原TP删除、新增TP、FP移除/新增均0。原29/新增41、43/27病例、68/64来源、25/107曝光及小结石分组保留；三网classwise质量另存冻结参照。\n')
    report.append('原29初筛和此前model1相对三网各项不退化失败判断不变；不以相同掩膜通过本轮不变性检查就采用model1或验收S2。85/523个既有光纤-结石冲突未解决。70帧是重复开发验证；42/120随机参考未完成，不外推全片精度；独立尖端/方向真值仍无，tip_metrics为空。\n')
    report.append('## 独立进程完整性能\n')
    report.append(table(['臂','视频','管线p50/p95/max ms','拟合p50/p95 ms','状态p50/p95 ms','周期p50/p95 ms','含视频审计fps','CUDA MiB','源帧p95预算'],[[arm,vid,f"{v['timings']['pipeline_total_ms']['p50']:.2f}/{v['timings']['pipeline_total_ms']['p95']:.2f}/{v['timings']['pipeline_total_ms']['max']:.2f}",f"{v['timings']['fit_fov_ms']['p50']:.2f}/{v['timings']['fit_fov_ms']['p95']:.2f}",f"{v['timings']['state_update_ms']['p50']:.3f}/{v['timings']['state_update_ms']['p95']:.3f}",f"{v['timings']['cycle_ms']['p50']:.2f}/{v['timings']['cycle_ms']['p95']:.2f}",f"{v['offline_fps_including_render_audit_encode']:.2f}",f"{v['cuda_peak_allocated_mib']:.2f}",v['pipeline_p95_within_source_interval']] for arm,items in s['full_video'].items() for vid,v in items.items()]))
    report.append('同轮匹配raw2基线、两张实际U-Net共49,202,790参数、RTX4070Ti、FP32、batch1、12帧临时状态预热后清空。管线包括RGB/PIL、两网、原图掩膜及两份三类概率传CPU、当前FOV/拟合和状态更新/派生。周期另含解码、视频编码、哈希/选中帧概率审计、组件来源报告；编码释放计入离线fps，最终JSON写盘与模型加载在计时之外。组件来源重复计算是诊断开销，位于管线计时外、周期内。只与本轮比较，不相减两个p95或与历史接口混用。峰值是CUDA allocated，不代表设备总占用或CPU内存。\n')
    report.append('每片时间基准：'+json.dumps({vid:s['full_video']['raw2'][vid]['timestamp_mode'] for vid in ['video_1','video_2']},ensure_ascii=False)+'。可靠性预扫描验证单调PTS及RGB身份；不用推理墙钟速度代替视频时间，70稀疏帧也未串成连续序列。p95筛查不是实际采集/显示队列验收，所有失败臂仍完整回放。\n')
    report.append('## 判断与下一步\n')
    gated=s['full_video']['gated_state_only'];passed=all(v['pipeline_p95_within_source_interval'] for v in gated.values())
    report.append('新状态器的因果、异常拒绝与三次确认机制已通过测试和完整日志重放；真实几何准确性仍未验证，分割质量没有变化。门控臂本轮管线源帧p95筛查：'+('两片均通过' if passed else '至少一片未通过')+'。应同时看拟合与状态阶段耗时，不能因为状态向量很小就把整体管线称为实时。\n')
    report.append('按冻结协议结束T0，不继续改alpha/beta、跳变门槛、确认数或TTL。下一步先另行制定T1候选搜索协议：保留全图当前分割，历史只限定低成本当前图像观测范围，以当前概率/双侧边缘/对比度验证候选；**当前证据充分时允许覆盖历史，不能一律等待三帧；当前证据不足时才更多依赖连续性，必要时区分镜头运动与光纤自身变化。** 无证据保持拒绝，允许可见断片。若当前拟合已主导成本，先在方案中限定轻量观测和拟合预算，不能直接叠加旧重网裁剪或模板相交。T1必须有无历史的匹配对照和同样完整两片测试；未冻结前不实施。没有几何真值时仍只能报告诊断，任何可见像素融合、训练、默认采用或S3都不由本次自动授权。\n')
    report.append(f"11项新增状态测试+3项旧几何回归（共14）通过；{a['passed']}项独立核验通过。重新检查202静态与10776完整掩膜记录、8个进程、三臂相同当前观测、旧跟踪器独立重放、新状态器全片与半片前缀重放、阶段计时代数/分位数及10段视频编码/哈希。源/权重/标签/划分/历史结果均冻结。\n")
    for vid in ['video_1','video_2']:report.append(f'- [{vid}完整五栏对比]({vid}_state_comparison.mp4)、[状态诊断板]({vid}_state_diagnostic_board.jpg)、[完整参数曲线]({vid}_parameter_traces.png)\n')
    report.append('\n叠加图中青色为当前观测支持的拟合/滤波状态，黄色为仅历史预测，洋红为待确认端部；所有均是未标定几何。视频状态条占用画面底部一小段显示空间，左侧Input始终保留完整原图，原生分割参考不受影响；诊断板在图外列状态，完整原图可另查引用。\n')
    (OUT/'report.md').write_text('\n'.join(report),encoding='utf-8')
    files=[OUT/'report.md',OUT/'diagnostic_selection.json']+[OUT/f'{vid}_{suffix}' for vid in ['video_1','video_2'] for suffix in ['state_diagnostic_board.jpg','parameter_traces.png']]
    atomic_json(OUT/'report_artifacts.json',{'generator_sha256':sha256(Path(__file__)),'files':{p.name:sha256(p) for p in files}})
    print(json.dumps({'report':str(OUT/'report.md'),'checks':a['passed']}))
if __name__=='__main__':main()
