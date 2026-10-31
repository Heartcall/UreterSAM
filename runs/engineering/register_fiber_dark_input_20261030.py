"""Register completed diagnostic evidence; retain the existing default and all frozen protocols."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.fiber_reference_workspace import read,sha256
from ureter_semis.experiment import atomic_json

OUT=ROOT/'runs/a1_fiber_dark_input_control_20261030'
REPORT=ROOT/'docs/progress/2026-10-30-fiber-dark-input-control.md'
ARMS=['raw2','fiber_vlift','fiber_channel_gamma']
NAMES={'raw2':'原输入两网','fiber_vlift':'RGB共同缩放提亮','fiber_channel_gamma':'逐通道Gamma'}


def f(x):return '不可计算' if x is None else f'{x:.6f}'


def main():
    if REPORT.exists():raise FileExistsError(REPORT)
    s=read(OUT/'summary.json');compact=read(OUT/'analysis/compact_results.json');audit=read(OUT/'engineering_checks.json')
    if any(s['passing_dark_candidate'].values()):raise ValueError('This negative-result registration must not silently handle a passing arm')
    backup=OUT/'documentation_before';backup.mkdir()
    files=['README.md','docs/plans/segmentation_temporal_occlusion_plan.md','annotations/data_review_status_20261027.json','annotations/video_evaluation_registry.json']
    for name in files:
        path=backup/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((ROOT/name).read_bytes())
    lines=['# 暗色光纤：固定输入响应三臂完整对照','',
        '2026-10-30。**固定gamma0.7的两个全局光纤输入变换均未通过冻结质量门槛，不采用。** 三臂完成70帧视频参考、132张图像的新推理和两个完整视频，6个独立进程8082正式帧。没有继续扫gamma、改变阈值、训练、修改默认或进入结石记忆。',
        '', '## 实现与复现', '',
        '仅新增`tools/fiber_photometric_input.py`、`tools/run_fiber_dark_input_control.py`及机制测试；历史推理代码/权重/配置/标签/划分保持。原图经同一PIL双线性缩为256，float32 RGB在ImageNet归一化之前变换，不回uint8。共同缩放为C/max(C)×max(C)^0.7，黑像素保持0；逐通道为C^0.7。二者最大通道值相同，但Lab亮度与颜色比例不同，不能当纯照明/纯色相因果对照。',
        '', '原输入共享一份张量；变换臂额外准备并传输一份光纤张量。每帧仅原old190结石与对应A0 model1光纤两个前向，共49,202,790参数；没有多跑原光纤、TTA、原概率取max、几何、历史或GT推理输入。每个静态/完整过程用forward hooks核对实际分支次数。两份三类原图概率和原图掩膜均传回CPU。',
        '', '606次静态配对帧推理（每次两个分支）全部真实运行，raw2的原图掩膜/概率/指标精确复现。结石概率在全部静态及8082帧对应输出中精确一致；最终结石掩膜仍会受光纤优先覆盖变化。机制测试10项、既有类别分工/单成员回归6项通过；独立工程核验'+str(audit['passed'])+'项通过。合成测试与身份核验不替代实际质量。',
        '', '## 整体质量', '', '以下为重复开发参考上的语义像素指标；70帧不是2694帧稠密精度，132图像不是独立最终测试。', '']
    for scope,title in [('video_native','70帧视频参考（43/27）'),('image_native','132张图像开发集（68/64来源）')]:
        lines += ['### '+title,'','| 方案 | 结石Dice | 结石Precision | 结石Recall | 光纤Dice | 光纤Precision | 光纤Recall | 光纤边界F |','|---|---:|---:|---:|---:|---:|---:|---:|']
        for arm in ARMS:
            d=compact[scope][arm];m=d['metrics'];lines.append('| '+NAMES[arm]+' | '+' | '.join(f(m[k]) for k in ['stone_dice','stone_precision','stone_recall','fiber_dice','fiber_precision','fiber_recall'])+' | '+f(d['boundaries']['2']['f_score'])+' |')
        lines+=['','| 方案 | 光纤TP | 光纤FP | 光纤FN | 新增正确TP | 删除原正确TP | 新增FP | 移除FP | 原P≥0.9正确光纤被删 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for arm in ARMS:
            d=compact[scope][arm];m=d['metrics'];c=d['paired_changes']['fiber']
            lines.append('| '+NAMES[arm]+' | '+' | '.join(str(x) for x in [m['fiber_tp'],m['fiber_fp'],m['fiber_fn'],c['gained_tp'],c['lost_tp'],c['added_fp'],c['removed_fp'],d['baseline_highconfidence_TP_lost']])+' |')
        lines+=['','| 方案 | 结石TP | 结石FP | 结石FN | 结石边界F | 语义光纤骨架覆盖 | 实际改变帧数 |','|---|---:|---:|---:|---:|---:|---:|']
        for arm in ARMS:
            d=compact[scope][arm];m=d['metrics']
            lines.append('| '+NAMES[arm]+' | '+' | '.join(str(m[k]) for k in ['stone_tp','stone_fp','stone_fn'])+' | '+f(d['boundaries']['1']['f_score'])+' | '+f(d['semantic_centerline']['reference_coverage'])+' | '+str(d['changed_frames'])+' |')
        lines+=['']
    lines+=['形态学骨架覆盖仅表示参考语义光纤骨架被预测覆盖的比例，允许可见片段不连通；它不是真实物理中心线、杆体长度或尖端坐标。逐帧混淆矩阵、3px边界计数、变化、原29/新增41、两病例、来源及25/107历史曝光分组均保留在静态JSON与CSV。',
        '', '## 原图亮度分组与主要失败', '',
        '亮度始终按原RGB的OpenCV Lab L与HSV V编码[0,85,170,256]分组，不用提亮图重分组。该分组是显示图像明暗代理，不是设备照度或照明因果。GT内部距边界超过3×min(H,W)/400像素的统计也在JSON中单列。',
        '', '| 参考集合 | 方案 | 低Lab组GT | 低Lab组TP | 低Lab组FN | 低Lab组Recall | 中亮组Recall | 高亮组Recall |','|---|---|---:|---:|---:|---:|---:|---:|']
    for scope,label in [('video_native','70视频'),('image_native','132图像')]:
        for arm in ARMS:
            bins=compact[scope][arm]['brightness']['Lab_Lstar'];b=bins[0]
            lines.append('| '+label+' | '+NAMES[arm]+' | '+' | '.join(str(b[k]) for k in ['gt_pixels','tp','fn'])+' | '+' | '.join(f(x['recall']) for x in bins)+' |')
    lines+=['', '| 分组 | 方案 | 低Lab组Recall | 中亮组Recall | 高亮组Recall |','|---|---|---:|---:|---:|']
    for scope in ['video_native','image_native']:
        for group in compact[scope]['raw2']['groups']:
            for arm in ARMS:
                bins=compact[scope][arm]['groups'][group]['brightness']['Lab_Lstar']
                lines.append('| '+group+' | '+NAMES[arm]+' | '+' | '.join(f(x['recall']) for x in bins)+' |')
    lines += ['', '共同缩放提高低亮组召回，但视频新增4990个参考FP并删除1126个原TP，图像新增6309个FP、光纤Dice/Precision和边界退化。逐通道Gamma的视频整体Dice/边界改善，图像光纤却删除1709个原TP、净减少1450个TP，Recall下降，548个被删像素原P≥0.9；video_1中亮组和132中亮组也退化。两臂均未通过全部整体/TP保护条件，不能以总体Dice或暗组均值掩盖具体损失。',
        '', '原FN且原P<0.1的恢复：共同缩放视频1134/23574、图像344/4111，逐通道Gamma视频1684/23574、图像2/4111。这里的恢复必须与其他位置的损失和新增FP一起看；网络外观响应有变化，但不是普遍暗端修复。具体原低P区域的新概率分布及高置信FP在`analysis/probability_response.json`。',
        '', '### 固定video_2显示第1209帧', '',
        '整帧人工语义参考支持暗蓝可见前段被漏分；没有独立尖端ROI/端点标注。本帧低Lab组GT2776，原TP150/FN2626，共同缩放TP202/FN2574、逐通道TP189/FN2587，分别只净恢复52/39像素。可视化及连续概率显示两变换仍在杆体内部明暗过渡附近截止，大部分暗前段依旧低P。这不能证明亮度是唯一原因，也不能把GT轴投影当真实尖端误差。',
        '', '![第1209帧：原图、整帧参考、亮度、三臂掩膜与概率](../../runs/a1_fiber_dark_input_control_20261030/analysis/video_2_frame1209_comparison.jpg)',
        '', 'video_1第78帧以及video_2第311/439/622/643/1266帧均保留固定对照；后五帧没有像素参考，不能报告准确率。全部202张静态图均输出对比板，不只展示成功帧；53条逐帧TP损失违反记录（共同缩放10、逐通道43）原样保留。按已有标签记为FP的组织/亮斑仍有来源标签完整性限制，不能自动反推所有物体真实身份。',
        '', '## 两段完整视频与成本', '',
        '每臂独立新进程运行video_1全部722、video_2全部1972帧，共6进程8082正式配对帧；每进程12预热帧另记（共72）。全部帧RGB身份/PTS对齐，标注70帧的掩膜及两份概率与静态完全一致。完整视频没有稠密参考，精度只能在已有参考处报告；42/120固定抽样尚未齐备，不外推全片准确率。',
        '', '| 方案 | 视频 | 改变帧数 | 总管线中位/p95/max ms | 变换p95 ms | 含解码/核验/导出FPS | CPU峰值MiB | GPU峰值MiB |','|---|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        for vid,d in s['full_video'][arm].items():
            t=d['timings']['pipeline_total_ms'];lines.append('| '+NAMES[arm]+' | '+vid+' | '+str(d['changed_frames'])+' | '+ '/'.join(f'{t[k]:.2f}' for k in ['p50','p95','max'])+' | '+f"{d['timings']['fixed_transform_ms']['p95']:.3f}"+' | '+f"{d['offline_fps_including_decode_audit_export']:.2f}"+' | '+f"{d['windows_peak_working_set_mib']:.1f}"+' | '+f"{d['cuda_peak_allocated_mib']:.1f}"+' |')
    lines+=['', '| 方案 | 视频 | RGB/PIL p95 | 归一化/传输p95 | 结石前向p95 | 光纤前向p95 | 原图映射/传回p95 | 总耗时门槛 |','|---|---|---:|---:|---:|---:|---:|---|']
    for arm in ARMS:
        for vid,d in s['full_video'][arm].items():
            lines.append('| '+NAMES[arm]+' | '+vid+' | '+' | '.join(f"{d['timings'][k]['p95']:.3f}" for k in ['rgb_pil_grid_ms','normalize_transfer_ms','stone_forward_ms','fiber_forward_ms','native_mapping_transfer_ms'])+' | '+('通过' if d['pipeline_p95_within_source_interval'] else '未通过')+' |')
    lines+=['', '各阶段分位数不能相加/相减推导总耗时，本表总管线由每帧同步直接计时。CPU用Windows原生内存接口；GPU为torch峰值分配。独立进程与分支计数、运行环境、各阶段p50/p95/max、CPU过程时间、GPU保留内存、源帧预算及所有逐帧时间在对应JSON。成本能满足源帧p95预算并不代表质量通过，也不代表已验证临床实时显示/队列尾延迟。',
        '', '完整视频上方为原手术RGB与三个原坐标掩膜，下方明确标为光纤网络256输入预览；原RGB没有被替换。',
        '', '- [video_1完整三臂对比](../../runs/a1_fiber_dark_input_control_20261030/video_1_comparison.mp4)',
        '- [video_2完整三臂对比](../../runs/a1_fiber_dark_input_control_20261030/video_2_comparison.mp4)',
        '', '## 验收、证据边界和回滚', '',
        '原29门槛均通过，70/132与同轮raw2及冻结classwise3的八项不退化、亮度组保护、粗删除TP及逐帧损失等预冻结条件均如实检查。共同缩放有15项拒绝，逐通道Gamma有20项拒绝，详见`static_quality.json`；两臂即使暗组FN严格减少也没有通过总验收。raw2是控制，未满足“严格改善自己”的目标不作为新方法失败；它对classwise的既有质量取舍仍保留。',
        '', '几何拟合/接受/回退不适用于本轮（没有几何分支）。变换固定、无自适应状态、无隐藏轮廓填充；新增分割只来自当前网络，不等于每个像素都有可靠真实物体证据。实验输出保留，不部署；默认A0与历史源代码未改，无需覆盖权重回滚。若日后在实验工具禁用变换，raw2路径即保留原输入。',
        '', '现12帧尖端专项仍待确认，14帧工作单仅规划；本轮没有新增人工/预测标签。尖端ROI Recall/Precision、真实端点/长度/角度、激光落点及遮挡恢复全部未验证，JSON保持null。video_1/video_2及派生物、stone_new均不训练；521训练及144保留test像素读取0，原521/132/144/29、当前70与参考身份不变。Godot不作真实标定。',
        '', '## 后续最小一步', '',
        '停止这两个全局变换，不继续调gamma或自动训练。下一步先制定并冻结“当前图像中杆体与暗前段的局部证据”协议：利用已有失败帧区分内部颜色/亮度过渡和真实外边界，核对仅从原网络杆体、当前RGB色度/局部对比与双侧边缘提出暗前段候选的覆盖及负例；几何只限制候选搜索，不直接填充。没有可靠证据或存在歧义时回退原输出，并保留断开的可见片段。先做无需训练的可证伪小对照，任何局部融合/搜索扩展另行固定范围与保护条件后实施。',
        '', '本轮不能证明局部外观模型一定有效，也没有排除标签、低分辨率与形状泛化等原因。只有后续轻量证据对照表明网络泛化是主要限制，再单独提出有标签光度增强/监督或轻量尖端改进训练计划，明确合规初始化、数据、成本、验证和回滚并等待批准。T2-P旧边界对照保留未执行，不放宽P≥0.1的保护范围，S3不启动。',
        '', '## 可复现产物', '',
        '- [冻结配置](../../configs/fiber_dark_input_control_20261030.json)、[先行协议](../plans/fiber_dark_input_control_20261030.md)',
        '- [结果汇总与完整拒绝项](../../runs/a1_fiber_dark_input_control_20261030/summary.json)',
        '- [逐帧202张×3臂指标CSV](../../runs/a1_fiber_dark_input_control_20261030/analysis/per_frame_metrics.csv)',
        '- [全部202张静态对比板](../../runs/a1_fiber_dark_input_control_20261030/analysis/all_static_boards)',
        '- [独立核验](../../runs/a1_fiber_dark_input_control_20261030/engineering_checks.json)、[输入/源码/权重身份](../../runs/a1_fiber_dark_input_control_20261030/preflight.json)',
        '- [源码快照](../../runs/a1_fiber_dark_input_control_20261030/source_snapshot.zip)', '']
    REPORT.write_text('\n'.join(lines),encoding='utf-8')
    update='2026-10-30 暗段固定输入完整对照更新（覆盖以上历史下一步）：三臂606静态及6独立进程8082完整帧全部完成，原输入与结石概率精确复现。共同缩放提高低亮召回却增加FP/删除TP；逐通道Gamma视频光纤Dice0.951919、图像0.770020，但图像Recall下降且删除1709个原TP。第1209帧低亮FN2626只净恢复52/39，主要暗段仍漏分；两臂均拒绝，未调gamma或采用。下一步仅制定当前图像杆体/暗前段局部证据与内部过渡排除协议，保护真实像素并先验证候选/负例，不自动训练、改默认、放宽T2-P或进入S3。[完整结果与两片视频](../progress/2026-10-30-fiber-dark-input-control.md)。'
    mainplan=ROOT/'docs/plans/segmentation_temporal_occlusion_plan.md';text=mainplan.read_text(encoding='utf-8').replace('版本：v2.21，','版本：v2.22，',1)
    old=next(line for line in text.splitlines() if line.startswith('- [ ] **当前下一步：暗端固定输入三臂实现与完整对照**'))
    replacement='- [x] **暗端固定输入三臂实现与完整对照**：606静态、6进程8082完整帧、10新机制+6回归测试及'+str(audit['passed'])+'项独立核验完成；raw2精确复现、结石概率不变。共同缩放暗组召回提高却增加FP/损失TP；逐通道Gamma视频Dice改善而图像Recall退化；1209主要暗段仍低P。两臂未通过，停止全局gamma路线，默认不变。详见[结果](../progress/2026-10-30-fiber-dark-input-control.md)。\n- [ ] **当前下一步：局部杆体/暗前段证据协议**：先分离内部颜色/亮度过渡与真正可见外边界，设计仅从原神经杆体和当前RGB/边缘提出暗前段候选的覆盖及负例对照；几何只辅助搜索，歧义回退，不填隐藏形状。先冻结局部范围、保护与失败条件，再实施，不自动训练、改默认或S3。尖端专项待独立确认，T2-P原协议保留未执行。'
    if old not in text:raise ValueError('Main plan anchor missing')
    mainplan.write_text(text.replace(old,replacement,1)+'\n\n'+update+'\n',encoding='utf-8')
    readme=ROOT/'README.md';text=readme.read_text(encoding='utf-8');anchor='## 实现说明'
    if text.count(anchor)!=1:raise ValueError('README anchor')
    readme.write_text(text.replace(anchor,update.replace('../progress/','docs/progress/')+'\n\n'+anchor,1),encoding='utf-8')
    statuspath=ROOT/'annotations/data_review_status_20261027.json';status=read(statuspath)
    status.update(fiber_dark_input_implemented=True,fiber_dark_input_executed=True,fiber_dark_input_adopted=False,
        fiber_dark_input_summary=str(OUT/'summary.json'),fiber_dark_input_report=str(REPORT.relative_to(ROOT)).replace('\\','/'),
        fiber_dark_input_passing_candidates=s['passing_dark_candidate'],fiber_dark_input_static_paired_frames=606,fiber_dark_input_full_paired_frames=8082,
        current_execution_checks=str(OUT/'engineering_checks.json'),
        implementation_plan_revision='v2.22_20261030_dark_input_control_completed_rejected',
        latest_instruction='Continue frozen fixed dark-input control implementation and full-video evaluation',
        latest_instruction_effect='606static+8082full completed, both fixed transforms rejected; original weights/labels/default retained, no training/S3.',
        plan_next_action='Design and freeze local shaft/dark-front current-image evidence/negative-control protocol; no automatic implementation or training.')
    atomic_json(statuspath,status)
    registry_path=ROOT/'annotations/video_evaluation_registry.json';registry=read(registry_path)
    registry['previous_inference_control_before_dark_input']={k:registry[k] for k in ['latest_inference_control','latest_inference_control_result','latest_inference_control_result_sha256','latest_control_adopted']}
    registry.update(latest_inference_control='configs/fiber_dark_input_control_20261030.json',latest_inference_control_result=str(OUT/'summary.json'),
        latest_inference_control_result_sha256=sha256(OUT/'summary.json'),latest_control_adopted=False,latest_plan_executed=True,
        latest_dark_input_report=str(REPORT.relative_to(ROOT)).replace('\\','/'),latest_dark_input_accuracy_scope='70video+132image development references, not dense whole-video or anatomical tip accuracy',
        current_next_step='design_local_shaft_dark_front_current_image_evidence_protocol')
    atomic_json(registry_path,registry)
    atomic_json(OUT/'registration.json',{'generator_sha256':sha256(Path(__file__)),'plan_revision':'v2.22','result_summary_sha256':sha256(OUT/'summary.json'),
        'artifacts':{name:sha256(ROOT/name) for name in files+[REPORT.relative_to(ROOT).as_posix()]},
        'backup':str(backup),'default_changed':False,'training_updates':0,'adopted':False})
    print(json.dumps({'registered':True,'plan':'v2.22','checks':audit['passed'],'adopted':False}))


if __name__=='__main__':main()
