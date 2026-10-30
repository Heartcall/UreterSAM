"""Publish the fixed experiment, including rejected results and budget-control attribution."""
from collections import Counter
from pathlib import Path

import numpy as np

from tools.fiber_reference_workspace import ROOT, read, load_mask, sha256
from ureter_semis.experiment import atomic_json

out = ROOT / 'runs/a1_fiber_blue_aplus_20261029'
config = read(ROOT / 'configs/fiber_blue_aplus_20261029.json'); s = read(out / 'summary.json')
video = read(out / 'video_native/summary.json'); image = read(out / 'image_native/summary.json')
audit = read(out / 'engineering_checks.json'); center = read(out / 'centerline_coverage_proxy.json')
if not audit['passed']: raise RuntimeError('Do not publish failed engineering audit')
attribution = {}
for scope, manifest_path in [('video_native', config['video_manifest']), ('image_native', config['image_manifest'])]:
    result = video if scope == 'video_native' else image; groups = {'neither_arm_budget_rejected': Counter(), 'budget_rejection_present': Counter()}
    for i, row in enumerate(read(ROOT / manifest_path)['val']):
        with __import__('PIL.Image', fromlist=['Image']).open(row['mask']) as im: target = np.asarray(im).copy()
        raw = load_mask(out / f'{scope}/raw/{i:04d}.png', target.shape, {0, 1, 2})
        control = load_mask(out / f'{scope}/neural_relaxation_control/{i:04d}.png', target.shape, {0, 1, 2})
        blue = load_mask(out / f'{scope}/blue_aplus/{i:04d}.png', target.shape, {0, 1, 2})
        a = result['records'][i]['application']; conflict = any(a[n]['fallback'] == 'addition_budget' for n in ('neural_relaxation_control', 'blue_aplus'))
        g = groups['budget_rejection_present' if conflict else 'neither_arm_budget_rejected']; g['frames'] += 1
        excluded_by_blue = (control == 2) & (blue != 2)
        newly_in_blue = (blue == 2) & (control != 2)
        if not conflict and newly_in_blue.any(): raise AssertionError('Without budget rejection, blue support must be a subset of uniform boost')
        g['control_TP_not_kept_by_blue'] += int((excluded_by_blue & (target == 2)).sum())
        g['control_FP_avoided_by_blue'] += int((excluded_by_blue & (target != 2) & (target != 255)).sum())
        g['blue_TP_missing_from_control'] += int((newly_in_blue & (target == 2)).sum())
        g['blue_FP_missing_from_control'] += int((newly_in_blue & (target != 2) & (target != 255)).sum())
    attribution[scope] = {k: dict(v) for k, v in groups.items()}
atomic_json(out / 'color_control_attribution.json', {'groups': attribution,
    'interpretation': 'Shared budget has nonlinear fallback: a smaller blue proposal can pass when uniform boost is fully rejected. Separate these cases before attributing all gains to color discrimination.',
    'source_sha256': sha256(Path(__file__))})

fmt = lambda x: 'N/A' if x is None else f'{x:.6f}'
labels = {'raw': '冻结classwise', 'neural_relaxation_control': '纯概率放宽对照', 'blue_aplus': '蓝色A+'}
lines = ['# A+蓝色正向软补充：静态与两个完整视频对照（2026-10-29）', '',
    '**结论：两种新增方案均未通过，不采用。** 蓝色A+在视频已审70帧略增Dice和Recall，但降低Precision；132图像的Dice、Precision与边界F下降，且CPU分支增加约13ms，完整视频p95约54–56ms。所有原光纤和结石像素均保留，仍不能由此推出整体质量达标。', '',
    '本轮先将用户新增5帧加入冻结版本：当前70帧（video_1为43，video_2为27），原29、累计新增41及上一版65均单列保留。原65到新70的基线指标变化来自参考集合增加，不能称模型训练退化。11个历史方案已按70帧重算，40项入集核验通过。', '',
    '用户明确允许“先做A+，尖端指标待补”。当前只有整帧语义参考，没有独立尖端/杆体标注；本轮没有生成解剖尖端真值。120帧概率样本完成42帧，不发布全片准确率估计。', '',
    '## 固定设计与历史方案的区别', '',
    '- 原classwise：旧190单网负责结石，A0双网平均logits负责光纤，先映射FP32概率到原图再argmax，光纤优先；共3网。默认demo及其A0权重不替换。',
    '- 蓝色证据取`e=(B-max(R,G))/(R+G+B+eps)`，`Pblue=sigmoid((e-.02)/.06)`。这是未校准的解析外观分数，不是从验证数据拟合的身份概率。色彩可靠性为`min(S/.25,1)*min(V/.2,1)`；白色/低饱和只减小加分，不删除原像素。',
    '- 只在原网络光纤2像素邻域（400像素尺度）内的原背景像素，给`log(Pfiber/max(Pbackground,Pstone))`增加至多0.35的颜色支持；无图像/网络支持的隐藏轮廓不填充。不裁切原掩膜、不选最大组件、允许可见片区不连通。没有全轮廓模板、轴线推理或时序状态。',
    '- 提议新增数超过`min(ceil(原光纤面积*.05),ceil(图像面积*.001))`时整帧回退原掩膜，不能取top-k或临时调预算。无足够原光纤锚点、非有限概率或没有支持像素也回退。',
    '- 纯概率放宽对照使用相同邻域、保护及预算，但统一加0.35；用于判断收益是否只是阈值变宽。两种新方案全部跑两个完整视频，不因静态失败免测。',
    '- 与旧16×16 HSV后验＋亮度硬门控不同：不估计共享颜色模型、不读TRAIN像素、不把低颜色分当背景、不与固定模板求交。没有训练、参数拟合、144个保留test像素读取、默认变更或结石记忆。', '',
    '协议和源码在计算结果前冻结，无阈值搜索。旧29/132门槛保留；本次更严格地要求原29、新41、合计70与132图像的光纤Dice/Precision/Recall/边界F均不退化，至少一组Dice改善、原光纤和结石像素零删除。两组都触发退化拒绝，未放宽门槛。', '',
    '## 静态分割质量', '',
    '| 参考 | 方案 | 结石Dice | 光纤Dice | 光纤Precision | 光纤Recall | 光纤边界F |', '|---|---|---:|---:|---:|---:|---:|']
for scope, d in [('视频70帧', video), ('图像132张', image)]:
    for name, m in d['models'].items():
        q = m['metrics']; lines.append(f'| {scope} | {labels[name]} | {fmt(q["stone_dice"])} | {fmt(q["fiber_dice"])} | {fmt(q["fiber_precision"])} | {fmt(q["fiber_recall"])} | {fmt(m["boundaries"]["2"]["f_score"])} |')
lines += ['', '结石预测逐像素保持不变：视频70帧结石Precision/Recall为0.899826/0.628722，132图像为0.880229/0.934820。图像及病例/来源分组完整三类指标、逐帧TP/FP/FN与3像素容差边界F保留在静态summary.json中。', '',
    '| 参考 | 方案 | 光纤TP | 光纤FP | 光纤FN | 新恢复TP | 新增FP | 删除原正确像素 |', '|---|---|---:|---:|---:|---:|---:|---:|']
for scope, d, folder in [('视频70帧', video, 'video_native'), ('图像132张', image, 'image_native')]:
    for name, m in d['models'].items():
        q = m['metrics']; delta = audit['static_changes'][folder]['paired'].get(name, {})
        lines.append(f'| {scope} | {labels[name]} | {q["fiber_tp"]} | {q["fiber_fp"]} | {q["fiber_fn"]} | {delta.get("gained_tp",0)} | {delta.get("added_fp",0)} | {delta.get("lost_tp",0)} |')
lines += ['', '蓝色A+视频新增3844像素中2像素在未知区，不计TP/FP；新增图像718像素全部有参考。它在视频恢复2551个TP同时引入1291个FP；图像只恢复106个TP却增加612个FP。这不是靠牺牲Recall提高Precision的结果，而是增加召回仍不足以抵消误检与边界退化。', '',
    '## 原29帧与病例分组', '', '| 分组 | 帧数 | 方案 | 光纤Dice | Precision | Recall |', '|---|---:|---|---:|---:|---:|']
for group in ('original29', 'new41', 'video_1', 'video_2'):
    g = video['groups'][group]
    for name, m in g['models'].items():
        q = m['metrics']; lines.append(f'| {group} | {g["frames"]} | {labels[name]} | {fmt(q["fiber_dice"])} | {fmt(q["fiber_precision"])} | {fmt(q["fiber_recall"])} |')
lines += ['', '132图像的68/64来源与25个旧190历史曝光组、107个未匹配组分别保留，没有混合其泛化含义，也没有读取144个保留test关联组像素。', '',
    '## 颜色贡献与预算回退', '', '| 参考 | 分组 | 帧数 | 颜色过滤损失的对照TP | 颜色避免的对照FP | 蓝色有而对照无的TP | 蓝色有而对照无的FP |', '|---|---|---:|---:|---:|---:|---:|']
for scope, groups in attribution.items():
    for label, g in groups.items():
        lines.append(f'| {scope} | {label} | {g["frames"]} | {g["control_TP_not_kept_by_blue"]} | {g["control_FP_avoided_by_blue"]} | {g["blue_TP_missing_from_control"]} | {g["blue_FP_missing_from_control"]} |')
lines += ['', '没有预算拒绝时，颜色方案的新增像素是统一概率加分的子集；它确实排除了一部分FP，也排除了部分TP。存在预算拒绝的帧单列，因为统一加分可能整帧拒绝、蓝色较少提议却通过；不能将该非线性回退产生的全部差异归因于颜色分类能力。两方案均未通过，不以相对较好宣称完成改进。', '',
    '## 网络漏检与端部外观诊断', '',
    '70帧共有42534个原光纤FN，42473个在A0中被背景获胜、61个被结石获胜。28339个FN的光纤概率低于0.1（约66.6%）；原FN平均概率0.1045，TP平均0.9769。11020个FN至少一个成员能预测光纤，31514个被两成员同时拒绝。', '',
    'FN的蓝色分数仍平均0.9416（TP为0.9679），饱和度/亮度平均0.7450/0.5595，低于TP的0.9721/0.8434；原FP的蓝色分数仍达0.7862。可见蓝色有辅助信息，但不能独立区分真实光纤、边缘光晕和反光。这些关联支持继续检查网络/输入尺度/监督问题，尚不能单独证明光度增强不足、失焦或尖端外观是主因。', '',
    '参考PCA轴向代理有69/70帧可用，远端代理平均概率0.6199、蓝色分数0.9269；该区域不是解剖尖端标注。像video_2第1209、1224帧，RGB和蓝色分数支持的远端部分在原网络中仍很弱；有限正向加分无法可靠补齐，不能因为几何上应存在就填满。不存在“原高置信像素被A+删掉”的现象，全部静态与视频像素保护已核验。', '',
    '## 中心线覆盖补充', '',
    '为补充中心线完整性，另计算参考可见光纤掩膜的细化中心线覆盖代理，保留全部可见组件，忽略未知255及3像素邻域/图像边缘。它是事后补充指标，不改变冻结验收门槛，不是物理轴线、尖端或跨帧身份真值。3项细化自检通过。', '',
    '| 参考 | 方案 | 参考中心线像素 | 覆盖率 | 最长漏检片段像素 |', '|---|---|---:|---:|---:|']
for scope, data in center['sets'].items():
    for name, q in data['models'].items():
        lines.append(f'| {scope} | {labels[name]} | {q["reference_centerline_pixels"]} | {fmt(q["coverage"])} | {q["longest_missed_fragment_pixels"]} |')
lines += ['', '## 两个完整视频性能', '',
    '每方案、每视频独立新进程：6个进程、12帧预热后从第0帧顺序解码，batch1、FP32、全部3网。每方案722+1972=2694帧，三方案合计8082帧，全部计入统计，未跳过失败帧。所有原始classwise输出与历史逐像素一致，70个已审帧的静态/全片结果完全一致。', '',
    '| 方案 | 视频 | 帧数 | 管线P50 ms | P95 ms | 最大ms | CPU补充分支P50 ms | 含导出离线fps |', '|---|---|---:|---:|---:|---:|---:|---:|']
for name, videos in s['full_video'].items():
    for vid, q in videos.items():
        t = q['timings']['pipeline_ms']; r = q['timings']['refinement_ms']
        lines.append(f'| {labels[name]} | {vid} | {q["frames"]} | {t["p50"]:.3f} | {t["p95"]:.3f} | {t["max"]:.3f} | {r["p50"]:.3f} | {q["offline_fps_including_all_exports"]:.3f} |')
lines += ['', '管线包括预处理、三网、原图概率映射、融合、既有掩膜/概率CPU传输；候选额外传输A0完整三类原图概率并计算相同FP32 log-margin，再执行CPU分支。颜色分支实测中位12.75/14.26ms、p95约15.60/15.82ms；没有额外神经网络，CUDA峰值分配均319.82MiB。各进程的系统/网络耗时有波动，不能直接相减两个p95当作分支净成本。', '',
    '全周期另含解码、H.264渲染、RGB/参考身份核对、PNG和日志导出及写入收尾；离线fps不等于实时显示/队列验收。全部方案p95超过各源视频约33ms预算；蓝色A+video_2最大245.234ms也保留报告。本轮不宣称实时达标。', '',
    '| 视频 | 方案 | 改变掩膜帧数 | 回退帧数 | 新增像素（无全片真值） |', '|---|---|---:|---:|---:|']
for vid, q in audit['full_video_changes'].items():
    n = 722 if vid == 'video_1' else 1972
    for name in config['schemes'][1:]:
        changed = q['changed_frames'][name]
        lines.append(f'| {vid} | {labels[name]} | {changed} | {n-changed} | {q["added_pixels"][name]} |')
lines += ['', '这里没有几何拟合，拟合接受率为不适用；报告的是修正应用与回退。蓝色A+总改变2642/2694帧，52帧回退（49帧预算拒绝，3帧无支持）；全片新增像素只是预测变化，不当作TP或精度。低置信外观的权重为软值，回退时完整返回原掩膜，原结石和光纤像素均未删除。', '',
    '## 检查、产物与后续', '',
    '4项A+针对性测试及4项D0回归测试通过，覆盖白色/暗部中性、断片原光纤与结石保护、无远处填充、预算/NaN/无锚点回退；47项独立核验通过，逐项重算202张像素混淆矩阵与增量、41张新增FP32概率、全部8082帧身份/保护/时分位数、6个进程、静态/全片一致性与视频导出。70帧入集另有40项核验。无新训练、默认采用、参数扫描或结石时序启动。', '',
    '- 配置：configs/fiber_blue_aplus_20261029.json；运行及源码快照：本目录preflight.json、source_snapshot.zip、runtime.json。',
    '- 原图/基线/纯概率/蓝色四栏完整对比：video_1_comparison.mp4、video_2_comparison.mp4；分别722/1972帧，H.264，1600×434。',
    '- 已知失败与新增帧：video_1_diagnostic_board.jpg、video_2_diagnostic_board.jpg，显示RGB、人工参考、网络概率、蓝色分数、原掩膜与A+新增TP/FP。',
    '- 完整指标：video_native/summary.json、image_native/summary.json、static_gates.json、engineering_checks.json、centerline_coverage_proxy.json、color_control_attribution.json。', '',
    '本轮停止扩大纯颜色加分，不调整颜色阈值或预算、不将失败升级为默认。最低成本下一步恢复已准备的两个A0单光纤成员两网对照：先另存更新协议，覆盖当前70/132与每成员完整两片的独立计时；只比较两个已有成员，不训练、不搜索更多成员。它用于核验减少网络是否保留质量并降低已发现的三网尾延迟，不预先承诺能解决低概率漏检。成员均未达标时才另行设计轻量精修/训练侧诊断；任何新训练仍需独立数据、预算、验证/回滚计划和用户批准。尖端专项参考继续待补。']
(out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
atomic_json(out / 'report_identity.json', {'source_sha256': sha256(Path(__file__)), 'report_sha256': sha256(out / 'report.md'),
    'attribution_sha256': sha256(out / 'color_control_attribution.json')})
print({'report': str(out / 'report.md'), 'adopted': False})
