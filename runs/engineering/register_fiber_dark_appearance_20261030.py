"""Register user-observed dark distal-fiber omission, with reference evidence limits."""
import json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from PIL import Image
from tools.fiber_reference_workspace import read,sha256
from ureter_semis.experiment import atomic_json
OUT=ROOT/'runs/a1_fiber_dark_appearance_diagnosis_20261030'
DOC='docs/progress/2026-10-30-fiber-dark-appearance-observation.md'
ADDENDUM='docs/plans/fiber_dark_appearance_addendum_20261030.md'


def main():
    s=read(OUT/'summary.json');identity=read(OUT/'identity.json');checks={}
    checks['all_frozen_sources_unchanged']=all(sha256(ROOT/k)==v for k,v in identity['frozen_sources'].items())
    checks['all_reference_bytes_unchanged']=all(sha256(k)==v for k,v in identity['reference_files'].items())
    checks['all_cached_probabilities_unchanged']=all(sha256(k)==v for k,v in identity['cache_files'].items())
    checks['diagnostic_outputs_identity']=all(sha256(OUT/k)==v for k,v in identity['outputs'].items())
    checks['T2P_config_unchanged']=sha256(ROOT/'configs/fiber_local_evidence_control_20261030.json')==identity['frozen_T2P_config_sha256']
    for scope in ['video_native','image_native']:
        d=read(OUT/f'{scope}.json');old=ROOT/'runs/a1_two_network_experts_20261029'
        checks[scope+'_reference_counts']=d['frames']==len(d['records'])==(70 if scope=='video_native' else 132)
        for model,v in d['models'].items():
            baseline=read(old/f'{scope}/{model}.json')['summary']['metrics']
            for feature in ['HSV_V','Lab_Lstar']:
                bins=v['all'][feature]
                checks[scope+model+feature+'_TPFPFN_exact']=all(sum(x[k] for x in bins)==baseline['fiber_'+k] for k in ['tp','fp','fn'])
                checks[scope+model+feature+'_GTdenominator']=sum(x['gt_pixels'] for x in bins)==baseline['fiber_tp']+baseline['fiber_fn']
    checks['no_new_network_pixel_or_training_experiment']=s['new_model_forwards']==s['new_corrected_masks']==s['training_updates']==0 and not s['tip_truth']
    assert all(checks.values())
    atomic_json(OUT/'engineering_checks.json',{'passed':sum(checks.values()),'checks':checks,'role':'read-only identity/strata recount, not new algorithm tests',
        'auditor_registrar_sha256':sha256(Path(__file__)),'new_model_forwards':0,'new_corrected_masks':0})
    path=ROOT/DOC
    if path.exists():raise FileExistsError(path)
    lines=['# 暗色光纤前端漏分：用户观察与缓存核对\n',
           '2026-10-30。用户报告暗光下光纤头更暗、更深，掩膜排除前端而显得更短。本轮仅核对现有70/132参考和原始概率，新增网络前向0、修正掩膜0、训练0；没有改默认或冻结T2-P配置。\n',
           '## 观察得到支持，但因果和尖端专项真值仍有边界\n',
           'video_2显示第1209帧可直接对照：审核整帧参考包含RGB中可见暗蓝前段，原概率图在该段接近背景，两网原掩膜已明显截短；classwise3也出现同类现象。这个例子支持“暗色可见前段在神经网络阶段漏掉”，而不是T1把已识别尖端删除。第78帧video_1相对完整一并保留，不能只展示严重失败再当总体估计。\n',
           '| 原底座 | 视频低Lab亮度组Recall | 中亮组Recall | 高亮组Recall | 低亮组内部像素Recall |\n|---|---:|---:|---:|---:|']
    for model,v in s['results']['video_native']['models'].items():
        b=v['all']['Lab_Lstar'];lines.append(f"| {model} | {b[0]['recall']:.4f} | {b[1]['recall']:.4f} | {b[2]['recall']:.4f} | {b[0]['interior']['recall']:.4f} |")
    lines+=['\n低亮组共有46105个视频GT光纤像素；内部组排除距语义边界3像素以内区域，仍有漏分。该结果不是独立病例统计，像素和相邻帧相关，不证明变暗是唯一原因，也不能等同尖端Recall。Lab使用OpenCV显示图像亮度编码三分组；不是实际照明或辐照度。HSV V分组同时保留，因为深蓝的V可能仍高，不能仅用最大颜色通道描述视觉明暗。\n',
            '图像132张低Lab亮度组仅1211个GT像素，model1/classwise3 Recall为0.5813/0.4740；样本与标注完整性限制更大，不能将该小组外推临床泛化能力。所有逐帧/逐病例TP、FP、FN、原概率与内部统计都保留。\n',
            '第1209帧model1低Lab亮度组：GT2776、TP150、FN2626，2521个FN的原P_fiber<0.1；既定T2-P作用域仅覆盖105个该组FN。原T1当前候选未通过双边缘检查，没有可用当前几何观测。仅扩大平滑/颜色加分或强行延长模板，都不能从这些证据保证找回真实可见头部。\n',
            '参考主轴投影覆盖缺口仅作描述性记录，PCA轴两端没有真实入口/尖端身份，参考还可能有不连通片区；不能把其投影长度差当物理尖端误差或真实缺失长度。整帧语义参考证明可见像素遗漏，但不足以计算独立尖端区域指标。\n',
            '## 实施顺序调整\n',
            '暗色前端低概率问题前置，下一步先制定专门的当前图像/亮度与色度/杆头外观受控协议，明确最小端部参考、误检约束、全片测试和停止规则。原T2-P保持未实现状态，保留作局部边界对照，不作为完整暗尖端恢复方案；不放宽其P阈值、半径或TP保护来追求表面恢复。\n',
            '研究时须区分杆体与暗头外观，几何/历史可提出候选，但当前RGB必须支持可见像素；不能把内部亮暗过渡自动当物理末端，或把历史轮廓填成真实头部。推理光度变换也可能扩大组织误检，须先冻结对照，当前未执行。训练增强若需启动，另行明确数据、初始化、真实监督增强路径、预算与验证并等待批准。\n',
            '沿用既有12帧专项入口，未确认前不给模型预测生成尖端真值；不重新要求全部训练数据审核。视频/派生物只验证、stone_new排除、原521/132/144/29与70标签不变，未读训练/144保留test像素。70是重复开发参考，42/120抽样未完成，不外推全片精度。\n',
            f"{sum(checks.values())}项只读身份及分组重算核验通过，无新修正算法测试。\n",
            '- [RGB/参考/原掩膜/概率/FN对照](../../runs/a1_fiber_dark_appearance_diagnosis_20261030/dark_appearance_reference_board.jpg)\n- [视频70逐帧与病例统计](../../runs/a1_fiber_dark_appearance_diagnosis_20261030/video_native.json)、[132图像统计](../../runs/a1_fiber_dark_appearance_diagnosis_20261030/image_native.json)、[身份与分组核验](../../runs/a1_fiber_dark_appearance_diagnosis_20261030/engineering_checks.json)\n- [计划补充](../plans/fiber_dark_appearance_addendum_20261030.md)\n']
    path.write_text('\n'.join(lines),encoding='utf-8')
    Image.open(OUT/'dark_appearance_reference_board.jpg').crop((0,510,2000,1020)).save(OUT/'video_2_frame1209_dark_comparison.jpg',quality=95)
    main=ROOT/'docs/plans/segmentation_temporal_occlusion_plan.md';text=main.read_text(encoding='utf-8');assert '版本：v2.19' in text;lines=[]
    for line in text.replace('版本：v2.19','版本：v2.20',1).splitlines():
        if line.startswith('| S2 |'):line='| S2 | 暗色前端截短观察已在1209帧原概率/审核参考中核对，70/132亮度分组完成；原T2-P局部协议保留未实施 | 优先制定暗端低概率、杆头外观与当前图像证据专用协议；T2-P留作局部边界对照，不能恢复大部分P<0.1暗段。不调原门槛、不训练或默认采用/S3 |'
        elif line.startswith('- [ ] **当前下一步：T2-P有限像素对照实现**'):
            line=line.replace('**当前下一步：T2-P有限像素对照实现**','**T2-P有限像素对照实现（保留为局部边界对照）**',1)+'\n- [x] **暗色前端截短观察核对**：70/132原概率缓存只读分组，新增前向0；1209帧低亮GT2776/FN2626，其中2521低于P0.1，原局部E只覆盖105。整帧参考支持暗段遗漏，无独立尖端真值不报尖端Recall。详见[证据](../progress/2026-10-30-fiber-dark-appearance-observation.md)。\n- [ ] **当前下一步：暗端低概率与杆头外观专项协议**：先明确当前图像证据、最小端部参考、亮度/色度响应对照和搜索范围失效，再冻结有限方法、完整两片性能/精度统计和停止条件。原T2-P不作为完整解法，配置/门槛不动；不自动光度变换、填头、训练、默认替换或结石S3。见[计划补充](fiber_dark_appearance_addendum_20261030.md)。'
        lines.append(line)
    addition='2026-10-30 暗色前端观察更新（覆盖以上历史下一步）：用户指出暗光深色光纤头常漏分并使掩膜变短。已只读核对70/132原概率，1209帧审核参考包含的暗蓝前段在原模型即低概率；model1低Lab亮度组FN2626中2521的P<0.1，T2-P范围只覆盖105。视频低亮/高亮组Recall为model1 0.6212/0.9987、classwise0.5601/0.9994；是全光纤亮度分组，非尖端Recall/照明因果。下一步优先制定暗端低概率与杆头不同外观专用协议，T2-P保留未实现局部对照，不放宽其阈值或硬延长；无新前向、修正、训练、默认替换或S3。'
    main.write_text('\n'.join(lines)+'\n\n'+addition+'[本次记录](../progress/2026-10-30-fiber-dark-appearance-observation.md)。\n',encoding='utf-8')
    p=ROOT/'README.md';text=p.read_text(encoding='utf-8');assert text.count('\n## 实现说明')==1
    p.write_text(text.replace('\n## 实现说明','\n'+addition+'[证据与计划补充](docs/progress/2026-10-30-fiber-dark-appearance-observation.md)。\n\n## 实现说明',1),encoding='utf-8')
    p=ROOT/'annotations/data_review_status_20261027.json';status=read(p)
    status.update(fiber_dark_appearance_user_observation='Dark distal fiber appears excluded and semantic mask shortens in darker scenes',
        fiber_dark_appearance_readonly_executed=True,fiber_dark_appearance_report=DOC,fiber_dark_appearance_summary=str(OUT/'summary.json'),
        fiber_dark_appearance_new_forwards=0,fiber_dark_appearance_tip_truth_available=False,
        fiber_dark_appearance_plan_addendum=ADDENDUM,fiber_local_evidence_priority='retained_local_boundary_control_dark_distal_protocol_first',
        current_execution_checks=str(OUT/'engineering_checks.json'),latest_instruction='Investigate user-observed dark distal-fiber omission and shortening',
        latest_instruction_effect='Read-only70/132 brightness and original-probability verification; dark-frame1209 omission supported. T2P frozen untouched/unimplemented; no new inference/filter/training/default',
        execution_order_confirmation='Next prepare separate dark-distal low-probability and shaft/front appearance protocol; T2P retained secondary control, no automatic training or correction',
        remaining_prerequisites=[
            'Next freeze dark-distal-specific current-image/brightness-chroma/shaft-front appearance and search-evidence protocol with original controls, full videos and failure gates',
            'Independent12tip references pending; whole semantic masks/darkness bins/PCA projected extents are not tip truth or physical length error',
            'T2P retained unimplemented local control, cannot recover most Pbelow0.1 dark pixels; do not expand frozen thresholds/radius or fill predicted head',
            'No automatic photometric transform/training/default/stoneS3; videos validation only, stone_new excluded,521train/144test pixels excluded'
        ])
    atomic_json(p,status)
    p=ROOT/'annotations/video_evaluation_registry.json';registry=read(p)
    registry.update(latest_plan_addendum=ADDENDUM,latest_plan_addendum_sha256=sha256(ROOT/ADDENDUM),
        current_next_step='prepare_dark_distal_appearance_protocol_before_local_T2P',latest_darkness_read_only_result=str(OUT/'summary.json'),
        latest_darkness_read_only_result_sha256=sha256(OUT/'summary.json'),latest_darkness_diagnostic_new_forwards=0)
    atomic_json(p,registry)
    for p in [ROOT/DOC,ROOT/ADDENDUM]:
        assert all((p.parent/link).resolve().exists() for link in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')))
    assert sha256(ROOT/'configs/fiber_local_evidence_control_20261030.json')==identity['frozen_T2P_config_sha256']
    atomic_json(OUT/'registration.json',{'registrar_sha256':sha256(Path(__file__)),'progress_sha256':sha256(ROOT/DOC),'addendum_sha256':sha256(ROOT/ADDENDUM),
        'main_plan':'v2.20','new_forwards':0,'new_corrected_masks':0,'training_updates':0,'files':{p.name:sha256(p) for p in OUT.iterdir() if p.is_file()}})
    print(json.dumps({'registered':True,'main_plan':'v2.20','readonly_checks':sum(checks.values()),'T2P_unchanged':True,'new_forwards':0}))


if __name__=='__main__':main()
