"""Publish read-only T1 attribution and freeze unexecuted T2-P protocol."""
import json,re,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.fiber_reference_workspace import read,sha256
from ureter_semis.experiment import atomic_json
OUT=ROOT/'runs/a1_fiber_t1_failure_diagnosis_20261030'
CFG='configs/fiber_local_evidence_control_20261030.json'
DOC='docs/plans/fiber_local_evidence_control_20261030.md'
PROGRESS='docs/progress/2026-10-30-fiber-t1-failure-diagnosis.md'
PROTOCOL=ROOT/'runs/protocols/fiber_local_evidence_control_20261030_v1'


def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])+'\n'


def document_once(path,text):
    if path.exists():
        assert path.read_text(encoding='utf-8')==text,'Existing document differs; do not overwrite'
    else:path.write_text(text,encoding='utf-8')


def main():
    summary=read(OUT/'summary.json');forensic=read(OUT/'forensic_summary.json');audit=read(OUT/'engineering_checks.json');plan=read(ROOT/CFG)
    assert audit['passed']==34 and summary['actual_new_diagnostic_forward_calls']==5 and summary['new_candidate_masks']==0
    assert not(PROTOCOL/'identity.json').exists() and not(PROTOCOL/'engineering_checks.json').exists()
    assert not(ROOT/plan['planned_output']).exists() and all(not(ROOT/p).exists() for p in plan['planned_files'])
    old=ROOT/'runs/a1_fiber_temporal_observation_search_20261030';pre=read(old/'preflight.json');cfg=pre['config'];regpath=ROOT/'annotations/video_evaluation_registry.json';registry=read(regpath)
    retained_actual={k:v for k,v in registry.items() if k.startswith('latest_state_control_') or k.startswith('latest_inference_control')}
    report=['# T1只读失效归因与T2-P实验协议\n',
            '2026-10-30。**已完成202张参考的缓存概率/掩膜归因、四臂完整视频日志分析，以及5个既定视频帧的原始模型概率补导。没有执行任何像素修正、调整T1阈值、训练或替换默认。T2-P三臂协议已冻结，尚未实现。**\n',
            '## 范围与证据等级\n',
            '沿用70视频开发参考与132图像开发参考，不访问521活动训练或144保留test图像像素。部分开发图像物理文件仍位于原来源train文件夹，其用途按冻结联合manifest的val清单判断；本轮按真实清单排除训练/保留test，不按文件夹字符串误判。源图像、掩膜、辅助laser_dot、概率、旧输出、权重和代码身份全部核验。原70为重复开发材料，42/120随机抽样未齐，不报告全视频精度。\n',
            '全部202张仅复算已有概率与T1原观测，不新跑候选模型；另对video_2显示311、439、622、643、1266五帧补导同一冻结两网原始概率，每帧原图掩膜和RGB精确匹配T1全片记录，权重前后不变。这5次原模型前向仅用于证据核对，既不是新方法或精度实验，也不报告新的性能结论；五帧均不在70帧参考中，无像素GT。\n',
            '## 1. 无状态不等于网络没识别\n',
            '图像50张含光纤参考的首次失效分解如下。多候选全部失败时保留全部原因组合，不把同帧重复计数。\n']
    report.append(table(['首次状态结果','帧数'],summary['static']['image_native']['positive_stage_counts'].items()))
    report+=['23张宽度族失败图像共有48000个GT光纤像素，原模型已正确覆盖46260个，Recall为0.96375。把这些全部称为神经网络漏检不成立。23个远/近宽度比均>1.3，其倒数均落入原[0.12,1.30]区间；这只是算术敏感性，**没有翻转候选、没有重测接受结果，更不能声称23张均可恢复**。短粗可见片段的PCA轴、FOV入口启发式与方向符号都可能不合适，需独立验证，不能用理想形状裁掉这些原正确像素。\n',
             '3张杆体通过但方向不明，T1按协议不给入口/端部状态；7张首次因可见长度短失败，另有2张混合失败。仅1张无候选种子，即此前445个GT光纤像素的截图样本；原P_fiber极低，几何历史不能跨静态图像为其制造观测。70视频参考的68张有状态、2张双边缘支持不足。\n',
             '## 2. 网络、分类与几何后处理的责任边界\n']
    report.append(table(['参考','GT光纤','TP','FP','FN','FN中P<0.1','FN中0.1≤P<0.9','FP中P≥0.9'],[[scope,*[d['neural_counts'][k] for k in ['gt_fiber_pixels','tp','fp','fn','fn_below01','fn_01_to09','fp_ge09']]] for scope,d in summary['static'].items()]))
    report+=['全光纤FN的66.15%（视频）/67.76%（图像）在原模型中P_fiber<0.1，支持原像素预测不足这一事实；图像FP有70.80%在P_fiber≥0.9，支持高置信身份错误或参考完整性问题需要单列。**这些都是全光纤像素统计，不能当尖端漏分比例或断言光度增强不足。**\n',
             '原fiber-expert argmax已为2、却在原融合掩膜里丢失的GT像素数，两集合均为0；光纤优先映射没有再覆盖这些像素。T1四臂所有掩膜相同，没有任何TP被其状态器或几何后处理删除。以往固定模板/相交等实验确有大量TP损失，历史拒绝结论保留，不能混到当前T1责任统计里。85/523个GT光纤被原类别1保护的冲突仍在，不能把它们当“网络已识别光纤却被类别覆盖”，须查看原fiber-expert三类竞争。\n',
             '## 3. 双边缘拒绝：搜索位置与真实对比度仍须分开\n',
             '311和1266帧所有8个截面种子支持数都≥4，分别只有4/8、3/8双侧对比通过，故不是种子计数不足。在多段截面，一侧最大RGB差低于0.04、另一侧明显较高；这仅说明当前搜索条内证据不对称，不能在没有像素参考时区分真实低对比度、搜索条偏移、视野截断或局部纹理。311的历史候选获得5/8对并被弱更新；1266历史已过期。\n',
             '202张带语义参考的静态诊断进一步在完全相同轴线/截面上投影可见参考区间，只用于归因：\n']
    report.append(table(['参考','单个参考区间且采样界内','参考双侧对比≥0.04','其中参考边界未全落入原搜索条'],[[scope,d['reference_single_intervals'],d['reference_bilateral_contrast_intervals'],d['reference_contrast_intervals_outside_seed_strips']] for scope,d in forensic.items()]))
    report+=['这些是候选×截面的条件计数，不是独立帧样本，也不是边缘检测Recall；参考polygon边界不自动等于真实可见光学边界。它们证明“条内对比不足”不能直接归为“图像无边缘”。种子支持不足截面有视频28、图像8，均非2048点确定性限额造成（没有满4点经限额降到不足4点的截面）。不得据此直接扩大条宽或降低阈值。\n',
             '## 4. 亮斑身份混淆有来源标注佐证\n',
             '图像清单第17张（sample后缀d63c2244ceeb58c5590f17f2）没有光纤语义参考，却有1530像素来源laser_dot辅助标注。原网络预测光纤1381像素，其中1267像素与laser_dot重叠；前沿候选也在laser_dot内。该处原光纤概率中位约0.99959、1381像素中1203个P≥0.9，T1仍可拟合且接受。RGB可见局部亮斑与右缘蓝色细条分离，表明当前候选身份存在明确风险。\n',
             'laser_dot为原来源未审核标注，背景/光纤完整性仍不明；本轮只保留类别冲突证据，不生成新真值、不宣称已确认全部物理像素归属。单一“最多一根”、直线边缘或颜色阈值都不能保证身份。下一轮高置信保护会保留该处大部分预测，不能宣称局部滤波能解决这个显著失败。\n',
             '## 5. 外观与端部证据的实际边界\n',
             '按参考可见光纤的PCA无向轴，把像素投影分成4段，固定轴号朝x正向，仅是可复现编码，**不是入口0/尖端1**。各段概率、Recall、S/V及9个固定S/V分组均保留。视频第1/第4无向四分位的平均P约0.9535/0.6889，V约218.5/168.3，S约250.5/204.8；图像对应P约0.8989/0.9052、V约204.5/199.1。不同来源与位置分布不一致，不能把其中一端直接称尖端，更不能预先认定统一的过曝/变白原因。\n',
             '可见前沿候选在最近语义像素处，视频68个中62个落在光纤类别、3个背景、3个结石；图像12个中3个光纤、6个背景、3个结石。这仅是掩膜接触诊断，不是尖端准确率：像素取整、参考边界误差、可见遮挡边界和真实端部身份均未确定；落入GT光纤也不证明找到尖端。\n',
             '要计算真实尖端Recall/误差，沿用既有12帧专项入口，需独立确认可见尖端像素、端部点/局部方向及不可见/不确定区；可同时记录入口/朝向是否可判断。被遮挡部分不画成可见标签。未确认前仅定性审核，无需重新复核全部训练数据，也不阻止整帧语义开发对照。\n',
             '## 6. 最小下一步及作用域上界\n',
             '已另行制定T2-P三臂：原raw2、固定5点空间概率融合、相同5点的RGB加权融合。只处理原光纤边界带中0.1≤P<0.9的原背景/光纤像素，原石头和高置信像素保护；原概率占75%，邻居证据占25%。RGB相似性控制邻居影响，无蓝色硬裁、统一正向加分、共享颜色模板、几何裁剪或时序填补。T1位置不作为像素选择前提。\n',
             '机器配置在作用域计数前冻结，未按真值搜索参数；本轮只计数E，不执行五点滤波、不输出新掩膜。预采样预算没有拒绝帧：\n']
    report.append(table(['参考','E像素数','E内原TP（风险）','E内FP（机会）','E内FN（机会）'],[[scope,*[d['after_presampling_budget_totals'][k] for k in ['eligible_pixels','eligible_reference_tp_at_risk','eligible_reference_fp','eligible_reference_fn']]] for scope,d in forensic.items()]))
    report+=['机会计数不是恢复结果，尚未施加实际滤波、类别映射和变化面积回退；即使所有可接触错误都能理想修复，也无法覆盖低P背景/高P误检、原结石冲突或全无种子。TP风险反而提示必须使用严格的净TP、粗删除TP和逐帧保护门槛，不能拿Recall换Precision。\n',
             '计划要求202静态新原图复现及三方案各两片完整回放，6独立进程8082正式帧，即使静态或耗时失败也做完；保留原29、70/132各分组、逐帧错误、边界/中心线和全部回退。原门槛与classwise参考不变，新增TP损失上限和同轮raw2不退化检查，参数不扫描。尚未实现或运行，无新性能/精度结论。\n',
             '优先该有限方案的理由是参数少、当前RGB证据与空间平均可独立对照，避免先依赖已暴露失效的几何方向。它未被预设有效，失败即结束本轮局部路线；错误身份、强背景漏分、方向与监督改进必须另行诊断，训练仍需独立批准，视频/派生物与stone_new不训练。\n',
             '## 交付、核验与状态\n',
             '34项只读重算/身份核验通过；它们不是新像素算法的单元测试或性能验证。审计强化了旧门槛与原成员实验preflight哈希的核对，初版核验记录与源码保留；没有改模型、参数、标签或T1结果。计划及身份归档单独保存，默认A0、原联合划分、权重及全部历史输出保留。\n',
             '- [T2-P实施协议](../../docs/plans/fiber_local_evidence_control_20261030.md)、[机器配置](../../configs/fiber_local_evidence_control_20261030.json)\n- [静态归因板](static_failure_attribution_board.jpg)、[视频双边缘失效板](video_edge_failure_attribution_board.jpg)\n- [视频70逐帧诊断](video_native.json)、[图像132逐图诊断](image_native.json)、[条件统计与作用域上界](forensic_summary.json)\n- [5帧原始概率与状态身份](video_snapshots.json)、[34项核验](engineering_checks.json)、[输入/源码/输出身份](identity.json)\n']
    document_once(OUT/'report.md','\n'.join(report))
    doc=ROOT/PROGRESS
    document_once(doc,'# T1失效归因完成，T2-P协议已制定\n\n2026-10-30。**本轮是只读诊断和下一轮协议，不是像素修正实验。** 202张缓存复算、四臂完整视频日志与5帧原模型概率补导完成，34项核验通过。T1源码/参数、标签、权重及默认不变，没有训练。\n\n'
        '主要发现：\n\n'
        '- 图像50张含光纤参考中23张首先被宽度族拒绝，但这23张原模型Recall为0.96375；3张杆体合格但方向不明，只有1张无种子。反转宽度比可改变数值约束结论，但没有验证应反转轴线或可恢复23张。\n'
        '- 原全光纤FN的66.15%/67.76%（视频/图像）P<0.1；原fiber-expert已argmax为光纤却在原掩膜消失的GT像素为0。T1没有删任何分割像素，不能把这些FN归因于T1。尖端独立真值未补，以上不是尖端统计。\n'
        '- video_2的311/1266帧均有充分种子点，却只有4/8、3/8双边缘支持；一侧搜索条内对比不足，仍需区分真实低对比、窗口偏移与视野截断。参考条件测量发现部分真实语义边界在原搜索条外，不能一概归为图像无边缘。\n'
        '- 被接受的亮斑预测1381像素中1267与来源laser_dot区域重叠，前沿候选也在其中；支持身份混淆风险。辅助标注仍未审核，不能升级为新GT。图像FP中70.80% P≥0.9，低置信修正不能解决大部分高置信错误。\n'
        '- 无向参考轴分段仅作S/V、概率和Recall描述，不把某一端当解剖尖端。缺少独立端部参考，物理尖端/方向/落点指标仍不可计算。\n\n'
        '已制定独立T2-P三臂：raw2、固定五点空间融合、相同五点RGB加权融合；仅在0.1≤P<0.9的双侧边界带处理原0/2类别，原石头/高置信保护，75%原概率保留。无蓝色硬阈值、共享颜色模板、几何掩膜相交或历史填充。\n\n'
        '协议冻结后只读计数显示，作用域可接触视频11721个FN/7053个FP、图像1742个FN/16996个FP，同时覆盖27254/4398个原TP，存在误删风险。**这是机会/风险计数，不是实际改善或理想修复精度。** 本轮没有执行滤波或生成新分割。\n\n'
        '下一步按冻结协议实现有限三臂，202静态新复现，各臂完整722+1972帧，6独立进程8082帧；静态/成本失败仍做全片，原质量门槛保留并增加TP损失上限。参数不扫描、结果不自动采用；不自动训练或启动结石S3。70帧重复开发与42/120不完整样本不外推全片准确率，原521/132/144/29及70标签不变。\n\n'
        '- [完整归因报告](../../runs/a1_fiber_t1_failure_diagnosis_20261030/report.md)、[静态归因图](../../runs/a1_fiber_t1_failure_diagnosis_20261030/static_failure_attribution_board.jpg)、[视频边缘失败图](../../runs/a1_fiber_t1_failure_diagnosis_20261030/video_edge_failure_attribution_board.jpg)\n'
        '- [T2-P协议](../plans/fiber_local_evidence_control_20261030.md)、[配置](../../configs/fiber_local_evidence_control_20261030.json)、[34项核验](../../runs/a1_fiber_t1_failure_diagnosis_20261030/engineering_checks.json)\n')
    # Freeze the unexecuted protocol plus the evidence on which it is based.
    files={p:sha256(ROOT/p) for p in [CFG,DOC,PROGRESS,'configs/fiber_t1_failure_diagnosis_20261030.json',
        'configs/compare_two_network_experts_20261029_v2.json','tools/fiber_fast_observation.py','tools/fiber_evidence_tracker.py','tools/member_experts.py',
        'runs/a1_fiber_t1_failure_diagnosis_20261030/report.md','runs/a1_fiber_t1_failure_diagnosis_20261030/summary.json',
        'runs/a1_fiber_t1_failure_diagnosis_20261030/forensic_summary.json','runs/a1_fiber_t1_failure_diagnosis_20261030/engineering_checks.json']}
    PROTOCOL.mkdir(parents=True,exist_ok=True)
    if (PROTOCOL/'protocol_sources.zip').exists():
        with zipfile.ZipFile(PROTOCOL/'protocol_sources.zip') as z:
            assert set(z.namelist())==set(files) and all(z.read(p)==(ROOT/p).read_bytes() for p in files),'Existing partial archive differs'
    else:
        with zipfile.ZipFile(PROTOCOL/'protocol_sources.zip','w',zipfile.ZIP_DEFLATED) as z:
            for p in files:z.write(ROOT/p,p)
    atomic_json(PROTOCOL/'identity.json',{'status':'T2P_protocol_only_not_implemented_not_executed','files':files,'weights':pre['checkpoints'],
        'manifests':{cfg['data'][k]:sha256(ROOT/cfg['data'][k]) for k in ['video_manifest','image_manifest']},
        'frozen_base_preflight':str(old/'preflight.json'),'frozen_base_preflight_sha256':sha256(old/'preflight.json'),
        'frozen_base_source_archive_sha256':sha256(old/'source_snapshot.zip'),'archive_sha256':sha256(PROTOCOL/'protocol_sources.zip'),
        'planned_processes':6,'planned_full_frames':8082,'T2P_actual_forwards':0,'diagnostic_original_model_snapshot_forwards':5,
        'training_updates':0,'training_pixels_read':0,'reserved_test_pixels_read':0})
    checks={
        'configuration_and_finite_arms':plan['arms']==['raw2','spatial5','rgb_edge5'],
        'bounded5point_sampling':len(plan['filter']['neighbor_offsets'])==5,
        'neural_primary_fixed':plan['filter']['blend_alpha']==.25,
        'all_processes_full_two_videos':plan['execution']['expected_processes']==6 and plan['execution']['expected_full_forward_frames']==8082 and plan['execution']['all_arms_complete_even_static_or_latency_failure'],
        'prospective_scope_identity':all(d['eligible_scope_plan_sha256_before_calculation']==sha256(ROOT/CFG) for d in forensic.values()),
        'unchanged_T1_sources':all(sha256(ROOT/p)==v for p,v in pre['sources'].items()),
        'implementation_and_output_absent':all(not(ROOT/p).exists() for p in plan['planned_files']) and not(ROOT/plan['planned_output']).exists(),
        'frozen_files_match':all(sha256(ROOT/p)==v for p,v in files.items()),
        'unmodified_model_identity':read(OUT/'video_snapshots.json')['model_state_sha256']==read(old/'static_model_identity.json')['model_state_sha256']}
    for p in [ROOT/DOC,ROOT/PROGRESS,OUT/'report.md']:
        checks[p.name+'_links']=all((p.parent/link).resolve().exists() for link in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')))
    with zipfile.ZipFile(PROTOCOL/'protocol_sources.zip') as z:checks['archive_roundtrip']=all(z.read(p)==(ROOT/p).read_bytes() for p in files)
    assert all(checks.values()),checks
    atomic_json(PROTOCOL/'engineering_checks.json',{'passed':sum(checks.values()),'checks':checks,'role':'protocol consistency, not future algorithm tests'})
    p=ROOT/'docs/plans/segmentation_temporal_occlusion_plan.md';text=p.read_text(encoding='utf-8');assert '版本：v2.18'in text;lines=[]
    for line in text.replace('版本：v2.18','版本：v2.19',1).splitlines():
        if line.startswith('| S2 |'):line='| S2 | T1失效只读归因完成：202缓存+5原模型快照、34核验；T2-P局部当前证据三臂协议已冻结，未实现 | 下一步有限三臂像素对照，202静态及6独立进程8082完整帧；原门槛+TP保护，明确只覆盖低置信边界，不能解决高置信身份/强背景漏检。不训练或自动默认采用/S3 |'
        elif line.startswith('- [ ] **当前下一步：T1失效归因及独立像素修正协议**'):
            line='- [x] **T1失效归因及独立像素修正协议**：202缓存、四臂全片日志及5帧原模型快照核对完成；23宽度拒绝与网络漏检分开，亮斑源laser_dot冲突、边缘窗口/对比度限制及无尖端真值明确。34项核验通过；T2-P配置与协议冻结，尚未实现。详见[归因结果](../progress/2026-10-30-fiber-t1-failure-diagnosis.md)。\n- [ ] **当前下一步：T2-P有限像素对照实现**：独立新后处理比较raw2、spatial5、rgb_edge5；固定当前边界不确定区、75%原概率、原石头/高置信保护及整帧预算回退。先机制测试与202静态，再三臂各完整2694帧、6进程8082帧；静态/成本失败不免测。不调参数或原门槛，粗删除TP和逐帧损失另受限，失败保留。无尖端真值不称尖端更准；不训练、默认替换、额外几何修正或进入S3。'
        lines.append(line)
    addition='2026-10-30 T1归因与T2-P协议更新（覆盖以上历史下一步）：只读202缓存+5既定原模型概率快照完成，34项核验；23张宽度拒绝组原神经Recall0.96375，不能归为网络全漏；全FN约2/3概率<0.1，亮斑与未审核laser_dot标注重叠。T1不删像素，真实尖端原因仍无独立参考。已冻结三臂局部五点空间/RGB概率融合协议，仅低置信边界、75%原概率与保护/预算回退，尚未实现或执行；作用域错误计数仅机会，非改善。下一步机制测试、202静态和三臂完整两片6进程8082帧，原门槛/TP保护不放宽；无训练、默认替换或S3。'
    p.write_text('\n'.join(lines)+'\n\n'+addition+'[归因与协议](../progress/2026-10-30-fiber-t1-failure-diagnosis.md)。\n',encoding='utf-8')
    p=ROOT/'README.md';text=p.read_text(encoding='utf-8');assert text.count('\n## 实现说明')==1
    p.write_text(text.replace('\n## 实现说明','\n'+addition+'[完整记录](docs/progress/2026-10-30-fiber-t1-failure-diagnosis.md)。\n\n## 实现说明',1),encoding='utf-8')
    p=ROOT/'annotations/data_review_status_20261027.json';status=read(p)
    status.update(fiber_t1_failure_diagnosis_executed=True,fiber_t1_failure_diagnosis_report=PROGRESS,
        fiber_t1_failure_diagnosis_summary=str(OUT/'summary.json'),fiber_t1_failure_diagnosis_engineering_checks=34,
        fiber_t1_failure_diagnosis_original_snapshot_forwards=5,fiber_t1_failure_diagnosis_new_pixel_methods=0,
        fiber_local_evidence_plan_prepared=True,fiber_local_evidence_implemented=False,fiber_local_evidence_executed=False,fiber_local_evidence_adopted=False,
        fiber_local_evidence_plan=DOC,fiber_local_evidence_config=CFG,fiber_local_evidence_plan_identity=str(PROTOCOL/'identity.json'),
        current_execution_checks=str(OUT/'engineering_checks.json'),current_plan_checks=str(PROTOCOL/'engineering_checks.json'),
        last_executed_diagnostic_config='configs/fiber_t1_failure_diagnosis_20261030.json',latest_instruction='Continue next step: read-only T1 failure attribution and separate pixel protocol',
        latest_instruction_effect='Completed202cached references+5original-model diagnostic snapshots,34checks. T2P protocol frozen only, no pixel filter executed/new masks/training/default/S3',
        execution_order_confirmation='Next implement frozen finite3arm T2P and test202static+6full-video processes8082frames; no parameter search or automatic adoption',
        remaining_prerequisites=[
            'Next T2P implementation/tests; then202static original model and3arms x2694full frames in6fresh processes, including failed candidates',
            'T2P only low-confidence boundary evidence; image high-confidenceFP/strong-backgroundFN and static no-seed445 case not solved',
            'All unchanged original29/classwise/raw2 quality gates plus grossTP/individual-frame TP limits; preserve failures, no tuning or automatic geometry repair',
            'No independent tip GT;70repeated-development refs and42of120 incomplete sample cannot establish whole-video accuracy',
            'No training/default/stoneS3; model1 unadopted, video derivatives validation only, stone_new excluded,521train/144test pixels excluded'
        ])
    atomic_json(p,status)
    registry.update(latest_plan_config=CFG,latest_plan_document=DOC,latest_plan_identity=str(PROTOCOL/'identity.json'),latest_plan_executed=False,
                    latest_read_only_diagnostic_config='configs/fiber_t1_failure_diagnosis_20261030.json',latest_read_only_diagnostic_result=str(OUT/'summary.json'),
                    latest_read_only_diagnostic_result_sha256=sha256(OUT/'summary.json'),latest_read_only_diagnostic_new_pixel_methods=0)
    assert {k:v for k,v in registry.items() if k.startswith('latest_state_control_') or k.startswith('latest_inference_control')}==retained_actual
    atomic_json(regpath,registry)
    final_files=[ROOT/PROGRESS,ROOT/DOC,ROOT/CFG,OUT/'report.md',OUT/'forensic_summary.json',OUT/'engineering_checks.json',OUT/'static_failure_attribution_board.jpg',OUT/'video_edge_failure_attribution_board.jpg']
    atomic_json(OUT/'report_artifacts.json',{'registrar_sha256':sha256(Path(__file__)),'files':{str(p):sha256(p) for p in final_files},
                                         'protocol_identity_sha256':sha256(PROTOCOL/'identity.json'),'main_plan':'v2.19','new_pixel_method_executed':False,
                                         'actual_original_model_snapshot_forwards':5,'training_updates':0})
    print(json.dumps({'status':'T1_attribution_complete_T2P_plan_only','main_plan':'v2.19','read_only_checks':34,'protocol_checks':sum(checks.values()),'new_pixel_masks':0}))


if __name__=='__main__':main()
