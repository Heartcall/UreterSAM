"""Create diagnostic boards and explicit feasibility-only report after benchmarks."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from tools.fiber_reference_workspace import read,sha256
from ureter_semis.experiment import atomic_json
OUT=ROOT/'runs/a1_fiber_refinement_feasibility_20261030'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])+'\n'
def fmt(v):return '不可计算' if v is None else f'{v:.6f}' if isinstance(v,float) else str(v)
def overlay(rgb,mask):
    a=rgb.copy()
    for c,color in [(1,[40,255,80]),(2,[30,160,255]),(255,[160,160,160])]:
        region=mask==c;a[region]=(a[region]*.5+np.array(color)*.5).astype(np.uint8)
    return a
def main():
    pre=read(OUT/'preflight.json');cfg=pre['config'];s=read(OUT/'summary.json');audit=read(OUT/'engineering_checks.json');previous=ROOT/cfg['reference_run']
    data={scope:read(OUT/(scope+'.json')) for scope in ['video_native','image_native']};spatial=read(OUT/'error_spatial_diagnosis.json');selection={};font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
    for scope,key in [('image_native','image_manifest'),('video_native','video_manifest')]:
        rows=read(ROOT/cfg[key])['val'];frames=data[scope]['profiles']['hard2']['frames'];errors=data[scope]['errors']['frames'];m1=read(previous/f'{scope}/member1_2net.json')
        picks=[]
        rules=[('largest_outside_FN',lambda i:frames[i]['regions']['fn']['outside']),('largest_total_FP',lambda i:frames[i]['regions']['fp']['total']),('largest_lost_TP',lambda i:frames[i]['regions']['lost_tp_vs_classwise']['total']),('largest_stone_blocked_fiber',lambda i:frames[i]['regions']['gt_fiber_blocked_by_stone']['total'])]
        for label,score in rules:
            i=max(range(len(rows)),key=score)
            if i not in [x[0] for x in picks]:picks.append((i,label))
        selection[scope]=[{'sample_id':rows[i]['sample_id'],'selection':label,'native_shape':list(np.array(Image.open(rows[i]['mask'])).shape)} for i,label in picks]
        canvas=Image.new('RGB',(1440,36+len(picks)*292),(15,15,15));draw=ImageDraw.Draw(canvas)
        draw.text((6,8),'Reference: stone green / fiber blue / unknown gray. Yellow: automatic ROI, not refined masks. P map: 0 dark blue -> 1 red.',font=font,fill='white')
        for j,(i,label) in enumerate(picks):
            rgb=np.array(Image.open(rows[i]['image']).convert('RGB'));target=np.array(Image.open(rows[i]['mask']));coarse=np.array(Image.open(previous/f'{scope}/member1_2net/{i:04d}.png'))
            with np.load(m1['records'][i]['probability_reference']) as cache:p=cache['a0_member1_native'][2]
            heat=cv2.cvtColor(cv2.applyColorMap(np.round(p*255).astype(np.uint8),cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB)
            boxes=[]
            for profile in cfg['profiles']:
                a=overlay(rgb,coarse)
                for x0,y0,x1,y1 in data[scope]['profiles'][profile]['frames'][i]['boxes']:cv2.rectangle(a,(x0,y0),(x1-1,y1-1),(255,255,0),2)
                boxes.append(a)
            images=[rgb,overlay(rgb,target),heat,overlay(rgb,coarse),*boxes]
            labels=['Input','Reference','P_fiber model1','Coarse (unchanged)','hard2 ROI only','soft02_2 ROI only']
            top=36+j*292
            for c,(a,title) in enumerate(zip(images,labels)):
                draw.text((c*240+3,top+3),title,font=font,fill='white');canvas.paste(Image.fromarray(a).resize((240,240)),(c*240,top+23))
            caption=(f"{rows[i]['video_id']} frame {rows[i]['frame_index']+1}" if rows[i].get('video_id') else rows[i]['sample_id'])+' : '+label
            draw.text((5,top+268),caption,font=font,fill='white')
        canvas.save(OUT/f'{scope}_diagnostic_board.jpg',quality=94)
    atomic_json(OUT/'diagnostic_board_selection.json',{'role':'largest failure counts, not success selection or metric subset','frames':selection})
    report=['# 光纤专用局部精修可行性（2026-10-30）\n',
    '**已完成只读可行性和全片计算代理测试；当前两个有界ROI方案均不满足覆盖门槛，不启动这套裁剪训练。** 视频70帧的误差都在窗口内，但图像集存在无种子漏检；已有粗分割中的结石错误还会被“保护结石”约束保留。局部精修有研究空间，不能据此宣称已有精度改善或可用精修模型。\n',
    '## 范围、与旧实验的区别\n',
    '使用冻结的model1两网组合为候选、三网classwise为参考；读取70视频参考与132图像开发参考、已有连续概率和完整两片输出。旧联合ROI曾同时为结石/光纤分配预算，本轮只按光纤种子选区，并单列model1相对classwise丢失TP/新增FP。没有重跑训练、改模型、改默认、改变标签或原521/132/144/29划分。视频和派生物不训练，stone_new保持训练排除，TRAIN和144保留test像素读取0。\n',
    '窗口沿用192原图像素、边缘16、步距不超过160、小组件下限4；hard2用model1融合掩膜的光纤像素，soft02_2用model1原图光纤概率≥0.2，仅作为选区种子。均按新增种子覆盖贪心取最多2窗口，并列按坐标固定顺序。无蓝色/入口/最大组件排除、没有强制连通或形状填补。GT仅在选区之后用于评估，不参与窗口、排序或阈值。0.2不是分割阈值修改。不限窗口只作容量诊断，不是额外候选。未按结果调阈值、窗口或门槛。\n',
    '冻结覆盖条件分别作用于70/132：光纤总参考覆盖≥99%、每张有参考光纤帧≥95%、FN/FP各自覆盖≥95%、窗口数p95≤2。覆盖只表示可作用范围，不是分割准确率。完整两片计算代理的p95还需在各自源帧间隔内；这同样不是最终质量或实时显示验收。原分割门槛未改。\n',
    '## 自动ROI覆盖\n']
    coverage=[]
    for scope,d in data.items():
        for name,v in d['profiles'].items():
            a=v['aggregate'];r=a['regions'];coverage.append([scope,name,fmt(r['target']['coverage']),fmt(r['target']['minimum_frame_coverage']),fmt(r['fn']['coverage']),r['fn']['outside'],fmt(r['fp']['coverage']),fmt(a['roi_count_mean']),v['coverage_feasible']])
    report.append(table(['参考','方案','参考覆盖','最低帧覆盖','FN覆盖','窗外FN','FP覆盖','平均窗口','覆盖通过'],coverage))
    report.append('视频两组均覆盖453420个参考光纤像素、35637 FN、12492 FP，也覆盖model1相对classwise损失的2205 TP和新增3306 FP。图像两组覆盖110903/111348参考像素，5622/6067 FN及全部58343 FP；1025丢失TP和3254新增FP均在窗内，但445个原FN落在窗外。图像平均覆盖99.6004%掩盖了一张零覆盖；不限窗口仍漏这445像素，增加窗口数不是解法。所有70视频均有光纤参考；无光纤情形主要从图像开发集观察，不作为无光纤真实视频泛化证明。\n')
    report.append('## 无种子与强置信错误\n')
    for scope,d in data.items():
        report.append(table([scope+'错误类型','像素数','P<0.1','0.1–0.2','0.2–0.5','0.5–0.9','≥0.9'],[[k,v['pixels'],*v['probability_bins_0_01_02_05_09_1']] for k,v in d['errors']['summary'].items()]))
        report.append(table([scope+'空间/负帧诊断','值'],[[k,v] for k,v in spatial['sets'][scope]['summary'].items()]))
    report.append('video70丢失TP的2179/2205、image132的1012/1025位于当前光纤预测3px内，说明相对classwise的损失主要在轮廓附近；这不能自动证明解剖尖端漏分。图像58343个参考FP中41306个P≥0.9，因此“只删低置信游离候选”不能处理大部分参考FP。132图像的标签仍待审核、光纤完整性未核实，GT无光纤不等于已证实无光纤；表中FP是按现有参考计数，不自动修改标签或认定全部为模型错误。\n')
    report.append('FN/丢失TP的距离采用到粗预测光纤的距离；原始错误JSON的FP到预测距离恒为0，不能用于判断远处误检。补充error_spatial_diagnosis.json改用FP到参考光纤的距离，并统计不与参考光纤重叠的预测组件；这些是事后诊断，不是推理特征。保留不连通可见区域，没有用参考选择一个“正确组件”。\n')
    failures=[]
    for scope,d in data.items():
        for r in d['profiles']['hard2']['frames']:
            if r['regions']['fn']['outside']:
                failures.append([scope,r['sample_id'],r['roi_count'],r['regions']['target']['total'],r['regions']['fn']['outside'],r['uncapped_counts']['regions']['fn']['outside']])
    report.append(table(['参考','窗外漏检样本','窗口数','参考光纤像素','窗外FN','不限窗口仍在外FN'],failures))
    seed_cases=[]
    image_rows=read(ROOT/cfg['image_manifest'])['val'];image_m1=read(previous/'image_native/member1_2net.json')
    for i,record in enumerate(data['image_native']['profiles']['hard2']['frames']):
        if not record['regions']['fn']['outside']:continue
        target=np.array(Image.open(image_rows[i]['mask']))
        with np.load(image_m1['records'][i]['probability_reference']) as cache:p=cache['a0_member1_native'][2]
        yy,xx=np.where(target==2)
        seed_cases.append({'sample_id':record['sample_id'],'native_shape':list(target.shape),'reference_fiber_pixels':int((target==2).sum()),
            'reference_fiber_bbox':[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)],'gt_fiber_probability_max':float(p[target==2].max()),'gt_fiber_probability_mean':float(p[target==2].mean()),
            'source_image_sha256':sha256(image_rows[i]['image']),'reference_mask_sha256':sha256(image_rows[i]['mask']),'visual_observation':'Image includes desktop window/white margins; blue edge object is small in whole-frame input. Observation, not a proven causal explanation or authorization to relabel/crop.'})
    atomic_json(OUT/'missed_seed_case_details.json',{'diagnostic_not_inference':True,'cases':seed_cases})
    report.append('唯一窗外样本是v2的df80c83cf85358cd54605ee9：640×640图像、445个参考光纤像素，原预测光纤为0；这些参考像素的最大P仅0.000244、均值0.0000847。这说明本例不是窗口预算耗尽，也不是略微调低0.2种子阈值就能处理的边界临界情况。可视化可见桌面窗口/留白和边缘小蓝色目标，提示有效视野占比/缩放值得单列诊断；另一最大FP例落在明亮反光区。它们是画面观察，尚不能证明输入尺度或高光是唯一原因，也不自动裁剪FOV、删除样本或改标签。详情见missed_seed_case_details.json和六栏诊断板。\n')
    report.append('## 保护结石后的可修正上界\n')
    report.append('仅作事后上界：自动窗内的有效非结石粗像素，允许按GT完美切换背景/光纤；所有原结石像素保持，不能新写结石，未知与窗外保持。原fiber错占GT-stone只能清成背景，不能生成新的stone。该上界未作为预测/伪标签保存，没有用于训练或选区。\n')
    report.append(table(['参考','方案','实际粗光纤Dice','诊断上界Dice（非模型）','上界Recall（非模型）','被原结石锁住的GT光纤','上界结石Dice'],[[scope,name,fmt(d['coarse_actual_model']['metrics']['fiber_dice']),fmt(v['diagnostic_oracle_not_model']['metrics']['fiber_dice']),fmt(v['diagnostic_oracle_not_model']['metrics']['fiber_recall']),v['aggregate']['regions']['gt_fiber_blocked_by_stone']['total'],fmt(v['diagnostic_oracle_not_model']['metrics']['stone_dice'])] for scope,d in data.items() for name,v in d['profiles'].items()]))
    report.append('85个视频、523个图像参考光纤像素已被粗分割写为结石，即使在ROI中，也不能在“结石逐像素不变”条件下恢复。图像额外445像素位于窗外，故该上界仍有968 FN。保护结石只能防止新分支进一步改动，不能修复粗分割既有错误。model1相对三网的部分结石指标原已退化，在严格保护model1结石的前提下，单独改光纤无法让这些结石指标变成不退化；不能借高光纤上界宣称满足三网替换规则。\n')
    report.append('所有原29/新41、43/27病例、68/64来源、25/107曝光分组，逐帧覆盖与上界边界/小结石分组保留在video_native.json和image_native.json中。无独立尖端参考；轴线、远端或中心线代理没有转成尖端GT。42/120随机样本未完成，70开发参考不能外推全片精度。\n')
    report.append('## 两段完整视频成本代理\n')
    report.append('各臂/视频独立进程，FP32、batch1、12帧预热，3臂×2694=8082帧。匹配参照为本轮model1原始两网；hard/soft在相同粗管线之后，对每个ROI将192原图裁剪resize256，额外调用已有A0 model1光纤网，映射三类概率回原裁剪并传CPU，结果丢弃。模型参数/身份不变，实际构建2网，额外0–2次已有光纤网调用。这是同形状重网计算代理，未训练成精修模型，不能评价裁剪分割精度；真正轻量网络的成本和额外驻留显存仍未测。\n')
    report.append(table(['臂','视频','帧数','管线P50/P95/max ms','proposal P50/P95 ms','crop P50/P95 ms','ROI均值/p95','诊断离线fps','峰值MiB','帧预算p95通过'],[[name,vid,v['frames'],f"{v['timings']['proxy_pipeline_ms']['p50']:.2f}/{v['timings']['proxy_pipeline_ms']['p95']:.2f}/{v['timings']['proxy_pipeline_ms']['max']:.2f}",f"{v['timings']['proposal_ms']['p50']:.2f}/{v['timings']['proposal_ms']['p95']:.2f}",f"{v['timings']['crop_proxy_ms']['p50']:.2f}/{v['timings']['crop_proxy_ms']['p95']:.2f}",f"{v['roi_count']['mean']:.3f}/{v['roi_count']['p95']:.1f}",f"{v['offline_diagnostic_fps']:.2f}",f"{v['cuda_peak_allocated_mib']:.2f}",v['proxy_within_source_frame_budget_p95']] for name,items in s['full_video'].items() for vid,v in items.items()]))
    report.append('管线包含RGB/PIL预处理、全部粗网、两专家原图三类概率和融合掩膜传CPU、自动ROI、裁剪预处理/前向/概率传输及有限值检查。离线fps包含解码、渲染视频、逐帧哈希核验与编码器释放，最终汇总JSON写盘在计时之外；并非上轮PNG全导出管线。只用本轮匹配比较，不混用2026-10-29历史延迟，不相减两个p95当作净成本。完整timings.json记录总周期、各阶段及最大值；frame日志记录每帧boxes/calls与粗掩膜/RGB身份。\n')
    report.append('各臂8082次粗输出均与既有model1全片结果逐像素一致。没有真实新分割掩膜，四栏视频展示的是当前掩膜和自动窗口。无几何/颜色拟合或历史状态，拟合接受/回退率不适用；无种子时0窗口，保留粗分割，属于跳过而非恢复遗漏。自动ROI不能凭几何填隐藏尖端。\n')
    for vid in ['video_1','video_2']:report.append(f'- [{vid}完整ROI诊断视频]({vid}_roi_comparison.mp4)（Input/当前粗分割/hard窗口/soft窗口，非精修效果）\n')
    report.append('\n## 判断、停止条件和下一步\n')
    report.append('两方案均未通过图像逐帧覆盖和FN覆盖；按冻结条件结束这轮ROI方案，不提高cap、降低阈值、扩大边缘或使用真值补窗口，不自动训练。概率种子没有解决无种子遗漏，也没有表现出图像覆盖优势，优先保留简单hard2作为诊断基准。当前同形状重网代理还需分别看上表预算，不能推断所有未来轻量方案一定慢或一定实时。\n')
    report.append('下一步最小工作是**制定包含全图保底、明确空预测行为的光纤专项学习/推理对照计划**，而不是直接启动当前局部训练。证据支持进一步研究网络与监督/输入尺度（大量FN低P、FP高P），尚未证明颜色、光度增强或尖端外观是主因。先用已有图像标签完整性标志和错误来源分组约束训练计划，不再次把全量人工复核设为开发前置，也不擅自把未审标签改为真值。全图保底与局部细节应独立对照；训练数据限定521原TRAIN，开发验证132+70，视频不训练/伪标，144保留test不读像素，stone_new排除。\n')
    report.append('任何拟训练方案必须另行冻结：ImageNet或合规kidney初始化（不用旧混合A0初始化）、有标签真实增强路径与合理光度范围、背景/光纤监督和结石保护语义、固定预算与对照、原门槛及损失TP检查、独立全片性能/部分参考精度、不可用尖端指标、成本与回滚，然后等待用户批准。当前未选择新网络头或保证增广有效；未启动训练、未更换默认或进入S3。\n')
    report.append(f"4项新测试+6项旧ROI回归通过，{audit['passed']}项独立核验通过。审计复算全部静态覆盖、保护上界混淆矩阵与结石指标，8082条粗掩膜记录、2694个hard提案、全部70个已审帧soft提案、6进程和8个完整视频输出。全片soft概率未另存，审计对其余未审帧只验证边界/calls/身份；不宣称另做了2694帧概率重推理。源码、输入、权重、原输出身份不变。\n")
    report.append('\n- [预检与冻结协议身份](preflight.json)、[工程核验](engineering_checks.json)、[汇总](summary.json)、[空间诊断](error_spatial_diagnosis.json)\n- [图像失败诊断板](image_native_diagnostic_board.jpg)、[视频诊断板](video_native_diagnostic_board.jpg)\n')
    (OUT/'report.md').write_text('\n'.join(report),encoding='utf-8')
    atomic_json(OUT/'report_artifacts.json',{'generator_sha256':sha256(Path(__file__)),'files':{p.name:sha256(p) for p in [OUT/'report.md',OUT/'image_native_diagnostic_board.jpg',OUT/'video_native_diagnostic_board.jpg']}})
    print(json.dumps({'report':str(OUT/'report.md'),'checks':audit['passed']}))
if __name__=='__main__':main()
