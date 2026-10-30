"""Archive T1 protocol only. No image decoding, inference or model construction."""
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.fiber_reference_workspace import read, sha256

CFG = 'configs/fiber_temporal_observation_search_20261030.json'
DOC = 'docs/plans/fiber_temporal_observation_search_20261030.md'
OUT = ROOT / 'runs/protocols/fiber_temporal_observation_search_20261030_v1'
checks = []


def check(name, ok):
    if not ok:
        raise AssertionError(name)
    checks.append(name)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    c = read(ROOT / CFG)
    regpath = ROOT / 'annotations/video_evaluation_registry.json'
    statuspath = ROOT / 'annotations/data_review_status_20261027.json'
    reg, status = read(regpath), read(statuspath)
    prior = read(ROOT / 'runs/a1_fiber_temporal_state_gate_20261030/preflight.json')
    old_identity = read(ROOT / 'runs/protocols/fiber_temporal_state_gate_20261030_v1/identity.json')
    preserved_reg = {k: v for k, v in reg.items() if not k.startswith('latest_plan_')}
    check('new protocol archive absent', not OUT.exists())
    check('experiment output absent', not (ROOT / c['planned_output']).exists())
    check('implementation and tests not created', all(not (ROOT / p).exists() for p in c['planned_files']))
    check('four finite arms', c['arms'] == ['raw2', 'light_current', 'evidence_state', 'history_search'])
    check('two unchanged networks', c['base']['networks'] == 2 and c['base']['parameters'] == 49202790)
    check('bounded proposals', c['proposal']['maximum_current_axes'] == 2 and c['proposal']['maximum_history_axes'] == 1)
    check('bounded observation', c['observation']['shaft_stations'] == 8 and c['observation']['transverse_samples_per_side'] == 17 and c['observation']['front_samples_per_profile'] == 25)
    check('fixed score weights', abs(sum(c['quality']['score_weights_neural_edges_residual']) - 1) < 1e-12)
    check('strong and weak separate', c['quality']['strong_mean_fiber_probability_min'] > c['quality']['weak_mean_fiber_probability_min'])
    check('current strong evidence before history', 'BEFORE' in c['state']['strong_current'] and 'no fixed3-frame' in c['state']['strong_current'])
    check('history cannot become independent strong evidence', 'Only weak compatible' in c['history_search']['history_candidate_acceptance'])
    check('bounded partial prediction', c['state']['maximum_prediction_only_frames_per_group'] == 3 and 'unoriented line' in c['state']['partial'])
    check('unchanged visible mask policy', c['mask_policy'].startswith('EXACT raw member1') and 'No state/ROI/template' in c['mask_policy'])
    check('runtime budget is design target', c['budget']['additional_geometry_stage_p95_target_ms'] == 5 and 'design target' in c['budget']['policy'])
    check('all arms full videos even failure', c['execution']['all_arms_full_even_failure'] and c['execution']['fresh_process_per_arm_per_video'])
    check('eight full processes10776frames', c['execution']['expected_processes'] == len(c['arms']) * 2 == 8 and c['execution']['expected_full_forward_frames'] == sum(c['execution']['frames']) * 4 == 10776)
    check('frozen static counts and no tip truth', c['data']['static_video_frames'] == 70 and c['data']['static_image_frames'] == 132 and not c['data']['dedicated_tip_truth'])
    check('T0 completed retained', status['fiber_temporal_state_control_executed'] and not status['fiber_temporal_state_control_adopted'])
    check('T0 all existing sources unchanged', all(sha256(ROOT / p) == h for p, h in prior['sources'].items()))
    check('T0 original protocol unchanged', all(sha256(ROOT / p) == h for p, h in old_identity['files'].items()))
    check('weights unchanged without loading models', all(sha256(Path(v['path'])) == v['sha256'] for v in prior['checkpoints'].values()))
    check('frozen manifests unchanged', all(sha256(Path(p)) == h for p, h in old_identity['manifests'].items()))
    vm, im = read(ROOT / c['data']['video_manifest']), read(ROOT / c['data']['image_manifest'])
    check('metadata-only70/132 and521split', len(vm['val']) == 70 and len(im['val']) == 132 and len(im['train']['labeled']) == 521 and len(im['reserved_test']) == 144)
    check('active development reference unchanged', reg['active_video_test_frames'] == 70 and reg['whole_video_probability_sample_completed'] == 42 and not reg['whole_video_estimate_available'])
    check('actual state result identity unchanged', sha256(Path(reg['latest_state_control_result'])) == reg['latest_state_control_result_sha256'])
    check('actual inference result identity unchanged', sha256(Path(reg['latest_inference_control_result'])) == reg['latest_inference_control_result_sha256'])
    document = (ROOT / DOC).read_text(encoding='utf-8')
    local_links = re.findall(r'\]\(([^)]+)\)', document)
    check('design relative links resolve', all(((ROOT / DOC).parent / p).resolve().exists() for p in local_links))
    mainpath = ROOT / 'docs/plans/segmentation_temporal_occlusion_plan.md'
    maintext = mainpath.read_text(encoding='utf-8')
    check('main plan expected version', '版本：v2.16' in maintext)
    check('pending T1 protocol found once', maintext.count('- [ ] **当前下一步：T1低成本观测与历史辅助搜索协议**') == 1)

    files = {CFG: sha256(ROOT / CFG), DOC: sha256(ROOT / DOC)}
    for p in ['configs/fiber_temporal_state_gate_20261030.json', 'tools/fiber_shape_prior.py', 'tools/fiber_temporal_state.py', 'tools/run_fiber_temporal_state_control.py', 'tools/member_experts.py', 'tools/run_two_network_experts.py']:
        files[p] = sha256(ROOT / p)
    OUT.mkdir(parents=True)
    archive = OUT / 'protocol_sources.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(ROOT / p, p)
    identity = dict(status='protocol_only_not_implemented_not_executed', files=files,
                    weights=prior['checkpoints'], manifests=old_identity['manifests'],
                    inherited_input_file_identities_from='runs/a1_fiber_temporal_state_gate_20261030/preflight.json',
                    inherited_input_file_identities_preflight_sha256=sha256(ROOT / 'runs/a1_fiber_temporal_state_gate_20261030/preflight.json'),
                    archive_sha256=sha256(archive), planned_processes=8, planned_frames=10776,
                    actual_new_video_forwards=0, actual_new_static_forwards=0,
                    training_updates=0, training_pixels_read=0, reserved_test_pixels_read=0,
                    validation_pixels_decoded=0, note='Plan consistency checks only; not new unit tests or measured T1 performance')
    save(OUT / 'identity.json', identity)
    check('archive bytes match originals', all(sha256(ROOT / p) == h for p, h in files.items()))
    with zipfile.ZipFile(archive) as z:
        check('archive roundtrip exact bytes', all(z.read(p) == (ROOT / p).read_bytes() for p in files))
    addition = '2026-10-30 T1方案冻结更新（覆盖以上历史下一步）：已制定低成本当前图像观测、当前强证据优先、历史仅辅助搜索的四臂协议；尚未实现或运行。每帧最多2个当前轴线及1个历史候选，额外几何阶段p95≤5ms仅为设计目标，总管线仍独立检验源帧预算。强当前证据越界时本帧重获，不无条件等3帧；端部未观测不缩短光纤，历史不能填可见像素。下一步按冻结协议新增实现、机制测试、202静态及8独立进程10776帧完整回放；无训练、默认替换、像素融合或结石S3。'
    updated = []
    for line in maintext.replace('版本：v2.16', '版本：v2.17', 1).splitlines():
        if line.startswith('| S2 |'):
            line = '| S2 | T0完整状态对照已完成、成本失败；T1轻量当前观测/历史搜索四臂协议已冻结，尚未实现或实测 | 下一步按T1固定候选与证据规则实施、测试202静态和每臂两片完整回放；5ms只是设计目标，掩膜不变，不训练或默认采用 |'
        elif line.startswith('- [ ] **当前下一步：T1低成本观测与历史辅助搜索协议**'):
            line = '- [x] **T1低成本观测与历史辅助搜索协议**：四臂、当前强证据优先、最多2+1候选、部分组有效性、因果回退及预算已冻结；仅方案归档，未实现/执行。详见[T1方案](fiber_temporal_observation_search_20261030.md)。\n- [ ] **当前下一步：T1实现与完整四臂对照**：新增轻量观测器、证据状态器和独立实验入口；机制测试后202静态原图复现，四臂各完整2694帧、8独立进程10776帧；完整保留失败和成本结果。分割像素仍不变，不自动训练、像素融合、默认采用或进入S3。'
        updated.append(line)
    mainpath.write_text('\n'.join(updated) + '\n\n' + addition + '[方案](fiber_temporal_observation_search_20261030.md)。\n', encoding='utf-8')
    rp = ROOT / 'README.md'
    rt = rp.read_text(encoding='utf-8')
    check('README insertion anchor unique', rt.count('\n## 实现说明') == 1)
    rp.write_text(rt.replace('\n## 实现说明', '\n' + addition + '[方案](docs/plans/fiber_temporal_observation_search_20261030.md)。\n\n## 实现说明', 1), encoding='utf-8')
    prefix = 'fiber_temporal_observation_search'
    status.update({prefix + '_plan_prepared': True, prefix + '_implemented': False,
                   prefix + '_executed': False, prefix + '_adopted': False,
                   prefix + '_plan': DOC, prefix + '_config': CFG,
                   prefix + '_plan_identity': str(OUT / 'identity.json')})
    status.update(current_plan_checks=str(OUT / 'engineering_checks.json'),
                  latest_instruction='Continue next main-plan step: prepare and freeze T1 protocol',
                  latest_instruction_effect='T1 protocol archived only; zero new inference/training; T0 actual results unchanged',
                  execution_order_confirmation='T1 protocol ready; next implement finite four-arm experiment, no automatic training/pixel fusion/default/S3',
                  remaining_prerequisites=[
                      'Next: implement frozen T1 observer/state/runner and causal mechanism tests; no existing fitter or default changes',
                      'Then202static and4arms x2694full frames in8fresh processes, including failed candidates; geometry5ms is a target, not measured',
                      'Exact raw model1 masks retained; model1 remains unadopted and frozen classwise comparison retained',
                      'No independent tip/angle truth;70repeated-development refs and42of120 partial random sample cannot establish full-video accuracy',
                      'No training or pixel fusion/default replacement/stoneS3; videos never train, stone_new excluded,144test pixels excluded'
                  ])
    save(statuspath, status)
    reg.update(latest_plan_config=CFG, latest_plan_document=DOC, latest_plan_identity=str(OUT / 'identity.json'), latest_plan_executed=False)
    save(regpath, reg)
    check('actual result registry retained', {k: v for k, v in read(regpath).items() if not k.startswith('latest_plan_')} == preserved_reg)
    check('T0 execution checks retained', read(statuspath)['current_execution_checks'] == status['current_execution_checks'])
    check('T1 still unimplemented unexecuted', not (ROOT / c['planned_output']).exists() and all(not (ROOT / p).exists() for p in c['planned_files']))
    save(OUT / 'engineering_checks.json', dict(passed=len(checks), failed=0, checks=checks, scope='protocol consistency and identity, not T1 execution or unit tests'))
    print(json.dumps(dict(status='T1_protocol_frozen_not_executed', checks=len(checks), main_plan='v2.17', planned_full_forwards=10776, actual_new_forwards=0)))


if __name__ == '__main__':
    main()
