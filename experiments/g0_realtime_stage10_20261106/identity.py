"""Explicit frozen inputs; no repository-wide discovery or final access."""
from pathlib import Path
from tools.fiber_reference_workspace import ROOT,read,sha256


def sources():
    old=read(ROOT/'runs/g0_realtime_stage8_20261106/full/v7-pipeline_video_1/identity.json')['sources']
    nine=read(ROOT/'runs/g0_realtime_stage9_20261106/final_audit.json')['current_experiment_sources']
    frozen={**old,**nine}
    assert all(sha256(ROOT/name)==value for name,value in frozen.items())
    current={p.relative_to(ROOT).as_posix():sha256(p) for pattern in ('*.py','*.ps1')
             for p in Path(__file__).parent.glob(pattern)}
    assert not frozen.keys()&current.keys()
    return {**frozen,**current}
