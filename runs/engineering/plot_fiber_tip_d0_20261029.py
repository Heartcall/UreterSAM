"""Descriptive proxy figure. No thresholds, model fitting or tip truth generation."""
import hashlib
import json
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/a1_fiber_tip_d0_20261029'
os.environ['MPLCONFIGDIR']=str(OUT/'matplotlib_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    path=OUT/'summary.json';source=json.loads(path.read_text(encoding='utf-8'));diagnostic=OUT/'diagnosis_summary.json';d=json.loads(diagnostic.read_text(encoding='utf-8'))['video29']
    groups=source['video29']['appearance_pooled_reference_conditioned'];x=[.125,.375,.675,.925]
    fig,axes=plt.subplots(1,2,figsize=(12,4.7));fig.suptitle('D0: reference-axis proxies, NOT anatomical tip ground truth',fontsize=14,weight='bold')
    for key,label,color in [('fiber_probability','A0 fiber probability','#2862bb'),('saturation','HSV saturation','#008a7d'),('value','HSV brightness V','#c28312')]:
        axes[0].plot(x,[groups[f'axial_proxy_{i}'][key]['mean'] for i in range(4)],'o-',label=label,color=color,lw=2)
    axes[0].set(xlim=(0,1),ylim=(0,1.06),xlabel='Normalized reference axial position t',ylabel='Pooled mean (0..1)',title='Appearance and network support');axes[0].legend(loc='lower left');axes[0].grid(alpha=.2)
    values=[d['reference_axial_proxy'][str(i)]['recall'] for i in range(4)];bars=axes[1].bar(range(4),values,color=['#4c78a8']*3+['#db8a33'])
    for i,(bar,value) in enumerate(zip(bars,values)):
        count=d['reference_axial_proxy'][str(i)]['reference_pixels'];axes[1].text(i,value+.015,f'{value:.3f}\nn={count:,}',ha='center',va='bottom',fontsize=10)
    axes[1].set(ylim=(0,1.25),ylabel='Whole-fiber recall within reference proxy',title='Frozen classwise, all29 reviewed frames');axes[1].set_yticks([0,.2,.4,.6,.8,1]);axes[1].set_xticks(range(4),['0–.25','.25–.5','.5–.85','.85–1']);axes[1].set_xlabel('Reference proxy interval (last interval is NOT a tip label)');axes[1].grid(axis='y',alpha=.2)
    fig.text(.5,.025,'GT is used only for post-hoc measurement. No evidence here establishes overexposure, defocus, or physical tip accuracy.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.91));output=OUT/'reference_axis_proxy_profile.png';fig.savefig(output,dpi=170);plt.close(fig)
    (OUT/'profile_figure_identity.json').write_text(json.dumps({'source_sha256':digest(path),'diagnosis_sha256':digest(diagnostic),'plot_script_sha256':digest(__file__),'figure_sha256':digest(output)},indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':main()
