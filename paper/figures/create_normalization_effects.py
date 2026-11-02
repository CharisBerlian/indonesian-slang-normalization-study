"""Make a paper-ready paired normalization-effect forest plot from final JSON."""
from pathlib import Path
import json
import math
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pypdfium2 as pdfium

ROOT=Path(__file__).resolve().parents[2]
RESULTS=ROOT/'results/final_evaluation/paired_results.json'
PDF=ROOT/'output/pdf/normalization_effects.pdf'
PAPER=ROOT/'paper/figures/normalization_effects.pdf'
PNG=ROOT/'paper/figures/normalization_effects.png'


def main():
    data=json.loads(RESULTS.read_text(encoding='utf-8'))
    entries=[]
    for key in ('prdec','smsa'):
        entries.extend(data['datasets'][key]['models'][m] for m in ('word','character','indobert'))
    extent=max(.025, math.ceil(max(abs(v) for item in entries for v in (item['mean_effect_interval']['lower'],item['mean_effect_interval']['upper'],item['effect_across_seeds']['mean']))*200)/200+.005)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'ps.fonttype':42,
                         'axes.edgecolor':'#58636c','axes.labelcolor':'#28343d','xtick.color':'#28343d',
                         'ytick.color':'#28343d','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(7.3,3.65),sharex=True)
    names=('Word SVM','Character SVM','IndoBERT')
    colors=('#245781','#33715c','#9a563d'); markers=('o','s','D')
    for ax,ds,title in zip(axes,('prdec','smsa'),('PRDECT-ID','SmSA')):
        ax.set_title(title,fontsize=10,fontweight='bold',pad=10,loc='left')
        for x in (-.01,.01): ax.axvline(x,color='#a8afb5',lw=.8,ls=(0,(3,3)),zorder=0)
        ax.axvline(0,color='#38454e',lw=1.0,zorder=0)
        for y,model,color,marker in zip((2,1,0),('word','character','indobert'),colors,markers):
            entry=data['datasets'][ds]['models'][model]; point=entry['effect_across_seeds']['mean']; iv=entry['mean_effect_interval']
            ax.plot([iv['lower'],iv['upper']],[y,y],color=color,lw=1.55,solid_capstyle='round',zorder=2)
            ax.plot([iv['lower'],iv['upper']],[y,y],linestyle='none',marker='|',markersize=7,color=color,markeredgewidth=1,zorder=3)
            ax.scatter([point],[y],c=color,marker=marker,s=28,zorder=4,edgecolor='white',linewidth=.5)
        ax.set_yticks((2,1,0),names); ax.set_ylim(-.55,2.55); ax.set_xlim(-extent,extent)
        ax.spines['left'].set_visible(False); ax.tick_params(axis='y',length=0,pad=5)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5)); ax.tick_params(axis='x',labelsize=7)
        ax.set_xlabel('Normalized - original macro-F1',fontsize=8,labelpad=6)
        sd=data['datasets'][ds]['models']['indobert']['effect_across_seeds']['sample_sd']
        ax.text(0,-.34,f'IndoBERT effect seed SD: {sd:.4f}',transform=ax.transAxes,fontsize=7.2,color='#53616b',va='top')
    fig.subplots_adjust(left=.145,right=.98,top=.84,bottom=.32,wspace=.42)
    fig.text(.5,.07,'Points: pipeline effects. Bars: nominal 95% paired group-bootstrap intervals,',ha='center',fontsize=7.5,color='#28343d')
    fig.text(.5,.025,'conditional on trained models. Dashed lines: investigator-selected +/-0.01 margin.',ha='center',fontsize=7.5,color='#28343d')
    PDF.parent.mkdir(parents=True,exist_ok=True); PAPER.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(PDF,format='pdf',facecolor='white'); plt.close(fig)
    shutil.copyfile(PDF,PAPER)
    doc=pdfium.PdfDocument(str(PDF)); page=doc[0]
    page.render(scale=3).to_pil().save(PNG)
    print(f'Created {PDF}, {PAPER}, {PNG}')

if __name__=='__main__': main()
