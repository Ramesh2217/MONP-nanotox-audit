"""make_workflow_figure.py — generate the analysis-workflow diagram (Figure 3). No data input."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

BLUE="#56B4E9"; AMBER="#E69F00"; GREY="#999999"; DARK="#222222"; LIGHT="#F2F2F2"; GREEN="#009E73"
def _find_repo_root():
    here = Path(__file__).resolve().parent
    for candidate in (here, here.parent):
        if (candidate / "data").exists() or (candidate / "results").exists():
            return candidate
    return here.parent if here.name == "scripts" else here

REPO_ROOT = _find_repo_root()
FIGURES_DIR = REPO_ROOT / "results" / "figures"

def box(ax,x,y,w,h,text,fc=LIGHT,ec=GREY,tc=DARK,fs=9,bold=False,lw=1.2):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle="round,pad=0.08,rounding_size=0.12",fc=fc,ec=ec,lw=lw,zorder=2))
    ax.text(x,y,text,ha="center",va="center",fontsize=fs,color=tc,fontweight=("bold" if bold else "normal"),zorder=3)

def arrow(ax,x1,y1,x2,y2,color=GREY,lw=1.5):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=14,color=color,lw=lw,zorder=1,shrinkA=2,shrinkB=2))

def main():
    fig,ax=plt.subplots(figsize=(7.4,8.7)); ax.set_xlim(0,10); ax.set_ylim(0,12.4); ax.axis("off")
    ax.text(5,12.0,"Overview of the benchmark audit and validation workflow",ha="center",va="center",fontsize=12.5,fontweight="bold",color=DARK)
    box(ax,5,11.1,5.8,0.7,"Original NanoTox benchmark (483 records, 5 metal oxides)",fc="white",bold=True)
    box(ax,5,9.9,6.4,0.9,"Duplicate audit (performed before preprocessing):\n119 exact duplicate records removed \u2192 364 unique records",fc=LIGHT,fs=8.7)
    arrow(ax,5,10.75,5,10.35)
    arrow(ax,5,9.45,2.7,8.6); arrow(ax,5,9.45,7.3,8.6)
    box(ax,2.7,8.15,4.0,0.95,"Sample-level random\n5-fold CV\nMCC \u2248 0.61",fc=BLUE,tc="white",bold=True,fs=9)
    box(ax,7.3,8.15,4.0,0.95,"Leave-one-oxide-out (LOMO)\nacross five metal oxides\nMCC \u2248 \u22120.12",fc=AMBER,tc="white",bold=True,fs=8.6)
    box(ax,5,6.7,6.8,0.8,"Performance inflation increases with duplicate fraction\n(slope 0.261, R\u00b2 0.817)",fc="white",fs=8.7)
    arrow(ax,2.7,7.675,4.2,7.1); arrow(ax,7.3,7.675,5.8,7.1)
    box(ax,5,5.35,7.6,0.9,"Model benchmarking under LOMO: five ML models\ncompared with the dose-threshold baseline (MCC = 0.410)",fc=GREEN,tc="white",bold=True,fs=8.8)
    arrow(ax,5,6.3,5,5.8)
    y6=3.75
    box(ax,1.85,y6,3.3,1.45,"Bootstrap confidence\nintervals + permutation\ntests (XGBoost p=0.062;\nMLP p=0.004)",fc=LIGHT,fs=8.2)
    box(ax,5.0,y6,3.0,1.45,"Stability analyses\n(random seeds +\nhyperparameter\nsensitivity)",fc=LIGHT,fs=8.2)
    box(ax,8.15,y6,3.3,1.45,"Model interpretation\n(feature decomposition\n+ per-oxide analysis)",fc=LIGHT,fs=8.2)
    arrow(ax,5,4.9,1.85,4.48); arrow(ax,5,4.9,5.0,4.48); arrow(ax,5,4.9,8.15,4.48)
    box(ax,5,1.5,8.8,1.5,"Apparent model performance depends strongly on validation strategy\nand duplicate records. Under leave-one-oxide-out validation, no evaluated\nmodel exceeded the dose-threshold baseline; one model (MLP) identified a\nweak but statistically significant signal that remained below baseline.",fc="white",ec=DARK,bold=False,fs=8.5,lw=1.6)
    arrow(ax,1.85,3.0,4.2,2.25); arrow(ax,5.0,3.0,5.0,2.25); arrow(ax,8.15,3.0,5.8,2.25)
    plt.tight_layout(); FIGURES_DIR.mkdir(parents=True,exist_ok=True)
    out=FIGURES_DIR/"Figure_workflow.png"; plt.savefig(out,dpi=200,bbox_inches="tight",facecolor="white"); plt.close(fig)
    print(f"Wrote {out}")

if __name__=="__main__":
    main()
