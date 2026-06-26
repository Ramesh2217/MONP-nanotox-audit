"""make_graphical_abstract.py — generate the graphical abstract. No data input."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BLUE="#56B4E9"; AMBER="#E69F00"; GREEN="#009E73"; DARK="#222222"; GREY="#777777"
def _find_repo_root():
    here = Path(__file__).resolve().parent
    for candidate in (here, here.parent):
        if (candidate / "data").exists() or (candidate / "results").exists():
            return candidate
    return here.parent if here.name == "scripts" else here

REPO_ROOT = _find_repo_root()
FIGURES_DIR = REPO_ROOT / "results" / "figures"

def main():
    fig=plt.figure(figsize=(7.0,3.6))
    fig.suptitle("Validation strategy and duplicate records shape apparent ML performance",fontsize=11,fontweight="bold",y=1.02)
    ax=fig.add_axes([0.08,0.16,0.52,0.72])
    labels=["Random\n5-fold CV","Leave-one-\noxide-out","Dose rule\n(baseline)"]; values=[0.61,-0.12,0.41]; colors=[BLUE,AMBER,GREEN]
    bars=ax.bar(labels,values,color=colors,edgecolor="white",width=0.62,zorder=3)
    ax.axhline(0,color=DARK,lw=1.0,zorder=2); ax.set_ylabel("MCC",fontsize=9); ax.set_ylim(-0.25,0.75); ax.tick_params(labelsize=8.5)
    for b,v in zip(bars,values):
        ax.text(b.get_x()+b.get_width()/2,v+(0.04 if v>=0 else -0.06),f"{v:+.2f}",ha="center",va=("bottom" if v>=0 else "top"),fontsize=9,fontweight="bold",color=DARK)
    ax.set_title("Random-split vs material-level evaluation, and a dose baseline",fontsize=8.5,color=GREY,pad=6)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    ax2=fig.add_axes([0.64,0.12,0.34,0.78]); ax2.axis("off")
    ax2.text(0.0,0.94,"Key findings",fontsize=9.5,fontweight="bold",color=DARK)
    bullets=[("Random splitting inflates\nperformance; the gap grows\nwith duplicate records.",BLUE),("Holding out whole oxides\ndrops performance to chance.",AMBER),("A one-variable dose rule\nmatches or beats all five\nML models.",GREEN)]
    y=0.78
    for text,col in bullets:
        ax2.add_patch(plt.Rectangle((0.0,y-0.005),0.04,0.04,color=col,transform=ax2.transAxes,clip_on=False))
        ax2.text(0.09,y+0.015,text,fontsize=8.0,va="top",color=DARK); y-=0.30
    FIGURES_DIR.mkdir(parents=True,exist_ok=True)
    out=FIGURES_DIR/"Graphical_abstract.png"; plt.savefig(out,dpi=300,bbox_inches="tight",facecolor="white"); plt.close(fig)
    print(f"Wrote {out}")

if __name__=="__main__":
    main()
