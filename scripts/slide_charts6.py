import os; os.makedirs("deck_figs", exist_ok=True); os.makedirs("slide_figs", exist_ok=True)
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
INK,TEAL,AMBER,MUTED,GRID,RED="#16213A","#1F7A7A","#E0A030","#4A566E","#E3E0D8","#B2452F"
plt.rcParams.update({"font.size":17,"axes.edgecolor":MUTED,"axes.labelcolor":INK,"text.color":INK,"xtick.color":INK,"ytick.color":INK,"axes.spines.top":False,"axes.spines.right":False,"axes.titlesize":19,"axes.titleweight":"bold"})
O="results/tables/"; base=0.3040940619184571
c=pd.read_csv(O+"recommended_confusion.csv").iloc[0]; tn,fp,fn,tp=[int(c[k]) for k in ["tn","fp","fn","tp"]]
# confusion
cm=np.array([[tn,fp],[fn,tp]]); fig,ax=plt.subplots(figsize=(8.2,6.4))
cmap=LinearSegmentedColormap.from_list("c",["#F6F4EF","#E0A030"])
ax.imshow(cm/cm.sum(1,keepdims=True),cmap=cmap,vmin=0,vmax=.85)
lab=[["True negatives","False positives"],["False negatives","True positives"]]
for i in range(2):
    for j in range(2): ax.text(j,i,f"{lab[i][j]}\n{cm[i,j]:,}\n{cm[i,j]/cm[i].sum():.0%} of actual {'No' if i==0 else 'Yes'}",ha="center",va="center",fontsize=17,fontweight="bold" if i==j else None)
ax.set_xticks([0,1]); ax.set_xticklabels(["Predicted: No","Predicted: Yes"],fontsize=17); ax.set_yticks([0,1]); ax.set_yticklabels(["Actual: No","Actual: Yes"],fontsize=17)
ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
plt.tight_layout(); plt.savefig("slide_figs/confusion.png",dpi=130); plt.close()
# lift
l=pd.read_csv(O+"recommended_lift.csv"); fig,ax=plt.subplots(figsize=(12,6.2))
cols=[TEAL if v<.95 else (MUTED if v<1.1 else AMBER) for v in l.lift]
ax.bar(l.decile,l.lift,color=cols,width=.7); ax.axhline(1,color=INK,ls="--",lw=1.8)
for d,v in zip(l.decile,l.lift): ax.text(d,v+.02,f"{v:.2f}×",ha="center",fontsize=16,fontweight="bold")
ax.set_xticks(l.decile); ax.set_xlabel("Score decile (1 = lowest-scoring 10%, 10 = highest-scoring 10%)"); ax.set_ylabel("Rate ÷ average rate (30.4%)"); ax.set_ylim(0,1.6); ax.grid(axis="y",color=GRID); ax.set_axisbelow(True)
plt.tight_layout(); plt.savefig("slide_figs/lift.png",dpi=130); plt.close()
# PR
p=pd.read_csv(O+"recommended_pr_curve.csv"); fig,ax=plt.subplots(figsize=(10.5,6.4))
ax.plot(p.recall,p.precision,color=AMBER,lw=3.5,label="Recommended model"); ax.axhline(base,color=INK,ls="--",lw=2,label=f"No skill = base rate ({base:.1%})")
prec=tp/(tp+fp); rec=tp/(tp+fn); ax.scatter([rec],[prec],s=170,color=TEAL,zorder=4); ax.annotate(f"Operating point\nprecision {prec:.1%}, recall {rec:.1%}",(rec,prec),xytext=(rec+.1,prec+.12),fontsize=16,arrowprops=dict(arrowstyle="-",color=MUTED))
ax.set_xlabel("Recall (share of people with a history that we catch)"); ax.set_ylabel("Precision (share of flagged people with a history)"); ax.set_xlim(0,1); ax.set_ylim(.2,.62); ax.legend(loc="upper right",frameon=False,fontsize=16); ax.grid(color=GRID); ax.set_axisbelow(True)
plt.tight_layout(); plt.savefig("slide_figs/prcurve.png",dpi=130); plt.close(); print(prec,rec)
