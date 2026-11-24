import os; os.makedirs("deck_figs", exist_ok=True); os.makedirs("slide_figs", exist_ok=True)
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
INK,TEAL,AMBER,MUTED,GRID,RED="#16213A","#1F7A7A","#E0A030","#4A566E","#E3E0D8","#B2452F"
plt.rcParams.update({"font.size":17,"axes.edgecolor":MUTED,"axes.labelcolor":INK,"text.color":INK,"xtick.color":INK,"ytick.color":INK,"axes.spines.top":False,"axes.spines.right":False,"axes.titlesize":19,"axes.titleweight":"bold"})
O="results/tables/"
# --- interactions chart with explicit labels
l=pd.read_csv(O+"interaction_ladder.csv"); h=pd.read_csv(O+"interaction_strength.csv").head(6)
names={"Additive only (depth 1)":"Boosting: additive only\n(depth 1)","Up to 2-way interactions (depth 2)":"Boosting: up to 2-way\n(depth 2)","Up to 3-way interactions (depth 3)":"Boosting: up to 3-way\n(depth 3)","Tuned model (Section 5.2)":"Boosting: tuned\n(depth 8)","Logistic regression (additive)":"Logistic regression\n(additive, for reference)"}
l["n"]=l.model.map(names)
fig,axes=plt.subplots(1,2,figsize=(17,6.6),gridspec_kw={"width_ratios":[1,1.05]})
a=axes[0]; y=np.arange(len(l))[::-1]
a.hlines(y,.59,l.val_auc,color=GRID,lw=3); a.scatter(l.val_auc,y,s=170,color=[MUTED if "ogistic" in n or "additive only" in n else TEAL for n in l.n],zorder=3)
for yi,v in zip(y,l.val_auc): a.text(v+.0004,yi,f"{v:.4f}",va="center",fontsize=16,fontweight="bold")
a.set_yticks(y); a.set_yticklabels(l.n,fontsize=15); a.set_xlim(.590,.6035); a.set_xlabel("Validation ROC-AUC"); a.grid(axis="x",color=GRID); a.set_axisbelow(True)
a.set_title("Gradient boosting, limited to how many\nvariables a tree can combine",loc="left",fontsize=17)
a=axes[1]; h["lab"]=h.feature_a.str.replace(" Status","").str.replace(" Level","")+" × "+h.feature_b.str.replace(" Status","").str.replace(" Level","")
yy=np.arange(len(h))[::-1]; a.barh(yy,h.strength,color=AMBER,height=.62)
for yi,v in zip(yy,h.strength): a.text(v+.002,yi,f"{v:.3f}",va="center",fontsize=16,fontweight="bold")
a.set_yticks(yy); a.set_yticklabels(h.lab); a.set_xlim(0,.098); a.set_xlabel("Interaction strength (log-odds)"); a.grid(axis="x",color=GRID); a.set_axisbelow(True)
a.set_title("Friedman H-statistic on the tuned boosted\nmodel: which pairs interact most",loc="left",fontsize=17)
plt.tight_layout(); plt.savefig("slide_figs/interactions.png",dpi=130); plt.close()
# --- odds ratio chart
a=pd.read_csv(O+"odds_ratios_additive.csv").head(16).iloc[::-1]
b=pd.read_csv(O+"odds_ratios_interaction_model.csv"); b=b[b.term.str.contains(" x ")].head(12).iloc[::-1]
def lab(s):
    for k,v in {"Employment Status":"Employment","Education Level":"Education","Smoking Status":"Smoking","Family History of Depression":"Family history","History of Substance Abuse":"Substance abuse","Chronic Medical Conditions":"Chronic cond.","Physical Activity Level":"Activity","Marital Status":"Marital","Degree":"Deg.","Dietary Habits":"Diet","Alcohol Consumption":"Alcohol","Sleep Patterns":"Sleep"}.items(): s=s.replace(k,v)
    return s
fig,axes=plt.subplots(1,2,figsize=(17,7.4),gridspec_kw={"width_ratios":[1,1.12]})
for ax,d,ttl in [(axes[0],a,"Additive logistic regression\nlargest terms (odds ratio)"),(axes[1],b,"Interaction model: cross terms\n(adjustment on top of the main effects)")]:
    y=np.arange(len(d)); ax.barh(y,d.odds_ratio-1,left=1,color=[RED if v>1 else TEAL for v in d.odds_ratio],height=.64)
    ax.axvline(1,color=INK,lw=1.5); ax.set_yticks(y); ax.set_yticklabels([lab(t) for t in d.term],fontsize=13.5)
    for yi,v in zip(y,d.odds_ratio): ax.text(v+(.012 if v>1 else -.012),yi,f"{v:.2f}",va="center",ha="left" if v>1 else "right",fontsize=14,fontweight="bold")
    lo,hi=min(d.odds_ratio.min(),.7),max(d.odds_ratio.max(),1.25); ax.set_xlim(lo-.12,hi+.07); ax.set_title(ttl,loc="left",fontsize=17); ax.grid(axis="x",color=GRID); ax.set_axisbelow(True); ax.set_xlabel("Odds ratio (1 = no effect)")
plt.tight_layout(); plt.savefig("slide_figs/odds_ratios.png",dpi=130); plt.close()
