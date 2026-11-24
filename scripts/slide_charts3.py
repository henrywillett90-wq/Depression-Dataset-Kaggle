import os; os.makedirs("deck_figs", exist_ok=True); os.makedirs("slide_figs", exist_ok=True)
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
INK,TEAL,AMBER,MUTED,GRID="#16213A","#1F7A7A","#E0A030","#4A566E","#E3E0D8"
plt.rcParams.update({"font.size":17,"axes.edgecolor":MUTED,"axes.labelcolor":INK,"text.color":INK,"xtick.color":INK,"ytick.color":INK,"axes.spines.top":False,"axes.spines.right":False,"axes.titlesize":19,"axes.titleweight":"bold"})
O="results/tables/"
# 1 tuning
t=pd.read_csv(O+"bayes_trials_hgb.csv"); x=np.arange(1,len(t)+1); b=t.loc[t.cv_auc.idxmax()]
fig,ax=plt.subplots(figsize=(14,5.8))
ax.axhspan(b.cv_auc-b.cv_sd,b.cv_auc+b.cv_sd,color=AMBER,alpha=.22,lw=0)
ax.text(len(t)+.3,b.cv_auc+b.cv_sd-.0009,"fold-to-fold noise\n(±1 sd of the best)",ha="right",va="top",fontsize=15,color=MUTED)
ax.plot(x,t.cv_auc.cummax(),color=AMBER,lw=3,label="best so far")
ax.scatter(x,t.cv_auc,s=80,color=TEAL,zorder=3,label="one setting tried")
ax.set_ylim(.589,.6125); ax.set_xlim(.5,len(t)+.5)
ax.set_xlabel("Trial (each is a 5-fold cross-validation)"); ax.set_ylabel("Mean CV ROC-AUC"); ax.grid(axis="y",color=GRID); ax.set_axisbelow(True)
ax.legend(loc="lower right",frameon=False,fontsize=15); ax.set_title("All 25 settings score within 0.002 of each other",loc="left")
plt.tight_layout(); plt.savefig("slide_figs/tuning.png",dpi=130); plt.close()
# 2 interactions: ladder + H
l=pd.read_csv(O+"interaction_ladder.csv"); h=pd.read_csv(O+"interaction_strength.csv").head(6)
names={"Additive only (depth 1)":"Additive only","Up to 2-way interactions (depth 2)":"Up to 2-way","Up to 3-way interactions (depth 3)":"Up to 3-way","Tuned model (Section 5.2)":"Tuned boosting","Logistic regression (additive)":"Logistic (additive)"}
l["n"]=l.model.map(names)
fig,axes=plt.subplots(1,2,figsize=(17,6.2),gridspec_kw={"width_ratios":[1,1.05]})
a=axes[0]; y=np.arange(len(l))[::-1]
a.hlines(y,.59,l.val_auc,color=GRID,lw=3); a.scatter(l.val_auc,y,s=170,color=[MUTED if "Additive" in n or "additive" in n else TEAL for n in l.n],zorder=3)
for yi,v in zip(y,l.val_auc): a.text(v+.0004,yi,f"{v:.4f}",va="center",fontsize=16,fontweight="bold")
a.set_yticks(y); a.set_yticklabels(l.n); a.set_xlim(.590,.6035); a.set_xlabel("Validation ROC-AUC"); a.grid(axis="x",color=GRID); a.set_axisbelow(True)
a.set_title("Allowing interactions adds a little",loc="left")
a=axes[1]; h["lab"]=h.feature_a.str.replace(" Status","").str.replace(" Level","")+" × "+h.feature_b.str.replace(" Status","").str.replace(" Level","")
yy=np.arange(len(h))[::-1]; a.barh(yy,h.strength,color=AMBER,height=.62)
for yi,v in zip(yy,h.strength): a.text(v+.002,yi,f"{v:.3f}",va="center",fontsize=16,fontweight="bold")
a.set_yticks(yy); a.set_yticklabels(h.lab); a.set_xlim(0,.098); a.set_xlabel("Interaction strength (log-odds; 0 = purely additive)"); a.grid(axis="x",color=GRID); a.set_axisbelow(True)
a.set_title("Which pairs interact most",loc="left")
plt.tight_layout(); plt.savefig("slide_figs/interactions.png",dpi=130); plt.close()
# 3 heatmaps
import glob
files=[("Education_Level_x_Employment_Status","Education × Employment"),("Smoking_Status_x_Employment_Status","Smoking × Employment"),("Education_Level_x_Smoking_Status","Education × Smoking")]
gs=[pd.read_csv(O+f"risk_grid_{f}.csv",index_col=0) for f,_ in files]
vmin=min(g.values.min() for g in gs); vmax=max(g.values.max() for g in gs)
fig,axes=plt.subplots(1,3,figsize=(18,5.6))
from matplotlib.colors import LinearSegmentedColormap
cm=LinearSegmentedColormap.from_list("c",["#F6F4EF","#E0A030","#B2452F"])
for ax,g,(f,t_) in zip(axes,gs,files):
    ax.imshow(g.values,cmap=cm,vmin=vmin,vmax=vmax,aspect="auto")
    for i in range(g.shape[0]):
        for j in range(g.shape[1]): ax.text(j,i,f"{g.values[i,j]:.0%}",ha="center",va="center",fontsize=17,fontweight="bold",color=INK)
    ax.set_xticks(range(g.shape[1])); ax.set_xticklabels([c.replace("Degree","").replace("Non-smoker","Non").strip() for c in g.columns],fontsize=14,rotation=0)
    ax.set_yticks(range(g.shape[0])); ax.set_yticklabels([r.replace(" Degree","").replace("Non-smoker","Non-smoker") for r in g.index],fontsize=14)
    ax.set_title(t_,loc="left"); 
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)
plt.tight_layout(); plt.savefig("slide_figs/heatmaps.png",dpi=130); plt.close()
for f,_ in files: print(pd.read_csv(O+f"risk_grid_{f}.csv",index_col=0).round(3))
