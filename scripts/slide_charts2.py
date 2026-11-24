import os; os.makedirs("deck_figs", exist_ok=True); os.makedirs("slide_figs", exist_ok=True)
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
INK,TEAL,AMBER,MUTED,GRID,RED="#16213A","#1F7A7A","#E0A030","#4A566E","#E3E0D8","#B2452F"
plt.rcParams.update({"font.size":17,"axes.edgecolor":MUTED,"axes.labelcolor":INK,"text.color":INK,"xtick.color":INK,"ytick.color":INK,"axes.spines.top":False,"axes.spines.right":False,"axes.titlesize":19,"axes.titleweight":"bold"})
df=pd.read_csv(sys.argv[1]); df["y"]=(df["History of Mental Illness"]=="Yes").astype(int)
deg=df["Education Level"].isin(["Bachelor's Degree","Master's Degree","PhD"]); ns=df["Smoking Status"]!="Current"; un=df["Employment Status"]=="Unemployed"
df["tier"]=np.where(un&ns,"High",np.where(~un&deg&ns,"Low","Middle"))
pct=lambda x,_:f"{x:.0%}"; col={"High":AMBER,"Middle":MUTED,"Low":TEAL}
g=df.groupby("tier").y.agg(["mean","count"]); g["share"]=g["count"]/len(df)
fig,axes=plt.subplots(1,3,figsize=(17,5.9),gridspec_kw={"width_ratios":[1.15,1,1]})
a=axes[0]; order=["High","Middle","Low"]
a.bar(order,[g.loc[t,"mean"] for t in order],color=[col[t] for t in order],width=.62)
for i,t in enumerate(order): a.text(i,g.loc[t,"mean"]+.008,f"{g.loc[t,'mean']:.1%}",ha="center",fontsize=19,fontweight="bold"); a.text(i,.02,f"{g.loc[t,'share']:.0%}\nof people",ha="center",fontsize=15,color="white" if t!="Middle" else "white")
a.set_ylim(0,.5); a.yaxis.set_major_formatter(pct); a.set_title("Rate by tier",loc="left"); a.set_ylabel("Share with a history")
df["iq"]=pd.qcut(df.Income,4,labels=["Q1\nlow","Q2","Q3","Q4\nhigh"]); df["ab"]=pd.cut(df.Age,[0,34,49,64,100],labels=["18-34","35-49","50-64","65+"])
for ax,c,t in [(axes[1],"iq","By income quartile"),(axes[2],"ab","By age band")]:
    p=df.pivot_table(index=c,columns="tier",values="y",aggfunc="mean",observed=True)
    for k in order: ax.plot(range(len(p)),p[k].values,marker="o",ms=9,lw=3,color=col[k],label=k)
    ax.set_ylim(0,.5); ax.set_xticks(range(len(p))); ax.set_xticklabels(p.index); ax.yaxis.set_major_formatter(pct); ax.set_title(t,loc="left"); ax.grid(axis="y",color=GRID)
axes[1].legend(fontsize=15,loc="lower center",ncol=3,frameon=False,columnspacing=1.0,handlelength=1.2)
plt.tight_layout(); plt.savefig("slide_figs/tiers.png",dpi=130); plt.close()
# ladder with 3-tier rule
res={"Employment only":.5703,"All except income and employment":.5473,"Income only":.5906,"Income + employment":.5917,"All 14 variables, logistic":.5929,"All 14 variables, gradient boosting":.5973,"Three-tier rule (3 variables)":.5974}
r=pd.Series(res).sort_values(); fig,ax=plt.subplots(figsize=(14,6.6))
colors=[AMBER if "gradient" in k else (INK if "Three" in k else TEAL) for k in r.index]
ax.barh(r.index,r.values-.5,left=.5,color=colors,height=.62); ax.axvline(.5,color=MUTED,lw=2)
for i,v in enumerate(r.values): ax.text(v+.002,i,f"{v:.3f}",va="center",fontsize=17,fontweight="bold")
ax.set_xlim(.48,.64); ax.set_xlabel("ROC-AUC on held-out test data (0.50 = chance)"); ax.grid(axis="x",color=GRID); ax.set_axisbelow(True)
plt.tight_layout(); plt.savefig("slide_figs/auc_ladder.png",dpi=130); plt.close(); print("ok")
