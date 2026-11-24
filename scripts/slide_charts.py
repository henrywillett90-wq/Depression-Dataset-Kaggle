import os; os.makedirs("deck_figs", exist_ok=True); os.makedirs("slide_figs", exist_ok=True)
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
INK,TEAL,AMBER,MUTED,GRID="#16213A","#1F7A7A","#E0A030","#4A566E","#E3E0D8"
plt.rcParams.update({"font.size":17,"axes.edgecolor":MUTED,"axes.labelcolor":INK,"text.color":INK,"xtick.color":INK,"ytick.color":INK,"axes.spines.top":False,"axes.spines.right":False,"axes.titlesize":19,"axes.titleweight":"bold"})
df=pd.read_csv(sys.argv[1]); T="History of Mental Illness"; df["y"]=(df[T]=="Yes").astype(int); base=df.y.mean()
pct=lambda x,_:f"{x:.0%}"
def stat(col,order=None):
    g=df.groupby(col,observed=True).y.agg(["mean","count"]); g["ci"]=1.96*np.sqrt(g["mean"]*(1-g["mean"])/g["count"])
    return g.loc[order] if order else g
def save(n): plt.tight_layout(); plt.savefig("slide_figs/"+n,dpi=130); plt.close()
# 1 income deciles + employment
df["dec"]=pd.qcut(df.Income,10,labels=False); g=stat("dec")
fig,(a,b)=plt.subplots(1,2,figsize=(16,6.2),gridspec_kw={"width_ratios":[1.7,1]})
a.errorbar(range(1,11),g["mean"],yerr=g["ci"],marker="o",ms=10,color=TEAL,ecolor=INK,capsize=4,lw=2.5); a.axhline(base,color=AMBER,ls="--",lw=2.5)
a.yaxis.set_major_formatter(pct); a.set_xticks(range(1,11)); a.set_xlabel("Income decile (1 = lowest 10%)"); a.set_title("By income decile",loc="left"); a.grid(axis="y",color=GRID); a.set_ylim(.18,.43)
e=stat("Employment Status",["Employed","Unemployed"])
b.errorbar(e["mean"],[0,1],xerr=e["ci"],fmt="o",ms=13,color=TEAL,ecolor=INK,capsize=5,lw=2.5); b.axvline(base,color=AMBER,ls="--",lw=2.5)
b.set_yticks([0,1]); b.set_yticklabels(["Employed","Unemployed"]); b.xaxis.set_major_formatter(pct); b.set_xlim(.2,.45); b.set_ylim(-.6,1.6); b.set_title("By employment",loc="left"); b.grid(axis="x",color=GRID)
for i,v in enumerate(e["mean"]): b.text(v,i+.22,f"{v:.1%}",ha="center",fontsize=17,fontweight="bold")
fig.text(.01,.01,f"dashed line = overall {base:.1%}   |   lines = 95% intervals",fontsize=15,color=MUTED); plt.subplots_adjust(bottom=.18)
plt.savefig("slide_figs/income_employment_rates.png",dpi=130); plt.close()
# 2 three small panels
ORD={"Education Level":["High School","Associate Degree","Bachelor's Degree","Master's Degree","PhD"],"Sleep Patterns":["Poor","Fair","Good"],"Dietary Habits":["Unhealthy","Moderate","Healthy"]}
fig,axes=plt.subplots(1,3,figsize=(16,5.6))
for ax,(c,o) in zip(axes,ORD.items()):
    s=stat(c,o); ax.errorbar(s["mean"],range(len(s)),xerr=s["ci"],fmt="o",ms=11,color=TEAL,ecolor=INK,capsize=4,lw=2.5); ax.axvline(base,color=AMBER,ls="--",lw=2.5)
    ax.set_yticks(range(len(s))); ax.set_yticklabels([x.replace(" Degree","") for x in s.index]); ax.set_title(c,loc="left",fontsize=18); ax.xaxis.set_major_formatter(pct); ax.set_xlim(.25,.36); ax.grid(axis="x",color=GRID); ax.set_ylim(-.5,len(s)-.5)
    ax.set_xticks([.26,.30,.34])
fig.text(.01,.01,f"dashed line = overall {base:.1%}   |   lines = 95% intervals",fontsize=15,color=MUTED); plt.subplots_adjust(bottom=.17)
plt.savefig("slide_figs/three_categories.png",dpi=130); plt.close()
# 3 dumbbell: lowest-highest group rate for every variable
cats=["Education Level","Sleep Patterns","Dietary Habits","Alcohol Consumption","Marital Status","Physical Activity Level","Chronic Medical Conditions","Smoking Status","Family History of Depression","Number of Children","History of Substance Abuse"]
df["Age band"]=pd.cut(df.Age,[0,24,34,44,54,64,200],labels=["<25","25-34","35-44","45-54","55-64","65+"])
rows=[]
for c in cats+["Age band"]:
    s=stat(c)["mean"]; rows.append((c,s.min(),s.max()))
rows.append(("Employment Status",e["mean"].min(),e["mean"].max())); rows.append(("Income (decile 1 vs 10)",g["mean"].min(),g["mean"].max()))
d=pd.DataFrame(rows,columns=["v","lo","hi"]); d["gap"]=d.hi-d.lo; d=d.sort_values("gap")
fig,ax=plt.subplots(figsize=(16,7.4))
for i,r in enumerate(d.itertuples()):
    ax.plot([r.lo,r.hi],[i,i],color=MUTED,lw=3,zorder=1); ax.scatter([r.lo],[i],color=TEAL,s=110,zorder=2); ax.scatter([r.hi],[i],color=AMBER,s=110,zorder=2)
    ax.text(r.hi+.004,i,f"{(r.hi-r.lo)*100:.1f} pts",va="center",fontsize=15,color=INK)
ax.axvline(base,color=INK,ls="--",lw=1.5); ax.set_yticks(range(len(d))); ax.set_yticklabels(d.v); ax.xaxis.set_major_formatter(pct); ax.set_xlim(.18,.46); ax.grid(axis="x",color=GRID)
ax.set_title("Share with a history: group with the lowest rate (teal) to the highest (amber)",loc="left",fontsize=19)
save("dumbbell.png")
# 4 income by employment histogram
fig,ax=plt.subplots(figsize=(11,5.8)); cap=df.Income.quantile(.995)
for v,c in [("Employed",TEAL),("Unemployed",AMBER)]:
    ax.hist(df.loc[df["Employment Status"]==v,"Income"].clip(upper=cap),bins=60,alpha=.7,color=c,label=v)
ax.axvspan(22500,30000,color=INK,alpha=.12); ax.text(26250,ax.get_ylim()[1]*.92,"overlap\n22.5k to 30k",ha="center",fontsize=16)
ax.set_xlabel("Income"); ax.set_ylabel("People"); ax.legend(fontsize=17); ax.xaxis.set_major_formatter(lambda x,_:f"{x/1000:.0f}k"); save("income_by_employment.png")
# 5 AUC ladder
res={"Income only":.5906,"Employment only":.5703,"Income + employment":.5917,"All except income and employment":.5473,"All 14 variables":.5929,"All 14, gradient boosting":.5968}
r=pd.Series(res).sort_values(); fig,ax=plt.subplots(figsize=(14,6))
ax.barh(r.index,r.values-.5,left=.5,color=[AMBER if "gradient" in k else TEAL for k in r.index],height=.62); ax.axvline(.5,color=MUTED,lw=2)
for i,v in enumerate(r.values): ax.text(v+.002,i,f"{v:.3f}",va="center",fontsize=17,fontweight="bold")
ax.set_xlim(.48,.64); ax.set_xlabel("ROC-AUC on held-out test data (0.50 = chance)"); ax.grid(axis="x",color=GRID); ax.set_axisbelow(True); save("auc_ladder.png")
# 6 effect sizes
import pandas as pd
t=pd.read_csv("deck_figs/tests.csv").sort_values("effect_size"); fig,ax=plt.subplots(figsize=(14,7.4))
ax.barh(t.variable,t.effect_size,color=[TEAL if k=="Chi-square" else AMBER for k in t.test],height=.65)
for i,v in enumerate(t.effect_size): ax.text(v+.002,i,f"{v:.3f}",va="center",fontsize=15)
ax.axvline(.05,color=MUTED,ls="--",lw=1.5); ax.text(.052,.2,"0.05: 'very weak'",color=MUTED,fontsize=15)
ax.set_xlabel("Effect size   (teal = Cramér's V, amber = rank-biserial correlation)"); ax.set_xlim(0,.22); ax.grid(axis="x",color=GRID); ax.set_axisbelow(True); save("effect_sizes.png")
print("ok")
