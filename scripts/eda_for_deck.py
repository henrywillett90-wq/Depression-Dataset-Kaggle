import os; os.makedirs("deck_figs", exist_ok=True); os.makedirs("slide_figs", exist_ok=True)
import sys, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
INK,TEAL,AMBER,MUTED,PAPER="#16213A","#1F7A7A","#E0A030","#4A566E","#F6F4EF"
plt.rcParams.update({"font.family":"DejaVu Sans","axes.edgecolor":MUTED,"axes.labelcolor":INK,"text.color":INK,
  "xtick.color":INK,"ytick.color":INK,"axes.facecolor":"white","figure.facecolor":"white","axes.spines.top":False,"axes.spines.right":False})
OUT="deck_figs/"
df=pd.read_csv(sys.argv[1]); df.columns=df.columns.str.strip()
T="History of Mental Illness"
num=["Age","Number of Children","Income"]
cat=[c for c in df.columns if c not in num+[T,"Name"]]
y=(df[T].astype(str).str.strip().str.lower()=="yes").astype(int); df["y"]=y; base=y.mean()
res={"n":len(df),"base":base}

# ---- missing / quality ----
miss=pd.DataFrame({"missing":df.isna().sum(),"blank_text":[(df[c].astype(str).str.strip()=="").sum() if df[c].dtype==object or str(df[c].dtype).startswith("str") else 0 for c in df.columns],"n_unique":df.nunique()})
miss["missing_pct"]=miss["missing"]/len(df)*100
miss.to_csv(OUT+"missing.csv")
res["total_missing"]=int(df.isna().sum().sum()); res["dupe_rows"]=int(df.duplicated().sum()); res["dupe_names"]=int(df["Name"].duplicated().sum())
res["income_zero"]=int((df["Income"]<=0).sum()); res["age_range"]=[int(df.Age.min()),int(df.Age.max())]
res["income_q"]=[float(df.Income.quantile(q)) for q in (0,.5,.99,1)]

# ---- tests ----
rows=[]
for c in num:
    a,b=df.loc[y==1,c],df.loc[y==0,c]
    u,p=stats.mannwhitneyu(a,b,alternative="two-sided"); r=2*u/(len(a)*len(b))-1
    rows.append(dict(variable=c,test="Mann-Whitney U",statistic=u,effect_size=abs(r),signed=r,p_value=p,
        detail=f"median Yes {a.median():,.0f} vs No {b.median():,.0f}"))
for c in cat:
    tab=pd.crosstab(df[c],y); chi2,p,dof,_=stats.chi2_contingency(tab,correction=False)
    v=np.sqrt(chi2/(tab.values.sum()*(min(tab.shape)-1))); rate=df.groupby(c)["y"].mean()
    rows.append(dict(variable=c,test="Chi-square",statistic=chi2,effect_size=v,signed=np.nan,p_value=p,
        detail=f"{rate.idxmax()} {rate.max():.1%} vs {rate.idxmin()} {rate.min():.1%}"))
t=pd.DataFrame(rows).sort_values("effect_size",ascending=False).reset_index(drop=True)
t["p_bonf"]=(t.p_value*len(t)).clip(upper=1)
t["strength"]=t.effect_size.map(lambda x:"negligible" if x<.05 else "very weak" if x<.10 else "weak" if x<.20 else "moderate" if x<.30 else "strong")
t.to_csv(OUT+"tests.csv",index=False); res["tests"]=t[["variable","test","effect_size","strength","p_bonf","detail"]].round(5).to_dict("records")

# ---- charts ----
def save(n): plt.tight_layout(); plt.savefig(OUT+n,dpi=160); plt.close()
# effect sizes
fig,ax=plt.subplots(figsize=(9,6.2)); d=t.iloc[::-1]
ax.barh(d.variable,d.effect_size,color=[TEAL if k=="Chi-square" else AMBER for k in d.test])
for x,(i,r) in zip(d.effect_size,d.iterrows()): ax.text(x+.0004,list(d.index).index(i),f"{x:.3f}",va="center",fontsize=10,color=INK)
ax.axvline(.05,color=MUTED,ls="--",lw=1); ax.text(.05,-0.9,"0.05 = 'very weak'",color=MUTED,fontsize=9)
ax.set_xlabel("Effect size (Cramér's V, teal;  |rank-biserial|, amber)"); ax.set_title("How strongly each variable relates to history of mental illness",loc="left",fontsize=13)
save("effect_sizes.png")
# rate by category: 2 images
def rate(col,order=None):
    g=df.groupby(col,observed=True)["y"].agg(["mean","count"]); g["ci"]=1.96*np.sqrt(g["mean"]*(1-g["mean"])/g["count"])
    return g.loc[order] if order else g.sort_values("mean")
ORD={"Education Level":["High School","Associate Degree","Bachelor's Degree","Master's Degree","PhD"],"Physical Activity Level":["Sedentary","Moderate","Active"],
 "Alcohol Consumption":["Low","Moderate","High"],"Dietary Habits":["Unhealthy","Moderate","Healthy"],"Sleep Patterns":["Poor","Fair","Good"],"Smoking Status":["Non-smoker","Former","Current"]}
top=[c for c in t.variable if c in cat]
for k,(name,cols) in enumerate([("rates_a.png",top[:6]),("rates_b.png",top[6:12])]):
    if not cols: continue
    fig,axes=plt.subplots(2,3,figsize=(15,7.6))
    for ax,c in zip(axes.ravel(),cols):
        order=[o for o in ORD.get(c,[]) if o in set(df[c])] or None; g=rate(c,order)
        ax.errorbar(g["mean"],range(len(g)),xerr=g["ci"],fmt="o",color=TEAL,ecolor=INK,capsize=4,ms=11,lw=2); ax.set_yticks(range(len(g))); ax.set_yticklabels(g.index.astype(str)); ax.grid(axis="x",color="#E3E0D8"); ax.set_axisbelow(True)
        ax.axvline(base,color=AMBER,ls="--",lw=2)
        lo,hi=(g["mean"]-g["ci"]).min(),(g["mean"]+g["ci"]).max(); ax.set_xlim(max(0,lo-.03),min(1,hi+.03)); ax.set_ylim(-.6,len(g)-.4)
        v=t.loc[t.variable==c,"effect_size"].iloc[0]; ax.set_title(f"{c}  (V = {v:.3f})",fontsize=11,loc="left")
        ax.xaxis.set_major_formatter(lambda x,_:f"{x:.0%}")
    for ax in axes.ravel()[len(cols):]: ax.axis("off")
    fig.suptitle(f"Share with a history of mental illness, by group  |  dashed line = overall {base:.1%}  |  bars = 95% CI",x=.01,ha="left",fontsize=13)
    save(name)
# numeric: age band & income quartile & children
df["Age band"]=pd.cut(df.Age,[0,24,34,44,54,64,200],labels=["<25","25-34","35-44","45-54","55-64","65+"])
df["Income quartile"]=pd.qcut(df.Income,4,labels=["Q1 low","Q2","Q3","Q4 high"],duplicates="drop")
fig,axes=plt.subplots(1,3,figsize=(15,4.6))
for ax,c in zip(axes,["Age band","Income quartile","Number of Children"]):
    g=df.groupby(c,observed=True)["y"].agg(["mean","count"]); ci=1.96*np.sqrt(g["mean"]*(1-g["mean"])/g["count"])
    ax.errorbar(range(len(g)),g["mean"],yerr=ci,marker="o",color=TEAL,ecolor=INK,capsize=4,lw=2); ax.axhline(base,color=AMBER,ls="--",lw=2)
    ax.set_xticks(range(len(g))); ax.set_xticklabels(g.index.astype(str)); ax.set_title(c,loc="left",fontsize=12); ax.yaxis.set_major_formatter(lambda x,_:f"{x:.0%}")
save("rates_numeric.png")
# target balance + distributions
fig,axes=plt.subplots(1,3,figsize=(15,4.4))
axes[0].bar(["No","Yes"],[(y==0).mean(),(y==1).mean()],color=[TEAL,AMBER]); axes[0].set_title("Target balance",loc="left"); axes[0].yaxis.set_major_formatter(lambda x,_:f"{x:.0%}")
for ax,c,lab in [(axes[1],"Age","Age"),(axes[2],"Income","Income")]:
    for v,col,n in [(0,TEAL,"No"),(1,AMBER,"Yes")]:
        s=df.loc[y==v,c]; ax.hist(s.clip(upper=s.quantile(.99)),bins=40,alpha=.55,color=col,density=True,label=n)
    ax.set_title(f"{lab} by group (density)",loc="left"); ax.legend(title="History")
save("distributions.png")
# mutual information
from sklearn.feature_selection import mutual_info_classif
X=pd.DataFrame({c:df[c] for c in num}); 
for c in cat: X[c]=df[c].astype("category").cat.codes
s=X.sample(min(60000,len(X)),random_state=0)
mi=pd.Series(mutual_info_classif(s,y.loc[s.index],discrete_features=[c in cat for c in s.columns],random_state=0),index=s.columns).sort_values()
fig,ax=plt.subplots(figsize=(8,5.8)); ax.barh(mi.index,mi.values,color=AMBER); ax.set_title("Mutual information with the target (nats)",loc="left"); save("mutual_info.png")
res["mi"]=mi.sort_values(ascending=False).round(5).to_dict()
# predictor correlation
enc=X.copy(); enc["Income"]=np.log1p(enc["Income"]); c=enc.corr(method="spearman")
fig,ax=plt.subplots(figsize=(9,7.4)); im=ax.imshow(c,cmap="RdBu_r",vmin=-1,vmax=1); ax.set_xticks(range(len(c))); ax.set_yticks(range(len(c)))
ax.set_xticklabels(c.columns,rotation=60,ha="right",fontsize=8); ax.set_yticklabels(c.columns,fontsize=8); plt.colorbar(im,shrink=.7); ax.set_title("Spearman correlation between predictors",loc="left"); save("correlation.png")
off=c.where(~np.eye(len(c),dtype=bool)).abs().stack().sort_values(ascending=False); res["max_corr"]=[off.index[0][0],off.index[0][1],float(off.iloc[0])]
emp=df.groupby("Employment Status")["Income"].median().round(0).to_dict(); res["income_by_employment"]=emp
json.dump(res,open(OUT+"summary.json","w"),indent=1,default=float)
print(json.dumps({k:v for k,v in res.items() if k!="tests"},indent=1,default=float)); print(t[["variable","effect_size","strength","p_bonf"]].round(4).to_string())
