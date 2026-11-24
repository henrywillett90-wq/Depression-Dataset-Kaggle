import os; os.makedirs("deck_figs", exist_ok=True); os.makedirs("slide_figs", exist_ok=True)
import sys, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
INK,TEAL,AMBER,MUTED="#16213A","#1F7A7A","#E0A030","#4A566E"
plt.rcParams.update({"axes.edgecolor":MUTED,"axes.labelcolor":INK,"text.color":INK,"xtick.color":INK,"ytick.color":INK,"axes.spines.top":False,"axes.spines.right":False})
OUT="deck_figs/"; df=pd.read_csv(sys.argv[1]); T="History of Mental Illness"
df["y"]=(df[T]=="Yes").astype(int); base=df.y.mean(); df["LogInc"]=np.log1p(df.Income)
cat=["Marital Status","Education Level","Smoking Status","Physical Activity Level","Employment Status","Alcohol Consumption","Dietary Habits","Sleep Patterns","History of Substance Abuse","Family History of Depression","Chronic Medical Conditions"]
num=["Age","Number of Children","LogInc"]
tr,te=train_test_split(df,test_size=.2,stratify=df.y,random_state=42)
def lin(n,c):
    pre=ColumnTransformer([("n",StandardScaler(),n),("c",OneHotEncoder(handle_unknown="ignore"),c)])
    return Pipeline([("p",pre),("m",LogisticRegression(max_iter=2000))])
def hgb(n,c):
    pre=ColumnTransformer([("n","passthrough",n),("c",OrdinalEncoder(),c)])
    return Pipeline([("p",pre),("m",HistGradientBoostingClassifier(categorical_features=[False]*len(n)+[True]*len(c),max_depth=4,learning_rate=.1,random_state=42))])
def auc(m,n,c): m.fit(tr[n+c],tr.y); return roc_auc_score(te.y,m.predict_proba(te[n+c])[:,1])
res={}
specs=[("Income only",["LogInc"],[]),("Employment only",[],["Employment Status"]),("Income + employment",["LogInc"],["Employment Status"]),
 ("All except income and employment",["Age","Number of Children"],[c for c in cat if c!="Employment Status"]),("All 14 variables",num,cat)]
for n,a,b in specs: res[n]=auc(lin(a,b),a,b)
res["All 14, gradient boosting"]=auc(hgb(num,cat),num,cat)
res["Income + employment, gradient boosting"]=auc(hgb(["LogInc"],["Employment Status"]),["LogInc"],["Employment Status"])
print(json.dumps(res,indent=1))
# AUC ladder chart
r=pd.Series(res).sort_values(); fig,ax=plt.subplots(figsize=(10,5))
ax.barh(r.index,r.values,color=[AMBER if "gradient" in k else TEAL for k in r.index]); ax.set_xlim(.5,max(r.values)+.02); ax.axvline(.5,color=MUTED)
for i,v in enumerate(r.values): ax.text(v+.001,i,f"{v:.3f}",va="center")
ax.set_xlabel("Test ROC-AUC (0.50 = chance)"); ax.set_title("Where the predictive signal comes from",loc="left"); plt.tight_layout(); plt.savefig(OUT+"auc_ladder.png",dpi=160); plt.close()
# income deciles
df["dec"]=pd.qcut(df.Income,10,labels=False); g=df.groupby("dec").agg(rate=("y","mean"),n=("y","size"),lo=("Income","min"),hi=("Income","max"))
g["ci"]=1.96*np.sqrt(g.rate*(1-g.rate)/g.n); print(g.round(3))
fig,ax=plt.subplots(figsize=(10,5)); ax.errorbar(range(1,11),g.rate,yerr=g.ci,marker="o",ms=9,color=TEAL,ecolor=INK,capsize=4,lw=2); ax.axhline(base,color=AMBER,ls="--",lw=2)
ax.yaxis.set_major_formatter(lambda x,_:f"{x:.0%}"); ax.set_xticks(range(1,11)); ax.set_xlabel("Income decile (1 = lowest 10%, 10 = highest 10%)"); ax.set_ylabel("Share with a history")
ax.set_title("History of mental illness falls as income rises, in steps not a line",loc="left"); plt.tight_layout(); plt.savefig(OUT+"income_deciles.png",dpi=160); plt.close()
# income by employment overlap
fig,ax=plt.subplots(figsize=(10,4.8)); cap=df.Income.quantile(.995)
for v,c in [("Employed",TEAL),("Unemployed",AMBER)]:
    ax.hist(df.loc[df["Employment Status"]==v,"Income"].clip(upper=cap),bins=60,alpha=.65,color=c,label=v)
ax.set_xlabel("Income"); ax.set_ylabel("People"); ax.legend(); ax.set_title("Income and employment barely overlap",loc="left"); plt.tight_layout(); plt.savefig(OUT+"income_by_employment.png",dpi=160); plt.close()
ov=df.groupby("Employment Status").Income.agg(["min","median","max"]); print(ov)
lo,hi=df[df["Employment Status"]=="Employed"].Income.min(),df[df["Employment Status"]=="Unemployed"].Income.max(); print("overlap range",lo,hi)
o=df[(df.Income>=lo)&(df.Income<=hi)]; print(len(o),o.groupby("Employment Status").y.agg(["mean","count"]))
res["overlap"]=[float(lo),float(hi),len(o)]
json.dump(res,open(OUT+"extras.json","w"),default=float)
# 2 more: rate by employment within the overlap, by income band
o=o.assign(band=pd.qcut(o.Income,4)); print(o.groupby(["band","Employment Status"],observed=True).y.agg(["mean","count"]).round(3))
