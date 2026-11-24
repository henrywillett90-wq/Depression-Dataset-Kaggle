import os; os.makedirs("deck_figs", exist_ok=True); os.makedirs("slide_figs", exist_ok=True)
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
INK,TEAL,AMBER,MUTED,GRID="#16213A","#1F7A7A","#E0A030","#4A566E","#E3E0D8"
plt.rcParams.update({"font.size":17,"axes.edgecolor":MUTED,"axes.labelcolor":INK,"text.color":INK,"xtick.color":INK,"ytick.color":INK,"axes.spines.top":False,"axes.spines.right":False,"axes.titlesize":19,"axes.titleweight":"bold"})
t=pd.read_csv("deck_figs/tests.csv")
cat=t[t.test=="Chi-square"].sort_values("effect_size"); num=t[t.test=="Mann-Whitney U"].sort_values("effect_size")
fig,axes=plt.subplots(1,2,figsize=(17,6.8),gridspec_kw={"width_ratios":[1.9,1]})
for ax,d,c,ttl,xl in [(axes[0],cat,TEAL,"Categorical variables\nCramér’s V, from chi-square","Cramér’s V  (0 = no link, 1 = perfect link)"),
                      (axes[1],num,AMBER,"Numeric variables\nrank-biserial r, from Mann-Whitney U","|rank-biserial r|  (0 = no difference)")]:
    y=np.arange(len(d)); ax.barh(y,d.effect_size,color=c,height=.62)
    for yi,v in zip(y,d.effect_size): ax.text(v+.003,yi,f"{v:.3f}",va="center",fontsize=16,fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(d.variable.replace({"Family History of Depression":"Family history","History of Substance Abuse":"Substance abuse","Chronic Medical Conditions":"Chronic conditions","Marital Status":"Marital status","Employment Status":"Employment","Education Level":"Education","Physical Activity Level":"Physical activity"}),fontsize=16)
    ax.set_xlim(0,.25); ax.axvline(.05,color=MUTED,ls="--",lw=1.5); ax.set_xlabel(xl); ax.set_title(ttl,loc="left",fontsize=17); ax.grid(axis="x",color=GRID); ax.set_axisbelow(True)
plt.tight_layout(); plt.savefig("slide_figs/effects_split.png",dpi=130); plt.close(); print(cat[["variable","effect_size"]].tail(3), num[["variable","effect_size"]])
