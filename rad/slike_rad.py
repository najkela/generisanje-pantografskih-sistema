import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
U="/mnt/user-data/uploads/generisanje-pantografskih-sistema"
D=json.load(open(f"{U}/results/analiza/podaci_za_rad.json"))
BI="#2a78d6"; BA="#eb6834"; INK="#0b0b0b"; MUT="#52514e"; GRID="#e4e3df"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":6.2,"axes.edgecolor":MUT,"axes.labelcolor":INK,
 "xtick.color":MUT,"ytick.color":MUT,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,
 "grid.color":GRID,"grid.linewidth":0.5,"legend.frameon":False,"savefig.dpi":300,"axes.titlesize":8})
IME={"ellipse":"елипса","egg":"јаје","limacon":"лимасон","cassini":"Касинијев овал","superellipse":"суперелипса"}
def krive(mat,k,m):
    C=[np.array(d["calls"]) for d in D if d["mat"]==mat and d["curve"]==k and d["method"]==m]
    return C[0][:,0], np.array([c[:,1] for c in C])
fig,ax=plt.subplots(1,2,figsize=(3.4,1.45))
for a,(mat,k,nas) in zip(ax,[("e1_n6","limacon","лимасон, n = 6"),("e1_n7","egg","јаје, n = 7")]):
    for m,col,lab in (("baseline",BA,"МГА"),("bilevel",BI,"ДН")):
        x,Y=krive(mat,k,m)
        a.fill_between(x/1000,Y.min(0),Y.max(0),color=col,alpha=0.15,lw=0)
        a.plot(x/1000,np.median(Y,0),color=col,lw=1.1,label=lab)
    a.set_yscale("log"); a.set_title(nas); a.set_xlabel("позиви (хиљ.)")
ax[0].set_ylabel("Chamfer грешка"); ax[0].legend(loc="upper right",fontsize=6)
fig.tight_layout(pad=0.25); fig.savefig("slike/konvergencija.png"); plt.close(fig)

fig,a=plt.subplots(figsize=(3.3,3.1))
x=[];y=[]
for mat in ("e2","e1_n6","e1_n7","e1_n8"):
    b={(d["curve"],d["seed"]):d["final"] for d in D if d["mat"]==mat and d["method"]=="baseline" and d["curve"]!="circle"}
    for d in D:
        if d["mat"]==mat and d["method"]=="bilevel" and (d["curve"],d["seed"]) in b:
            x.append(b[(d["curve"],d["seed"])]); y.append(d["final"])
x=np.array(x); y=np.array(y); lo,hi=min(x.min(),y.min())/2,max(x.max(),y.max())*2
a.plot([lo,hi],[lo,hi],color=MUT,lw=0.9,ls="--")
a.scatter(x,y,s=13,color=BI,edgecolor="white",linewidth=0.5,zorder=3)
a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(lo,hi); a.set_ylim(lo,hi); a.set_aspect("equal")
a.set_xlabel("коначна грешка, МГА"); a.set_ylabel("коначна грешка, ДН")
a.text(0.96,0.06,f"{(y<x).sum()}/{len(x)} испод\nдијагонале",transform=a.transAxes,ha="right",va="bottom",fontsize=7.5,color=INK)
fig.tight_layout(pad=0.3); fig.savefig("slike/parovi.png"); plt.close(fig)
print("ок", (y<x).sum(), len(x))
