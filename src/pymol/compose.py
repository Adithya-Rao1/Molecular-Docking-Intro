"""Step 11c. 
Compose the three annotated figures (overview + zoom circle, labels, distance) from the PyMOL renders.
Run from work/pymol/ with PYTHONPATH=src (Makefile default behavior).
"""
import json, math, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D
from PIL import Image
V=json.load(open("cx_views.json"))
from redock.common import REF, WORK, load_poses
X,meta=load_poses(WORK if (WORK/"poses_xyz.npy").exists() else REF)
prot={}
for l in open("protein_A.pdb"):
    if l.startswith("ATOM"): prot[(int(l[22:26]),l[12:16].strip())]=np.array([float(l[30:38]),float(l[38:46]),float(l[46:54])])
def proj(view,p):
    R=np.array(view["R"]).T; o=np.array(view["origin"]); d=view["dist"]
    c=R@(np.asarray(p)-o); s=view["H"]/(2*d*math.tan(math.radians(10)))
    return view["W"]/2+c[0]*s, view["H"]/2-c[1]*s, s
# overview ligand mask -> circle
BGC=np.array([20,24,28])
ov=np.array(Image.open("cx_overview.png").convert("RGB")).astype(int)
lg=np.array(Image.open("cx_overview_lig.png").convert("RGB")).astype(int)
m=(np.abs(lg-BGC).sum(2)>40); ys,xs=np.nonzero(m); cx,cy=xs.mean(),ys.mean(); rr=max(xs.max()-xs.min(),ys.max()-ys.min())/2
px,py,_=proj(V["overview"],X[45].mean(0))
print("mask centroid",round(cx),round(cy),"predicted",round(px),round(py))
# crop overview to the protein body
ink=(np.abs(ov-BGC).sum(2)>40); yy,xx=np.nonzero(ink)
pad=40; x0,x1,y0,y1=xx.min()-pad,xx.max()+pad,yy.min()-pad,yy.max()+pad
def figure(zoom_png,zkey,pose_idx,pose_col,title,out,hb=None):
    zv=V[zkey]; zi=np.array(Image.open(zoom_png).convert("RGB"))
    Wf=16; fig=plt.figure(figsize=(Wf,7.4),facecolor='#14181c'); 
    ax=fig.add_axes([0,0,1,1]); ax.set_facecolor("#14181c"); ax.set_xlim(0,1600); ax.set_ylim(740,0); ax.axis("off")
    # overview on the left
    ow=x1-x0; oh=y1-y0; sc=640/oh; ox=40; oy=50
    ax.imshow(ov[y0:y1,x0:x1].astype(np.uint8),extent=[ox,ox+ow*sc,oy+oh*sc,oy],zorder=1,interpolation="lanczos")
    ccx=ox+(cx-x0)*sc; ccy=oy+(cy-y0)*sc; cr=max(rr*sc*1.9,38)
    ax.add_patch(Circle((ccx,ccy),cr,fc=(0.42,0.65,0.9,0.14),ec=(0.42,0.65,0.9,0.9),lw=1.6,ls="-",zorder=3))
    # big zoom circle on the right
    Rz=330; zcx,zcy=1130,370
    # crop zoom image to square around centre, scaled so circle shows ~ the pocket
    h,w=zi.shape[:2]; half=int(h*0.50)
    xm,ym,_=proj(zv,(X[45].mean(0)+prot[(769,"N")])/2)
    cxz=int(np.clip(xm,half,w-half)); cyz=int(np.clip(ym,half,h-half))
    crop=zi[cyz-half:cyz+half,cxz-half:cxz+half]
    yy_,xx_=np.mgrid[:2*half,:2*half]; mask=((xx_-half)**2+(yy_-half)**2<=half**2)
    rgba=np.dstack([crop,np.where(mask,255,0).astype(np.uint8)])
    ax.imshow(rgba,extent=[zcx-Rz,zcx+Rz,zcy+Rz,zcy-Rz],zorder=2,interpolation="lanczos")
    ax.add_patch(Circle((zcx,zcy),Rz,fc="none",ec=(0.42,0.65,0.9,1.0),lw=2.2,zorder=5))
    # dashed tangent connectors
    dx,dy=zcx-ccx,zcy-ccy; L=math.hypot(dx,dy); ux,uy=dx/L,dy/L
    for sgn in (1,-1):
        nx,ny=-uy*sgn,ux*sgn
        a=(ccx+nx*cr,ccy+ny*cr); b=(zcx+nx*Rz,zcy+ny*Rz)
        ax.plot([a[0],b[0]],[a[1],b[1]],ls=(0,(5,4)),color=(0.42,0.65,0.9),lw=1.4,zorder=4)
    # scale of zoom image in figure px
    s_img=zv["H"]/(2*zv["dist"]*math.tan(math.radians(10)))  # image px per A
    f=(2*Rz)/(2*half)  # figure units per image px
    def tofig(p):
        x,y,_=proj(zv,p); return zcx+(x-cxz)*f, zcy+(y-cyz)*f
    # labels
    lab=lambda t,xy,xyt,col="#e6e9ec",ha="left":ax.annotate(t,xy=xy,xytext=xyt,fontsize=12,color=col,ha=ha,va="center",zorder=7,
        arrowprops=dict(arrowstyle="-",color=col,lw=0.9,shrinkA=0,shrinkB=2),bbox=dict(boxstyle="round,pad=0.2",fc="#14181c",ec="none",alpha=0.8))
    pm=tofig(prot[(769,"N")]); pt=tofig(prot[(766,"OG1")]); pk=tofig(prot[(721,"NZ")])
    x=X[pose_idx]; xt=X[45]
    # ligand hinge atom (aromatic N closest to Met769 N)
    ar=[8,10]  # 0-based indices of aromatic N (9,11 in 1-based)
    nn=min(ar,key=lambda i:np.linalg.norm(x[i]-prot[(769,"N")])); pn=tofig(x[nn])
    ax.plot([pm[0],pn[0]],[pm[1],pn[1]],ls=(0,(3,2)),color="#e6e9ec",lw=1.5,zorder=6)
    dist=np.linalg.norm(x[nn]-prot[(769,"N")]); mid=((pm[0]+pn[0])/2,(pm[1]+pn[1])/2)
    ax.text(mid[0]+8,mid[1],f"{dist:.1f} Å",fontsize=11,zorder=8,va="center",color="#e6e9ec",bbox=dict(boxstyle="round,pad=0.15",fc="#14181c",ec="none",alpha=0.9))
    def radial(p,t,col="#7fc27f",off=75):
        v=np.array(p)-np.array([zcx,zcy]); r=np.linalg.norm(v); u=v/r
        tx=np.array(p)+u*off; tx=np.array([zcx,zcy])+u*min(np.linalg.norm(tx-[zcx,zcy]),Rz*0.93)
        ha="left" if u[0]>=0 else "right"
        lab(t,p,(tx[0],tx[1]),col,ha)
    radial(pm,"Met769 (hinge)"); radial(pt,"Thr766 (gatekeeper)"); radial(pk,"Lys721")
    # titles / legend
    ax.text(ox,26,"EGFR kinase domain (PDB 1M17, chain A)",fontsize=13,weight="bold",color="#e6e9ec")
    ax.text(zcx,26,title,fontsize=13,weight="bold",ha="center",color="#e6e9ec")
    ent=[Line2D([],[],color="#7fd1e8",lw=2.2,label="crystal pose (thin)" if pose_idx!=45 else "crystal pose")]
    if pose_idx!=45: ent.append(Line2D([],[],color=pose_col,lw=4,label="docked pose"))
    ent+= [Line2D([],[],color="#7fc27f",lw=3,label="key residues"),Line2D([],[],color="#9aa7b4",lw=3,label="pocket residues (4.5 Å)")]
    ax.legend(handles=ent,loc="lower left",bbox_to_anchor=(0.02,0.0),frameon=False,fontsize=10.5,ncol=2,labelcolor="#e6e9ec")
    fig.savefig(out,dpi=170,facecolor="#14181c"); plt.close(fig)
    return pn,pm
for name,zc,key,idx,col,title in (("crystal","cx_zoom_crystal.png","cx_zoom_crystal.png",45,"#7fd1e8","Crystal pose: hinge hydrogen bond to Met769"),
                  ("vina_top","cx_zoom_vina.png","cx_zoom_vina.png",0,"#ee8470","Vina top-ranked pose (7.9 Å RMSD)"),
                  ("near_native","cx_zoom_near.png","cx_zoom_near.png",38,"#d6b658","Near-native pose (1.2 Å RMSD, Vina rank 3)")):
    figure(zc,key,idx,col,title,f"fig_complex_{name}.png")
print("ok")
