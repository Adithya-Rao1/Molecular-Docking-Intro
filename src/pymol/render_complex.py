from pymol import cmd
import json, numpy as np, math
W,H=2400,1800; Wo,Ho=3200,2400
def hexrgb(h): h=h.lstrip("#"); return [int(h[i:i+2],16)/255 for i in (0,2,4)]
C=dict(bg="#14181c",helix="#6f8fb8",strand="#b79f6e",loop="#6c7884",grey="#5c6874",pocket="#9aa7b4",key="#7fc27f",
       xtal="#7fd1e8",vina="#ee8470",near="#d6b658",N="#3d6bff",O="#ff5252")
for k,v in C.items(): cmd.set_color("m_"+k,hexrgb(v))
cmd.reinitialize()
for k,v in C.items(): cmd.set_color("m_"+k,hexrgb(v))
cmd.load("protein_A.pdb","prot"); cmd.load("crystal.sdf","xtal"); cmd.load("vina_top.sdf","vina"); cmd.load("near_native.sdf","near")
for k,v in dict(ray_opaque_background=1,antialias=2,ray_trace_mode=0,two_sided_lighting=1,ambient=0.5,reflect=0.45,direct=0.45,ray_trace_gain=0.0,depth_cue=0,specular=0.15,ray_shadow=0,orthoscopic=1,cartoon_fancy_helices=1,cartoon_transparency=0.0,field_of_view=20).items(): cmd.set(k,v)
cmd.bg_color("m_bg"); cmd.hide("everything"); cmd.remove("elem H"); cmd.dss("prot")
def arr(sel): return np.array(cmd.get_coords(sel))
L=arr("xtal"); P=arr("prot and name CA"); lc=L.mean(0); pc=P.mean(0)
q=L-lc; n=np.linalg.svd(q)[2][2]
z=(lc-pc); z/=np.linalg.norm(z); y=n-np.dot(n,z)*z; y/=np.linalg.norm(y); x=np.cross(y,z)
R=np.vstack([x,y,z])
def setv(Rm,origin,dist,slab=(-300,300)):
    cmd.set_view(list(Rm.flatten())+[0,0,-dist]+list(origin)+[dist+slab[0],dist+slab[1],0])
def info(Rm,origin,dist,w,h): return dict(R=Rm.tolist(),origin=list(map(float,origin)),dist=float(dist),W=w,H=h)
def lig_style(show,ghost=None):
    for o,c in (("xtal","m_xtal"),("vina","m_vina"),("near","m_near")):
        if o in show or o==ghost:
            cmd.show("sticks",o); cmd.color(c,o+" and elem C"); cmd.color("m_N",o+" and elem N"); cmd.color("m_O",o+" and elem O")
            cmd.set_bond("stick_radius",0.07 if o==ghost else 0.26,o)
def color_prot_ss():
    cmd.color("m_loop","prot"); cmd.color("m_helix","prot and ss h"); cmd.color("m_strand","prot and ss s")
out={}

# overview
color_prot_ss(); cmd.show("cartoon","prot"); lig_style(["xtal"]); cmd.set_bond("stick_radius",0.34,"xtal")
ext=np.abs((P-pc)@R.T); hgt=max(2*ext[:,1].max(),2*ext[:,0].max()*Ho/Wo)+8
d_ov=hgt/(2*math.tan(math.radians(10))); cen=(P.max(0)+P.min(0))/2; setv(R,cen,d_ov)
cmd.ray(Wo,Ho); cmd.png("cx_overview.png"); out["overview"]=info(R,cen,d_ov,Wo,Ho)
cmd.hide("everything","prot"); cmd.ray(Wo,Ho); cmd.png("cx_overview_lig.png")

# zoom (face-on, hinge at the bottom)
cmd.select("pocket","byres (prot within 4.5 of xtal)")
cmd.select("keyres","prot and resi 769+766+721")
cmd.select("keyseg","prot and resi 766-770+719-722")   # labelled segments, highlighted
cmd.select("region","byres (prot within 15 of xtal)")
met=arr("prot and resi 769 and name N")[0]
zz=n.copy()
if np.dot(zz,(lc-pc))<0: zz=-zz
hd=met-lc; hd-=np.dot(hd,zz)*zz; hd/=np.linalg.norm(hd); yy=-hd; xx=np.cross(yy,zz); RZ=np.vstack([xx,yy,zz])
def zoom_scene(show,ghost,fn):
    cmd.hide("everything")
    cmd.show("cartoon","region"); cmd.color("m_grey","region"); cmd.set("cartoon_transparency",0.15,"region")
    cmd.color("m_key","keyseg"); cmd.set("cartoon_transparency",0.0,"keyseg")
    cmd.show("sticks","pocket"); cmd.set_bond("stick_radius",0.09,"pocket")
    cmd.color("m_pocket","pocket and elem C"); cmd.color("m_N","pocket and elem N"); cmd.color("m_O","pocket and elem O")
    cmd.color("m_key","keyres and elem C")
    cmd.hide("sticks","pocket and name C+O+N+CA and not resi 769+766+721")
    lig_style(show,ghost)
    Hz=26.0; d=Hz/(2*math.tan(math.radians(10)))
    # clip away what lies in front of the ligand plane so the ligand is not hidden
    cmd.set_view(list(RZ.flatten())+[0,0,-d]+list(lc)+[d-2.2,d+40,0])
    cmd.ray(W,H); cmd.png(fn); out[fn]=info(RZ,lc,d,W,H)
zoom_scene(["xtal"],None,"cx_zoom_crystal.png")
zoom_scene(["vina"],"xtal","cx_zoom_vina.png")
zoom_scene(["near"],"xtal","cx_zoom_near.png")
json.dump(out,open("cx_views.json","w")); print("done")
