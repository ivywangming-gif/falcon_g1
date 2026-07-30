#!/usr/bin/env python3
"""Validate CP1.5 videos, create result-card montages, and write SHA manifest."""

from __future__ import annotations
import argparse,hashlib,json,os,subprocess
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt

REPO=Path(__file__).resolve().parents[1]
TOOLS=REPO/".cache/tools/ffprobe_ubuntu/extracted/usr/bin";LIB=REPO/".cache/tools/ffprobe_ubuntu/extracted/usr/lib/x86_64-linux-gnu"
ENV=dict(os.environ,LD_LIBRARY_PATH=f"{LIB}:{LIB/'pulseaudio'}")


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()
def probe(path):
    p=subprocess.run([str(TOOLS/"ffprobe"),"-v","error","-show_entries","format=duration,size","-show_entries","stream=codec_name,width,height,r_frame_rate","-of","json",str(path)],env=ENV,text=True,capture_output=True,check=True)
    return json.loads(p.stdout)
def card(path,title,command,measured,heading,survival,precision):
    fig=plt.figure(figsize=(6.4,4.8),dpi=100,facecolor="#10151d");fig.text(.06,.87,title,color="white",fontsize=20,weight="bold")
    lines=[f"desired command  {command}",f"measured mean    {measured}",f"final heading drift  {heading:+.4f} rad",f"SURVIVAL  {'PASS' if survival else 'FAIL'}",f"PRECISION {'PASS' if precision else 'FAIL'}","world XY trace is shown in the following rollout inset"]
    for i,line in enumerate(lines):fig.text(.08,.70-i*.105,line,color=("#56e39f" if "PASS" in line else "#ff6b6b" if "FAIL" in line else "white"),fontsize=14)
    plt.axis("off");path.parent.mkdir(parents=True,exist_ok=True);fig.savefig(path);plt.close(fig)
def montage(name,items,timestamp):
    temp=REPO/".cache/tmp"/f"cp1_5_montage_{name}_{timestamp}";temp.mkdir(parents=True,exist_ok=True);concat=[]
    for i,item in enumerate(items):
        image=temp/f"card_{i:02d}.png";clip=temp/f"card_{i:02d}.mp4";card(image,**item["card"])
        subprocess.run([str(TOOLS/"ffmpeg"),"-y","-loop","1","-i",str(image),"-t","1.5","-r","10","-c:v","mpeg4","-q:v","4","-pix_fmt","yuv420p",str(clip)],env=ENV,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=True)
        concat += [clip,Path(item["video"])]
    listing=temp/"concat.txt";listing.write_text("".join(f"file '{p}'\n" for p in concat))
    output=Path(f"/root/autodl-tmp/FALCON_CP1_{'6_REPOSITION' if name=='REPOSITION' else '5_'+name}_{timestamp}.mp4")
    subprocess.run([str(TOOLS/"ffmpeg"),"-y","-f","concat","-safe","0","-i",str(listing),"-vf","fps=10,scale=640:480,format=yuv420p","-c:v","libx264","-preset","veryfast","-crf","24",str(output)],env=ENV,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=True)
    return output
def main():
    p=argparse.ArgumentParser();p.add_argument("--timestamp",required=True);a=p.parse_args();constant=json.loads((REPO/"artifacts/cp1_5/constant_command_summary.json").read_text());groups=defaultdict(list)
    for r in constant:groups[r["case"]].append(r)
    items=[]
    for case,rows in sorted(groups.items()):
        row=next(r for r in rows if r["seed"]==101);v3=json.loads((Path(row["run_root"])/"precision_evaluation_v3.json").read_text());cmd=v3["command"];err=v3["error_statistics"]
        measured=[cmd["vx"]+err["vx_body"]["signed_mean_error"],cmd["vy"]+err["vy_body"]["signed_mean_error"],cmd["yaw_rate"]+err["yaw_rate_body"]["signed_mean_error"]]
        items.append({"video":row["video"],"card":{"title":case,"command":[cmd["vx"],cmd["vy"],cmd["yaw_rate"]],"measured":[round(x,4) for x in measured],"heading":v3["trajectory"]["heading_drift_final"],"survival":all(x["survival_pass"] for x in rows),"precision":all(x["precision_pass"] for x in rows)}})
    montages=[montage("LOW_SPEED_MATRIX",items,a.timestamp)]
    push_path=REPO/"artifacts/cp1_5/push_ready_summary.json"
    if push_path.is_file():
        push=json.loads(push_path.read_text());pg=defaultdict(list)
        for r in push["rollouts"]:pg[r["case"]].append(r)
        pitems=[]
        for case,rows in sorted(pg.items()):
            row=next(r for r in rows if r["seed"]==101);v3=json.loads((Path(row["run_root"])/"precision_evaluation_v3.json").read_text());cmd=v3["command"];err=v3["error_statistics"]
            pitems.append({"video":row["video"],"card":{"title":"push_ready_"+case,"command":[cmd["vx"],cmd["vy"],cmd["yaw_rate"]],"measured":[round(cmd["vx"]+err["vx_body"]["signed_mean_error"],4),round(cmd["vy"]+err["vy_body"]["signed_mean_error"],4),round(cmd["yaw_rate"]+err["yaw_rate_body"]["signed_mean_error"],4)],"heading":v3["trajectory"]["heading_drift_final"],"survival":all(x["survival_pass"] for x in rows),"precision":all(x["precision_pass"] for x in rows)}})
        montages.append(montage("PUSH_READY",pitems,a.timestamp))
    external_path=REPO/"artifacts/cp1_5/external_load_summary.json"
    if external_path.is_file() and json.loads(external_path.read_text()).get("rollouts"):
        ext=json.loads(external_path.read_text());eg=defaultdict(list)
        for r in ext["rollouts"]:eg[(r["mode"],r["load"])].append(r)
        eitems=[]
        for key,rows in sorted(eg.items()):
            row=next(r for r in rows if r["seed"]==101);v3=json.loads((Path(row["run_root"])/"precision_evaluation_v3.json").read_text());cmd=v3["command"];err=v3["error_statistics"]
            eitems.append({"video":row["video"],"card":{"title":"force_"+"_".join(key),"command":[cmd["vx"],cmd["vy"],cmd["yaw_rate"]],"measured":[round(cmd["vx"]+err["vx_body"]["signed_mean_error"],4),round(cmd["vy"]+err["vy_body"]["signed_mean_error"],4),round(cmd["yaw_rate"]+err["yaw_rate_body"]["signed_mean_error"],4)],"heading":v3["trajectory"]["heading_drift_final"],"survival":all(x["survival_pass"] for x in rows),"precision":all(x["precision_pass"] for x in rows)}})
        montages.append(montage("FORCE_LOADS",eitems,a.timestamp))
    videos=sorted({Path(r["video"]) for r in constant}|{p for p in montages})
    if push_path.is_file():videos+=sorted({Path(r["video"]) for r in json.loads(push_path.read_text())["rollouts"]})
    if external_path.is_file():videos+=sorted({Path(r["video"]) for r in json.loads(external_path.read_text()).get("rollouts",[])})
    videos=sorted(set(videos))
    entries=[{"path":str(v),"sha256":sha(v),"ffprobe":probe(v)} for v in videos]
    manifest={"phase":"CP1_5","timestamp":a.timestamp,"montages":[str(x) for x in montages],"videos":entries}
    target=Path("/root/autodl-tmp/FALCON_VIDEO_MANIFEST.json");target.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n");(REPO/"artifacts/cp1_5/video_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    latest=Path("/root/autodl-tmp/FALCON_CP1_5_LATEST.mp4");latest.unlink(missing_ok=True);latest.symlink_to(montages[0]);print(json.dumps(manifest,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
