"""Read-only dataset admission audit; never changes active splits or trains."""
from __future__ import annotations

from collections import Counter,defaultdict
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image,ImageDraw
import yaml

from ureter_semis.experiment import atomic_json,sha256
from .convert_dataset import read_yolo_sample

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"runs/data_audit/kidney_v2_20261028"
POPCOUNT=np.array([i.bit_count() for i in range(256)],dtype=np.uint8)


def feature(image):
    h,w=image.shape[:2];crop=image[h//10:h-h//10,w//10:w-w//10]
    rgb=cv2.resize(crop,(32,32),interpolation=cv2.INTER_AREA)
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)
    values=cv2.dct(gray)[:8,:8].flatten()[1:]
    bits=values>np.median(values)
    value=np.uint64(sum(int(b)<<i for i,b in enumerate(bits)))
    return value,rgb


def read_image(path):
    with Image.open(path) as image:return np.asarray(image.convert("RGB"))


def variants(image):
    return [np.ascontiguousarray(np.rot90(a,k)) for a in (image,np.fliplr(image)) for k in range(4)]


def verified_pairs(query_features,reference_features,exclude_self=False):
    """D4-invariant conservative near-copy candidates, not case verification."""
    refs=np.array([f[0] for f in reference_features],dtype=np.uint64)
    pairs=[]
    for qi,features in enumerate(query_features):
        hashes=np.array([f[0] for f in features],dtype=np.uint64)
        xor=np.bitwise_xor(hashes[:,None],refs[None,:])
        distance=POPCOUNT[xor.view(np.uint8).reshape(len(features),len(refs),8)].sum(2)
        for ri in np.flatnonzero(distance.min(0)<=4):
            if exclude_self and qi>=ri:continue
            best=None
            for orientation in np.flatnonzero(distance[:,ri]<=4):
                a=features[orientation][1].astype(float)/255
                b=reference_features[ri][1].astype(float)/255
                mae=float(np.abs(a-b).mean())
                ag=a.mean(2).ravel();bg=b.mean(2).ravel()
                corr=float(np.corrcoef(ag,bg)[0,1]) if ag.std()>1e-6 and bg.std()>1e-6 else 0
                # Brightness-robust second branch is still only a duplicate candidate.
                if mae<=.03 or (corr>=.985 and mae<=.12):
                    row={"query_index":qi,"reference_index":int(ri),"d4_orientation":int(orientation),
                        "phash_distance":int(distance[orientation,ri]),"rgb_mae":mae,"gray_correlation":corr}
                    if best is None or mae<best["rgb_mae"]:best=row
            if best is not None:pairs.append(best)
    return pairs


def family_components(records,references,old_pairs,internal):
    parent={}
    def find(key):
        parent.setdefault(key,key)
        if parent[key]!=key:parent[key]=find(parent[key])
        return parent[key]
    def union(a,b):parent[find(a)]=find(b)
    for row in records:find("v2:"+row["family"])
    for row in references:find("v1:"+row["family"])
    for pair in old_pairs:
        union("v2:"+records[pair["query_index"]]["family"],"v1:"+references[pair["reference_index"]]["family"])
    for pair in internal:
        union("v2:"+records[pair["query_index"]]["family"],"v2:"+records[pair["reference_index"]]["family"])
    active={find("v1:"+r["family"]) for r in references if r["role"]=="active_supervised190"}
    newtrain={find("v2:"+r["family"]) for r in records if r["split"]=="train"}
    counts={}
    for split in ("train","valid","test"):
        subset=[r for r in records if r["split"]==split]
        counts[split]={"images_in_component_with_existing_supervised_training":sum(find("v2:"+r["family"]) in active for r in subset),
            "images_in_component_with_v2_train":sum(find("v2:"+r["family"]) in newtrain for r in subset),
            "fiber_annotated_filename_families":len({r["family"] for r in subset if r["fiber_instances"]})}
    components=defaultdict(lambda:{"v1_families":[],"v2_families":[]})
    for key in parent:
        components[find(key)]["v1_families" if key.startswith("v1:") else "v2_families"].append(key[3:])
    return {"method":"Conservative union of same-version augmentation filename families and near-copy candidate edges; not patient identities",
        "split_counts":counts,"components":list(components.values())}


def main():
    if OUT.exists():raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    source=ROOT/"data/Kidney stones-fiber-.v2i.yolov8"
    old=ROOT/"data/Kidney stones-fiber-.v1i.yolov8"
    manifest_path=ROOT/"datasets/ureter_semis_a1_sup_kidney_only_20261027/manifest.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    yaml_data=yaml.safe_load((source/"data.yaml").read_text(encoding="utf-8"))
    names=dict(enumerate(yaml_data["names"]))
    assert names=={0:"fiber",1:"stone",2:"tissue"}
    role_by_hash={row["image_sha256"]:"active_supervised190" for row in manifest["train"]["labeled"]}
    role_by_hash.update({row["image_sha256"]:"reserved800_not_trained" for row in manifest["reserved_unlabeled"]})
    atomic_json(OUT/"protocol.json",{"source":str(source),"yaml_sha256":sha256(source/"data.yaml"),
        "manifest_sha256":sha256(manifest_path),"mode":"Read-only admission audit, source annotations unreviewed, no active split changes",
        "near_copy_method":"Interior80% pHash63 Hamming<=4 under8 flip/90degree orientations; RGB32 MAE<=0.03 or(gray corr>=0.985 and MAE<=0.12). Candidates, not case identities",
        "training_updates":0})
    records=[];q_features=[];families=defaultdict(list);issues=[];input_hashes={}
    for split in ("train","valid","test"):
        for path in sorted((source/split/"images").glob("*.jpg")):
            label=source/split/"labels"/(path.stem+".txt")
            digest=sha256(path);input_hashes[str(path)]=digest;input_hashes[str(label)]=sha256(label) if label.exists() else None
            image=read_image(path)
            mask,metadata,auxiliary=read_yolo_sample(path,label,names)
            family=path.stem.split(".rf.")[0]
            cls_counts=metadata["annotation_counts"]
            row={"index":len(records),"split":split,"image":str(path),"label":str(label),"family":family,
                "image_sha256":digest,"width":metadata["width"],"height":metadata["height"],
                "fiber_instances":cls_counts.get("fiber",0),"stone_instances":cls_counts.get("stone",0),
                "tissue_instances":cls_counts.get("tissue",0),"fiber_pixels":int((mask==2).sum()),"stone_pixels":int((mask==1).sum()),
                "source_annotation_flags":metadata["review_flags"]}
            records.append(row);families[family].append(row)
            q_features.append([feature(v) for v in variants(image)])
            if not metadata["label_file_present"]:issues.append({"path":str(path),"issue":"missing_label"})
    references=[];reference_features=[]
    for split in ("train","valid","test"):
        for path in sorted((old/split/"images").glob("*.jpg")):
            digest=sha256(path)
            references.append({"image":str(path),"split":split,"image_sha256":digest,
                "role":role_by_hash.get(digest,"outside_active_training_and_reserve"),"family":path.stem.split(".rf.")[0]})
            reference_features.append(feature(read_image(path)))
    old_pairs=verified_pairs(q_features,reference_features)
    for row in old_pairs:
        row.update(query=records[row["query_index"]]["image"],reference=references[row["reference_index"]]["image"],
            query_split=records[row["query_index"]]["split"],reference_role=references[row["reference_index"]]["role"])
    # Original orientations suffice because all eight query orientations are checked.
    internal=verified_pairs(q_features,[f[0] for f in q_features],exclude_self=True)
    cross_split=[r for r in internal if records[r["query_index"]]["split"]!=records[r["reference_index"]]["split"]]
    video_references=[];video_features=[]
    for name in ("video_1","video_2"):
        path=ROOT/"data/videos"/(name+".mp4");cap=cv2.VideoCapture(str(path));index=0
        if not cap.isOpened():raise ValueError(path)
        while True:
            ok,bgr=cap.read()
            if not ok:break
            video_references.append({"video":name,"frame_index":index})
            video_features.append(feature(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)));index+=1
        cap.release()
    video_pairs=verified_pairs(q_features,video_features)
    for row in video_pairs:row.update(query=records[row["query_index"]]["image"],**video_references[row["reference_index"]])
    old_hashes={r["image_sha256"] for r in references}
    conflicts={f:sorted({r["split"] for r in rows}) for f,rows in families.items() if len({r["split"] for r in rows})>1}
    excluded=json.loads((ROOT/"annotations/excluded_training_sources_20261027.json").read_text(encoding="utf-8"))
    blocked=[r["image"] for r in records if r["image_sha256"] in set(excluded["excluded_image_sha256"])]
    components=family_components(records,references,old_pairs,internal)
    components["excluded_stone_new_exact_hash_matches"]=blocked
    atomic_json(OUT/"family_overlap_components.json",components)
    atomic_json(OUT/"source_inventory.json",{"records":records,"input_hashes":input_hashes,"old_references":references})
    atomic_json(OUT/"duplicate_candidates.json",{"v2_vs_v1":old_pairs,"v2_internal":internal,
        "v2_internal_cross_split":cross_split,"v2_vs_validation_videos":video_pairs})
    split_stats={}
    for split in ("train","valid","test"):
        rows=[r for r in records if r["split"]==split]
        split_stats[split]={"images":len(rows),"filename_families":len({r["family"] for r in rows}),
            "with_fiber_annotation":sum(r["fiber_instances"]>0 for r in rows),"with_stone_annotation":sum(r["stone_instances"]>0 for r in rows),
            "with_tissue_annotation":sum(r["tissue_instances"]>0 for r in rows),
            "fiber_instances":sum(r["fiber_instances"] for r in rows),"stone_instances":sum(r["stone_instances"] for r in rows),
            "exact_image_overlap_with_v1":sum(r["image_sha256"] in old_hashes for r in rows)}
        for role in ("active_supervised190","reserved800_not_trained","outside_active_training_and_reserve"):
            split_stats[split]["near_v1_"+role]=len({r["query_index"] for r in old_pairs if r["query_split"]==split and r["reference_role"]==role})
    summary={"source":str(source),"categories":yaml_data["names"],"images":len(records),"filename_families":len(families),
        "split_statistics":split_stats,"image_dimensions":dict(Counter(f"{r['width']}x{r['height']}" for r in records)),
        "family_split_conflicts":conflicts,"v2_internal_cross_split_near_copy_pairs":len(cross_split),
        "near_copy_queries_overlapping_v1":len({r["query_index"] for r in old_pairs}),
        "near_copy_filename_families_overlapping_v1":len({records[r["query_index"]]["family"] for r in old_pairs}),
        "v1_images_compared":len(references),"validation_video_frames_compared":len(video_references),
        "validation_video_near_copy_pairs":len(video_pairs),"annotation_parse_issues":issues,
        "annotations_manually_reviewed":False,"case_independence_verified":False,"active_manifest_changed":False,"training_updates":0,
        "conservative_family_component_overlap":components["split_counts"],"excluded_stone_new_exact_hash_match_count":len(blocked)}
    assert sha256(manifest_path)==json.loads((OUT/"protocol.json").read_text())["manifest_sha256"]
    atomic_json(OUT/"summary.json",summary)
    with (OUT/"inventory.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    # Representative different families: raw/annotation pairs, not automatic review approval.
    chosen=[]
    for split in ("train","valid","test"):
        available=[r for r in records if r["split"]==split and r["fiber_instances"]]
        seen=set();distinct=[]
        for r in available:
            if r["family"] not in seen:distinct.append(r);seen.add(r["family"])
        chosen.extend(distinct[i] for i in np.linspace(0,len(distinct)-1,4,dtype=int))
    canvas=Image.new("RGB",(4*320,6*190),"#202020");draw=ImageDraw.Draw(canvas)
    for index,row in enumerate(chosen):
        image=read_image(Path(row["image"]));mask,_,_=read_yolo_sample(Path(row["image"]),Path(row["label"]),names)
        overlay=image.astype(float)
        for cls,color in ((1,np.array([40,240,70])),(2,np.array([40,150,255]))):
            selection=mask==cls;overlay[selection]=overlay[selection]*.45+color*.55
        for col,array in enumerate((image,overlay.astype(np.uint8))):
            x=(index%2)*640+col*320;y=(index//2)*190
            thumb=Image.fromarray(array).resize((176,176));canvas.paste(thumb,(x+72,y))
            draw.text((x+4,y+176),f"{row['split']} {row['family']} {'raw' if col==0 else 'stone=green fiber=blue'}",fill="white")
    canvas.save(OUT/"annotation_examples.jpg",quality=92)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
