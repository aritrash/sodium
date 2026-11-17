from __future__ import annotations
import hashlib, json, subprocess, sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from PIL import Image
from scipy.interpolate import PchipInterpolator

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def image_files(source: Path):
    if source.is_file():
        return [source] if source.suffix.lower() in IMAGE_EXTENSIONS else []
    return sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)

def inventory(source: Path, output: Path):
    paths = image_files(source); output.mkdir(parents=True, exist_ok=True); rows = []
    for p in paths:
        with Image.open(p) as im:
            rows.append({"relative_path": str(p.relative_to(source)) if source.is_dir() else p.name, "filename": p.name, "width_px": im.width, "height_px": im.height, "format": im.format, "mode": im.mode, "bytes": p.stat().st_size, "sha256": sha256(p), "classification": "UNCLASSIFIED", "review_status": "PENDING"})
    pd.DataFrame(rows).to_csv(output / "image_inventory.csv", index=False)
    print(f"Inventoried {len(rows)} images -> {output / 'image_inventory.csv'}")

def classify_filename(filename: str) -> dict:
    stem = Path(filename).stem; lower = stem.lower()
    result = {"classification":"UNCLASSIFIED","plot_family":"unknown","x_quantity":"","y_quantity":"","x_scale":"linear","y_scale":"linear","filename_measurement_suffix":"","classification_method":"filename_rules_v1","review_status":"PENDING"}
    if "3dg" in lower or "3dp" in lower:
        result.update(classification="PLOT_3D", plot_family="3d_characteristic", review_status="REFERENCE_ONLY"); return result
    if "main_panel" in lower:
        result.update(classification="INSTRUMENT_UI", plot_family="instrument_interface", review_status="REFERENCE_ONLY"); return result
    body = lower.removeprefix("nos_rca_12ax7a_")
    if "_vhtr_" in body:
        body, suffix = body.split("_vhtr_", 1); result["filename_measurement_suffix"] = "Vhtr_" + suffix
    if "_vs_" not in body:
        result.update(classification="UNCLASSIFIED_2D_OR_OTHER"); return result
    left, right = body.split("_vs_", 1); lt, rt = left.split("_"), right.split("_")
    if lt[0] == "log": result["y_scale"] = "log"; lt = lt[1:]
    if rt[0] == "log": result["x_scale"] = "log"; rt = rt[1:]
    result["y_quantity"] = "_".join(lt); result["x_quantity"] = "_".join(rt)
    y = result["y_quantity"]
    if y == "ik": result.update(classification="PLOT_2D_CHARACTERISTIC", plot_family="current_characteristic")
    elif y in {"gm","gp","gs"}: result.update(classification="PLOT_2D_PARAMETER", plot_family="conductance_parameter")
    elif y in {"rp","rm","rs"}: result.update(classification="PLOT_2D_PARAMETER", plot_family="resistance_parameter")
    else: result.update(classification="PLOT_2D_PARAMETER", plot_family="other_parameter")
    result["review_status"] = "PROFILE_REQUIRED"; return result

def classify_inventory(source: Path, inventory_path: Path, output: Path):
    df = pd.read_csv(inventory_path)
    if "filename" not in df.columns: raise ValueError("Inventory must contain a filename column")
    classified = pd.DataFrame([classify_filename(name) for name in df["filename"]])
    merged = pd.concat([df.drop(columns=[c for c in classified.columns if c in df.columns]), classified], axis=1)
    output.mkdir(parents=True, exist_ok=True); dest = output / "classified_image_inventory.csv"; merged.to_csv(dest, index=False)
    summary = merged.groupby(["classification","plot_family"], dropna=False).size().reset_index(name="image_count")
    summary.to_csv(output / "classification_summary.csv", index=False)
    print(f"Classified {len(merged)} images -> {dest}"); print(summary.to_string(index=False))

def load_profile(profile_path: Path):
    with profile_path.open() as f: profile = json.load(f)
    required = {"profile_id","graph_type","roi","x_axis","y_axis","curves"}
    missing = required - profile.keys()
    if missing: raise ValueError(f"Profile missing keys: {sorted(missing)}")
    return profile

def crop_and_adjust_ticks(image_path: Path, roi, ticks, crop_path: Path):
    with Image.open(image_path) as im:
        x0,y0,x1,y1 = map(int,roi)
        if not (0 <= x0 < x1 <= im.width and 0 <= y0 < y1 <= im.height): raise ValueError(f"ROI {roi} outside image dimensions {im.size}: {image_path}")
        im.crop((x0,y0,x1,y1)).save(crop_path)
    return [[int(x)-x0,int(y)-y0] for x,y in ticks]

def validate_curve(df: pd.DataFrame, x_axis, y_axis):
    required = {"x","y","x_px","y_px"}
    if not required.issubset(df.columns): return {"status":"REVIEW_REQUIRED","reasons":["Missing expected LinePilot columns"],"points":len(df)}
    reasons=[]; xlo,xhi=sorted([float(x_axis["minimum"]),float(x_axis["maximum"])]); ylo,yhi=sorted([float(y_axis["minimum"]),float(y_axis["maximum"])])
    if df[["x","y"]].isna().any().any(): reasons.append("NaN coordinates")
    if ((df["x"]<xlo-1e-6)|(df["x"]>xhi+1e-6)).any(): reasons.append("X outside calibrated axis range")
    if ((df["y"]<ylo-1e-6)|(df["y"]>yhi+1e-6)).any(): reasons.append("Y outside calibrated axis range")
    if df.duplicated(subset=["x","y"]).any(): reasons.append("Duplicate coordinate pairs")
    if len(df)<20: reasons.append("Fewer than 20 points")
    return {"status":"REVIEW_REQUIRED" if reasons else "EXTRACTION_COMPLETE","reasons":reasons,"points":len(df)}

def run_profile(source: Path, profile_path: Path, linepilot: Path, output: Path, linepilot_python: Path | None = None):
    profile=load_profile(profile_path); images=image_files(source)
    if not images: raise SystemExit(f"No supported images found: {source}")
    if not linepilot.is_file(): raise SystemExit(f"LinePilot script not found: {linepilot}")
    interpreter=linepilot_python or Path(sys.executable)
    if not interpreter.is_file(): raise SystemExit(f"LinePilot Python interpreter not found: {interpreter}")
    output.mkdir(parents=True,exist_ok=True); extracted=output/"linepilot_raw"; crops=output/"prepared"; reports=output/"reports"
    for d in (extracted,crops,reports): d.mkdir(exist_ok=True)
    rows=[]
    for image_path in images:
        image_id=image_path.stem; crop=crops/f"{image_id}_crop.png"; ticks=crop_and_adjust_ticks(image_path,profile["roi"],profile["ticks"],crop)
        for curve in profile["curves"]:
            label=curve["id"]; dest=extracted/image_id/label; dest.mkdir(parents=True,exist_ok=True)
            tick_string=",".join(f"[{x},{y}]" for x,y in ticks)
            cmd=[str(interpreter),str(linepilot),str(crop),"--color",",".join(map(str,curve["rgb"])),"--axis",f'{profile["x_axis"]["minimum"]},{profile["x_axis"]["maximum"]},{profile["y_axis"]["minimum"]},{profile["y_axis"]["maximum"]}',"--ticks",tick_string,"--out",str(dest)]
            started=datetime.now(timezone.utc).isoformat(); proc=subprocess.run(cmd,capture_output=True,text=True)
            (dest/"linepilot_stdout.log").write_text(proc.stdout); (dest/"linepilot_stderr.log").write_text(proc.stderr)
            csvs=sorted(dest.glob("*_digitized_points.csv")); status="EXTRACTION_FAILED"; reasons=[f"LinePilot exit code {proc.returncode}"]; count=0; csv_path=""
            if proc.returncode==0 and csvs:
                csv_path=str(csvs[0]); qc=validate_curve(pd.read_csv(csvs[0]),profile["x_axis"],profile["y_axis"]); status,reasons,count=qc["status"],qc["reasons"],qc["points"]
            rows.append({"image":str(image_path),"image_sha256":sha256(image_path),"profile_id":profile["profile_id"],"curve_id":label,"rgb":",".join(map(str,curve["rgb"])),"linepilot_command":subprocess.list2cmdline(cmd),"started_utc":started,"exit_code":proc.returncode,"csv_path":csv_path,"points":count,"status":status,"reasons":"; ".join(reasons)})
            print(f"{image_path.name} / {label}: {status} ({count} points)")
    pd.DataFrame(rows).to_csv(reports/"extraction_report.csv",index=False)
    (output/"run_manifest.json").write_text(json.dumps({"profile_id":profile["profile_id"],"generated_utc":datetime.now(timezone.utc).isoformat(),"source":str(source.resolve()),"linepilot":str(linepilot.resolve()),"linepilot_python":str(interpreter.resolve()),"profile":str(profile_path.resolve()),"policy":"Extraction output is never automatically promoted to validated empirical data."},indent=2))
    print(f"Report: {reports/'extraction_report.csv'}")

def review_curve(input_csv: Path, output_dir: Path, x_col: str, y_col: str, accept: bool, reviewer: str):
    df=pd.read_csv(input_csv)
    if not {x_col,y_col}.issubset(df.columns): raise ValueError(f"Input must contain columns {x_col!r} and {y_col!r}")
    out=output_dir/(input_csv.stem+"_candidate.csv"); output_dir.mkdir(parents=True,exist_ok=True)
    data=df[[x_col,y_col]].copy()
    data[x_col]=pd.to_numeric(data[x_col],errors="coerce"); data[y_col]=pd.to_numeric(data[y_col],errors="coerce")
    reasons=[]
    if data.isna().any().any(): reasons.append("Missing or non-numeric coordinates")
    if len(data)<3: reasons.append("Fewer than three points")
    if data[x_col].duplicated().any(): reasons.append("Duplicate X values; resolve explicitly before interpolation")
    if not data[x_col].is_monotonic_increasing: reasons.append("X values are not strictly increasing")
    if accept and reasons: raise ValueError("Cannot manually validate: " + "; ".join(reasons))
    data.to_csv(out,index=False)
    status="MANUALLY_VALIDATED" if accept else "REVIEW_REQUIRED"
    sidecar={"source_csv":str(input_csv.resolve()),"source_sha256":sha256(input_csv),"output_csv":str(out.resolve()),"created_utc":datetime.now(timezone.utc).isoformat(),"review_status":status,"reviewer":reviewer if accept else None,"checks":reasons,"manual_acceptance_flag":bool(accept),"note":"Manual acceptance is an explicit user assertion; the tool does not judge physical correctness."}
    (output_dir/(input_csv.stem+"_validation.json")).write_text(json.dumps(sidecar,indent=2))
    print(f"{status}: {out}")
    if reasons: print("Review flags: " + "; ".join(reasons))

def build_pchip(input_dir: Path, output_dir: Path, step: float, x_col: str, y_col: str):
    if step<=0: raise ValueError("PCHIP step must be positive")
    output_dir.mkdir(parents=True,exist_ok=True); files=sorted(input_dir.glob("*_candidate.csv"))
    if not files: raise SystemExit(f"No *_candidate.csv files found in {input_dir}")
    for path in files:
        sidecar=input_dir/(path.stem.removesuffix("_candidate")+"_validation.json")
        if not sidecar.exists(): print(f"SKIP {path.name}: validation sidecar missing"); continue
        meta=json.loads(sidecar.read_text())
        if meta.get("review_status")!="MANUALLY_VALIDATED": print(f"SKIP {path.name}: not manually validated"); continue
        df=pd.read_csv(path)
        if not {x_col,y_col}.issubset(df.columns): print(f"SKIP {path.name}: missing {x_col}/{y_col}"); continue
        df=df[[x_col,y_col]].apply(pd.to_numeric,errors="coerce").dropna().sort_values(x_col)
        if df[x_col].duplicated().any() or len(df)<3: print(f"SKIP {path.name}: duplicate X or fewer than 3 points"); continue
        x=df[x_col].to_numpy(float); y=df[y_col].to_numpy(float)
        if np.any(np.diff(x)<=0): print(f"SKIP {path.name}: X must be strictly increasing"); continue
        interpolator=PchipInterpolator(x,y,extrapolate=False); grid=np.arange(x[0],x[-1]+step*1e-9,step)
        values=interpolator(grid); out=pd.DataFrame({x_col:grid,y_col:values,"data_origin":"PCHIP_INTERPOLATED","source_file":path.name,"source_sha256":meta["source_sha256"]})
        dest=output_dir/f"{path.stem.removesuffix('_candidate')}_pchip_{step:g}.csv"; out.to_csv(dest,index=False)
        report={"source_candidate":str(path.resolve()),"source_sha256":meta["source_sha256"],"validation_status":meta["review_status"],"method":"PCHIP","step":step,"x_domain":[float(x[0]),float(x[-1])],"extrapolation":False,"input_points":len(x),"output_points":len(out),"generated_utc":datetime.now(timezone.utc).isoformat()}
        dest.with_suffix(".json").write_text(json.dumps(report,indent=2)); print(f"Generated {len(out)} points -> {dest}")


def _parse_index_list(value: str | None) -> set[int]:
    if not value: return set()
    result=set()
    for token in value.split(","):
        token=token.strip()
        if not token: continue
        if "-" in token:
            a,b=token.split("-",1); start,end=int(a),int(b)
            if start<0 or end<start: raise ValueError(f"Invalid index range: {token}")
            result.update(range(start,end+1))
        else:
            index=int(token)
            if index<0: raise ValueError(f"Point indices must be zero-based and non-negative: {index}")
            result.add(index)
    return result

def _parse_x_ranges(value: str | None) -> list[tuple[float,float]]:
    if not value: return []
    ranges=[]
    for token in value.split(","):
        a,b=map(float,token.strip().split(":",1))
        if b<a: a,b=b,a
        ranges.append((a,b))
    return ranges

def clean_curve(input_csv: Path, output_dir: Path, x_col: str, y_col: str, drop_indexes: str | None, drop_x_ranges: str | None):
    df=pd.read_csv(input_csv)
    if not {x_col,y_col}.issubset(df.columns): raise ValueError(f"Input must contain columns {x_col!r} and {y_col!r}")
    original_count=len(df)
    index_set=_parse_index_list(drop_indexes)
    invalid=sorted(i for i in index_set if i>=original_count)
    if invalid: raise ValueError(f"Point indices outside input range 0..{original_count-1}: {invalid}")
    x_ranges=_parse_x_ranges(drop_x_ranges)
    x=pd.to_numeric(df[x_col],errors="coerce")
    remove_mask=df.index.isin(index_set)
    range_matches=[]
    for lo,hi in x_ranges:
        mask=x.between(lo,hi,inclusive="both").fillna(False)
        range_matches.append({"minimum":lo,"maximum":hi,"matched_original_row_indices":[int(i) for i in df.index[mask]]})
        remove_mask |= mask
    removed_indices=[int(i) for i in df.index[remove_mask]]
    if len(removed_indices)==original_count: raise ValueError("Cleaning would remove every point; refusing to write an empty curve")
    cleaned=df.loc[~remove_mask].copy()
    output_dir.mkdir(parents=True,exist_ok=True)
    output_path=output_dir/(input_csv.stem+"_cleaned.csv")
    cleaned.to_csv(output_path,index=False)
    audit={
        "operation":"manual_point_exclusion",
        "source_csv":str(input_csv.resolve()),
        "source_sha256":sha256(input_csv),
        "output_csv":str(output_path.resolve()),
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "indexing":"zero-based original CSV row indices; header excluded",
        "original_point_count":original_count,
        "retained_point_count":len(cleaned),
        "removed_point_count":len(removed_indices),
        "removed_original_row_indices":removed_indices,
        "explicit_index_selection":sorted(index_set),
        "x_ranges":range_matches,
        "policy":"No automatic outlier removal or smoothing was performed. Original source CSV remains untouched."
    }
    audit_path=output_dir/(input_csv.stem+"_cleaning_audit.json")
    audit_path.write_text(json.dumps(audit,indent=2))
    print(f"Cleaned candidate: {output_path}")
    print(f"Retained {len(cleaned)}/{original_count}; removed {len(removed_indices)} points")
    print(f"Audit log: {audit_path}")
