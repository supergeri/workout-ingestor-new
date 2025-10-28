# ---------- Imports ----------
from fastapi import FastAPI, UploadFile, File, Form, Body, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
from PIL import Image
import pytesseract
import io, re, os, glob, shutil, subprocess, tempfile

# Optional dependency (guarded)
try:
    import yt_dlp  # type: ignore
except ImportError:
    yt_dlp = None  # we'll error nicely in /ingest/url if missing

# FIT export (robust for fit-tool 0.9.13+)
FitFileBuilder = None
FileType = None
WorkoutMessage = None
WorkoutStepMessage = None
Sport = None
DUR = None
TGT = None
try:
    from fit_tool.fit_file_builder import FitFileBuilder  # type: ignore
    from fit_tool.profile.messages.workout_message import WorkoutMessage  # type: ignore
    from fit_tool.profile.messages.workout_step_message import WorkoutStepMessage  # type: ignore
    from fit_tool.profile import profile_type as p  # type: ignore
    FileType = getattr(p, "FileType", None) or type("FileType", (), {"WORKOUT": 5})
    Sport = getattr(p, "Sport", None)
    DUR = getattr(p, "WktStepDuration", None) or getattr(p, "WorkoutStepDuration", None)
    TGT = getattr(p, "WktStepTarget", None) or getattr(p, "WorkoutStepTarget", None)
except Exception as e:
    print(f"[WARN] FIT export disabled: {e}")
    FitFileBuilder = None

# ---------- App ----------
app = FastAPI(title="Workout Ingestor API")

# ---------- Data models ----------
class Exercise(BaseModel):
    name: str
    sets: Optional[int] = None
    reps: Optional[int] = None
    reps_range: Optional[str] = None
    duration_sec: Optional[int] = None
    rest_sec: Optional[int] = None
    distance_m: Optional[int] = None
    distance_range: Optional[str] = None
    type: str = "strength"

class Block(BaseModel):
    label: Optional[str] = None
    structure: Optional[str] = None              # "3 rounds", "4 sets"
    rest_between_sec: Optional[int] = None       # between sets/rounds
    time_work_sec: Optional[int] = None          # for time-based circuits (e.g., Tabata 20s)
    default_reps_range: Optional[str] = None     # "10-12"
    exercises: List[Exercise] = Field(default_factory=list)

class Workout(BaseModel):
    title: str = "Imported Workout"
    source: Optional[str] = None
    blocks: List[Block] = Field(default_factory=list)

# ---------- OCR ----------
def ocr_image_bytes(b: bytes) -> str:
    img = Image.open(io.BytesIO(b))
    img = img.convert("L")
    return pytesseract.image_to_string(img)

# ---------- Video helpers ----------
def ytdlp_extract(url: str) -> Tuple[str, str, str]:
    if yt_dlp is None:
        raise HTTPException(status_code=500, detail="yt-dlp is not installed. Run: pip install yt-dlp")
    ydl_opts = {"quiet": True, "skip_download": True, "nocheckcertificate": True,
                "youtube_include_dash_manifest": False}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    title = info.get("title") or ""
    desc = info.get("description") or ""
    dl_url = ""
    for f in (info.get("formats") or []):
        if f.get("vcodec") != "none" and (f.get("ext") == "mp4" or "mp4" in str(f.get("ext"))):
            dl_url = f.get("url") or ""
            if dl_url:
                break
    return title, desc, dl_url

def ffmpeg_sample_frames(video_path: str, out_dir: str, fps: float = 0.75, max_secs: int = 25):
    trimmed = os.path.join(out_dir, "clip_trimmed.mp4")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-y", "-i", video_path, "-t", str(max_secs), "-an", trimmed],
        check=True
    )
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-y", "-i", trimmed, "-vf", f"fps={fps}", os.path.join(out_dir, "frame_%03d.png")],
        check=True
    )

def ocr_many_images_to_text(dir_with_pngs: str) -> str:
    texts = []
    for img_path in sorted(glob.glob(os.path.join(dir_with_pngs, "frame_*.png"))):
        try:
            with Image.open(img_path) as im:
                im = im.convert("L")
                txt = pytesseract.image_to_string(im)
                if txt.strip():
                    texts.append(txt)
        except Exception:
            continue
    return "\n".join(texts)

# ---------- Parser ----------
SKI_DEFAULT_WORK = 60
SKI_DEFAULT_REST = 90

# Re-used patterns
RE_DISTANCE = re.compile(r"\b(?P<d1>\d+)(?:[\-–](?P<d2>\d+))?\s*(m|meter|meters|km|mi|mile|miles)\b", re.I)
RE_REPS_RANGE = re.compile(r"(?P<rmin>\d+)\s*[\-–]\s*(?P<rmax>\d+)\s*reps?", re.I)
RE_REPS_AFTER_X = re.compile(r"[x×]\s*(?P<rmin>\d+)\s*[\-–]\s*(?P<rmax>\d+)\b", re.I)
RE_REPS_PLAIN_X = re.compile(r"[x×]\s*(?P<reps>\d+)\b", re.I)
RE_LABELED = re.compile(r"^[A-D]\d+[:\-]?\s*(.*)", re.I)
RE_HEADER = re.compile(r"(primer|strength|power|finisher|metabolic|conditioning|amrap|circuit|muscular\s+endurance|tabata|warm.?up)", re.I)
RE_WEEK = re.compile(r"^(week\s*\d+\s*of\s*\d+)", re.I)
RE_TITLE_HINT = re.compile(r"^(upper|lower|full)\s+body|workout|dumbbell", re.I)
RE_ROUNDS_SETS = re.compile(r"(?:(?P<n>\d+)\s*(?P<kind>rounds?|sets?))", re.I)
RE_REST_BETWEEN = re.compile(r"(?P<rest>\d+)\s*(s|sec|secs|seconds)\s*(rest|between)", re.I)
RE_TABATA_CFG = re.compile(
    r"""^[:\s]* (?P<work>\d+)\s*(s|sec|secs|seconds)? \s*
        (?:work|on)? \s* [/:]\s*
        (?P<rest>\d+)\s*(s|sec|secs|seconds)? \s* (?:rest|off)?
        (?:\s*(?:x|X)\s*(?P<rounds>\d+)|\s*(?P<rounds2>\d+)\s*[xX])? \s*$""",
    re.I | re.X
)
RE_SKI = re.compile(r"\b(ski\s*erg|skierg|skier)\b", re.I)

def _to_int(s: Optional[str]) -> Optional[int]:
    try:
        return int(s) if s is not None else None
    except Exception:
        return None

def _looks_like_header(ln: str) -> bool:
    # Short, mostly letters, uppercase → treat as a section label
    if len(ln) <= 28 and ln.replace("/", " ").isupper() and re.search(r"[A-Z]{3}", ln):
        return True
    return False

def _is_junk(ln: str) -> bool:
    # Skip very short or mostly punctuation / OCR gunk
    if len(ln) < 4:
        return True
    letters = re.sub(r"[^A-Za-z]", "", ln)
    return len(letters) <= 2

def parse_free_text_to_workout(text: str, source: Optional[str] = None) -> Workout:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    blocks: List[Block] = []
    current = Block(label="Block 1")
    wk_title = None

    for ln in lines:
        if _is_junk(ln):
            continue

        # Title capture
        if not wk_title:
            m_week = RE_WEEK.match(ln)
            if m_week:
                wk_title = m_week.group(1).title()
                continue
            if RE_TITLE_HINT.search(ln) and len(ln.split()) <= 6:
                wk_title = ln.title()
                continue

        # Tabata/time config (never an exercise)
        m_tab = RE_TABATA_CFG.match(ln)
        if m_tab:
            work = _to_int(m_tab.group("work")) or 20
            rest = _to_int(m_tab.group("rest")) or 10
            rounds = _to_int(m_tab.group("rounds")) or _to_int(m_tab.group("rounds2")) or 8
            current.time_work_sec = work
            current.rest_between_sec = rest
            current.structure = f"{rounds} rounds"
            if not current.label or current.label.lower() == "block 1":
                current.label = "Tabata"
            continue

        # Section headers
        if RE_HEADER.search(ln) or _looks_like_header(ln):
            if current.exercises:
                blocks.append(current)
            # Normalize a few known variants to nicer labels
            lbl = ln.title()
            if re.search(r"muscular\s+endurance", ln, re.I):
                lbl = "Muscular Endurance"
            if re.search(r"metabolic|conditioning", ln, re.I):
                lbl = "Metabolic Conditioning"
            current = Block(label=lbl)
            # Inline structure / default reps in header
            m_struct = RE_ROUNDS_SETS.search(ln)
            if m_struct:
                current.structure = f"{m_struct.group('n')} {m_struct.group('kind').lower()}"
            m_range = RE_REPS_RANGE.search(ln)
            if m_range:
                current.default_reps_range = f"{m_range.group('rmin')}-{m_range.group('rmax')}"
            continue

        # Standalone structure/rest/reps-range lines
        m_s = RE_ROUNDS_SETS.search(ln)
        m_r = RE_REST_BETWEEN.search(ln)
        m_range_only = RE_REPS_RANGE.search(ln)
        if m_s or m_r or m_range_only:
            if m_s:
                current.structure = f"{m_s.group('n')} {m_s.group('kind').lower()}"
            if m_r:
                current.rest_between_sec = _to_int(m_r.group("rest"))
            if m_range_only:
                current.default_reps_range = f"{m_range_only.group('rmin')}-{m_range_only.group('rmax')}"
            continue

        # Ski Erg special: set timed block config but don't misread distance lines like "200m ski"
        if RE_SKI.search(ln):
            current.label = "Ski Erg"
            # Set default timing if not already set
            current.time_work_sec = current.time_work_sec or SKI_DEFAULT_WORK
            current.rest_between_sec = current.rest_between_sec or SKI_DEFAULT_REST
            # If line also has distance, capture that as an exercise (e.g., "200m ski")
            m_dist_inline = RE_DISTANCE.search(ln)
            if m_dist_inline:
                d1, d2 = m_dist_inline.group("d1"), m_dist_inline.group("d2")
                if d2:
                    dist_range = f"{d1}-{d2}m"
                    current.exercises.append(Exercise(name=ln, distance_range=dist_range, type="strength"))
                else:
                    current.exercises.append(Exercise(name=ln, distance_m=_to_int(d1), type="strength"))
            else:
                # Otherwise add a generic timed step once
                if not current.exercises:
                    current.exercises.append(Exercise(name="Ski Erg", type="interval"))
            continue

        # ----- Exercises -----
        m_lab = RE_LABELED.match(ln)
        if m_lab:
            ln = m_lab.group(1)

        # reps-range (x6-10 or 6-10 reps), single reps (x10)
        reps_range = None
        reps = None
        m_rr = RE_REPS_RANGE.search(ln) or RE_REPS_AFTER_X.search(ln)
        if m_rr:
            reps_range = f"{m_rr.group('rmin')}-{m_rr.group('rmax')}"
        else:
            m_rx = RE_REPS_PLAIN_X.search(ln)
            if m_rx:
                reps = _to_int(m_rx.group("reps"))

        # distance
        distance_m = None
        distance_range = None
        m_dist = RE_DISTANCE.search(ln)
        if m_dist:
            d1, d2 = m_dist.group("d1"), m_dist.group("d2")
            if d2:
                distance_range = f"{d1}-{d2}m"
            else:
                distance_m = _to_int(d1)

        # inherit reps_range from header if none on line
        if not reps and not reps_range and current.default_reps_range:
            reps_range = current.default_reps_range

        # classify: strength if it has reps/reps_range or distance; else interval
        ex_type = "strength" if (reps or reps_range or distance_m or distance_range) else "interval"

        current.exercises.append(Exercise(
            name=ln.strip(" ."),
            reps=reps,
            reps_range=reps_range,
            distance_m=distance_m,
            distance_range=distance_range,
            type=ex_type
        ))

    if current.exercises:
        blocks.append(current)

    return Workout(title=(wk_title or "Imported Workout"), source=source, blocks=blocks)

# ---------- Pretty text export ----------
def render_text_for_tp(workout: Workout) -> str:
    lines = [f"# {workout.title}"]
    if workout.source:
        lines.append(f"(source: {workout.source})")
    lines.append("")
    for bi, b in enumerate(workout.blocks, 1):
        hdr = b.label or f"Block {bi}"
        meta = []
        if b.structure: meta.append(b.structure)
        if b.time_work_sec: meta.append(f"{b.time_work_sec}s work")
        if b.rest_between_sec: meta.append(f"{b.rest_between_sec}s rest")
        if meta: hdr += f" ({', '.join(meta)})"
        lines.append(f"## {hdr}")
        for e in b.exercises:
            parts = [e.name]
            if e.sets: parts.append(f"{e.sets} sets")
            if e.reps_range: parts.append(f"{e.reps_range} reps")
            elif e.reps: parts.append(f"{e.reps} reps")
            if e.distance_range: parts.append(e.distance_range)
            elif e.distance_m: parts.append(f"{e.distance_m}m")
            if b.time_work_sec and not e.reps and not e.reps_range and not e.distance_m and not e.distance_range:
                parts.append(f"{b.time_work_sec}s")
            lines.append("• " + " — ".join(parts))
        lines.append("")
    return "\n".join(lines)

# ---------- FIT export ----------
def canonical_name(name: str) -> str:
    CANON = {
        "db incline bench press": "Dumbbell Incline Bench Press",
        "trx row": "TRX Row",
        "trx rows": "TRX Row",
        "goodmorings": "Good Mornings",
        "kneeling medball slams": "Kneeling Med Ball Slams",
        "medball slams": "Kneeling Med Ball Slams",
    }
    low = " ".join(name.split()).lower()
    return CANON.get(low, name.strip())

def _upper_from_range(txt: str) -> Optional[int]:
    try:
        a, b = txt.replace("–", "-").split("-", 1)
        return int(b.strip())
    except Exception:
        return None

def infer_sets_reps(e: Exercise) -> tuple[int, int]:
    sets = e.sets or 3
    if e.reps:
        reps = e.reps
    elif e.reps_range:
        reps = _upper_from_range(e.reps_range) or 10
    else:
        reps = 8
    return sets, reps

def rounds_from_structure(structure: Optional[str]) -> int:
    if not structure:
        return 1
    m = re.match(r"\s*(\d+)", structure)
    return int(m.group(1)) if m else 1

def build_fit_bytes_from_workout(wk: Workout) -> bytes:
    if FitFileBuilder is None:
        raise RuntimeError("fit-tool not installed. Run: pip install fit-tool")
    ffb = FitFileBuilder()
    ffb.add(WorkoutMessage(sport=Sport.STRENGTH, name=(wk.title or "Workout")[:14]))
    step_index = 0

    for b in wk.blocks:
        reps_mode = not b.time_work_sec  # timed blocks => time steps
        rounds = max(1, rounds_from_structure(b.structure))
        between = b.rest_between_sec or (10 if not reps_mode else 60)

        for _ in range(rounds):
            for e in b.exercises:
                name = canonical_name(e.name)[:15]
                if reps_mode:
                    # distance-based strength: convert to time placeholder if no reps present
                    if (e.distance_m or e.distance_range) and not (e.reps or e.reps_range):
                        step_index += 1
                        ffb.add(WorkoutStepMessage(
                            message_index=step_index,
                            workout_step_name=name,
                            duration_type=DUR.TIME,
                            duration_value=45,  # heuristic placeholder
                            target_type=TGT.OPEN,
                        ))
                        step_index += 1
                        ffb.add(WorkoutStepMessage(
                            message_index=step_index,
                            workout_step_name="Rest",
                            duration_type=DUR.TIME,
                            duration_value=between,
                            target_type=TGT.OPEN,
                        ))
                        continue

                    sets, reps = infer_sets_reps(e)
                    for s in range(sets):
                        step_index += 1
                        ffb.add(WorkoutStepMessage(
                            message_index=step_index,
                            workout_step_name=name,
                            duration_type=DUR.REPS,
                            duration_value=reps,
                            target_type=TGT.OPEN,
                        ))
                        if s < sets - 1:
                            step_index += 1
                            ffb.add(WorkoutStepMessage(
                                message_index=step_index,
                                workout_step_name="Rest",
                                duration_type=DUR.TIME,
                                duration_value=between,
                                target_type=TGT.OPEN,
                            ))
                else:
                    # time-based (e.g., SkiErg/Tabata)
                    step_index += 1
                    ffb.add(WorkoutStepMessage(
                        message_index=step_index,
                        workout_step_name=name,
                        duration_type=DUR.TIME,
                        duration_value=b.time_work_sec or 20,
                        target_type=TGT.OPEN,
                    ))
                    step_index += 1
                    ffb.add(WorkoutStepMessage(
                        message_index=step_index,
                        workout_step_name="Rest",
                        duration_type=DUR.TIME,
                        duration_value=between,
                        target_type=TGT.OPEN,
                    ))

    return ffb.build(file_type=FileType.WORKOUT)

# ---------- Routes ----------
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ingest/text")
async def ingest_text(text: str = Form(...), source: Optional[str] = Form(None)):
    wk = parse_free_text_to_workout(text, source)
    return JSONResponse(wk.model_dump())

@app.post("/ingest/image")
async def ingest_image(file: UploadFile = File(...)):
    b = await file.read()
    text = ocr_image_bytes(b)
    wk = parse_free_text_to_workout(text, source=f"image:{file.filename}")
    return JSONResponse(wk.model_dump())

@app.post("/ingest/url")
async def ingest_url(url: str = Body(..., embed=True)):
    try:
        title, desc, dl_url = ytdlp_extract(url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read URL: {e}")

    collected_text = f"{title}\n{desc}".strip()
    ocr_text = ""
    if dl_url:
        tmpdir = tempfile.mkdtemp(prefix="ingest_url_")
        try:
            video_path = os.path.join(tmpdir, "video.mp4")
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error",
                 "-y", "-i", dl_url, "-t", "30", "-an", video_path],
                check=True
            )
            ffmpeg_sample_frames(video_path, tmpdir, fps=0.75, max_secs=25)
            ocr_text = ocr_many_images_to_text(tmpdir)
        except subprocess.CalledProcessError:
            pass
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    merged_text = "\n".join([t for t in [collected_text, ocr_text] if t]).strip()
    if not merged_text:
        raise HTTPException(status_code=422, detail="No text found in video or description")

    wk = parse_free_text_to_workout(merged_text, source=url)
    if title:
        wk.title = title[:80]
    return JSONResponse(wk.model_dump())

@app.post("/export/tp_text")
async def export_tp_text(workout: Workout):
    txt = render_text_for_tp(workout)
    return Response(
        content=txt,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="workout.txt"'},
    )

@app.post("/export/tcx")
async def export_tcx(workout: Workout):
    # very small TCX with notes summary
    def esc(x: str) -> str:
        return (x or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    notes = []
    for bi, b in enumerate(workout.blocks, 1):
        header = b.label or f"Block {bi}"
        meta = []
        if b.structure: meta.append(b.structure)
        if b.time_work_sec: meta.append(f"{b.time_work_sec}s work")
        if b.rest_between_sec: meta.append(f"{b.rest_between_sec}s rest")
        notes.append(header + (" (" + ", ".join(meta) + ")" if meta else ""))
        for e in b.exercises:
            parts = [e.name]
            if e.reps_range: parts.append(f"{e.reps_range} reps")
            elif e.reps: parts.append(f"{e.reps} reps")
            if e.distance_range: parts.append(e.distance_range)
            elif e.distance_m: parts.append(f"{e.distance_m}m")
            notes.append(" - " + ", ".join(parts))
    tcx = f"""<?xml version="1.0" encoding="UTF-8"?>
<TrainingCenterDatabase xmlns="http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2">
  <Activities>
    <Activity Sport="Other">
      <Id>2025-01-01T00:00:00Z</Id>
      <Lap StartTime="2025-01-01T00:00:00Z">
        <TotalTimeSeconds>0</TotalTimeSeconds>
        <DistanceMeters>0</DistanceMeters>
        <Intensity>Active</Intensity>
        <TriggerMethod>Manual</TriggerMethod>
        <Notes>{esc("\\n".join(notes))}</Notes>
      </Lap>
    </Activity>
  </Activities>
</TrainingCenterDatabase>
"""
    return Response(
        content=tcx,
        media_type="application/vnd.garmin.tcx+xml",
        headers={"Content-Disposition": 'attachment; filename="workout.tcx"'},
    )

@app.post("/export/fit")
async def export_fit(workout: Workout):
    try:
        blob = build_fit_bytes_from_workout(workout)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return Response(
        content=blob,
        media_type="application/octet-stream",
        headers={"Content-Disposition": 'attachment; filename="strength_workout.fit"'},
    )