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

class Superset(BaseModel):
    exercises: List[Exercise] = Field(default_factory=list)
    rest_between_sec: Optional[int] = None       # rest between exercises in superset

class Block(BaseModel):
    label: Optional[str] = None
    structure: Optional[str] = None              # "3 rounds", "4 sets"
    rest_between_sec: Optional[int] = None       # between sets/rounds
    time_work_sec: Optional[int] = None          # for time-based circuits (e.g., Tabata 20s)
    default_reps_range: Optional[str] = None     # "10-12"
    default_sets: Optional[int] = None           # number of sets/rounds (from structure)
    exercises: List[Exercise] = Field(default_factory=list)  # for backward compatibility
    supersets: List[Superset] = Field(default_factory=list)  # new superset support

class Workout(BaseModel):
    title: str = "Imported Workout"
    source: Optional[str] = None
    blocks: List[Block] = Field(default_factory=list)

# ---------- OCR ----------
def ocr_image_bytes(b: bytes) -> str:
    import numpy as np
    from PIL import ImageEnhance, ImageFilter
    
    img = Image.open(io.BytesIO(b))
    
    # Convert to grayscale
    img = img.convert("L")
    
    # Upscale image for better OCR (especially for small text like "Ax", "Az")
    # Scale to at least 300 DPI equivalent (2x-3x scaling helps with small text)
    width, height = img.size
    if width < 2000 or height < 2000:
        # Upscale by factor to ensure minimum dimensions
        scale_factor = max(2000 / width, 2000 / height, 2.0)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        img = img.resize((new_width, new_height), Image.LANCZOS)
    
    # Enhance contrast to improve binarization
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)  # Increase contrast by 2x
    
    # Apply slight sharpening to make edges clearer
    img = img.filter(ImageFilter.SHARPEN)
    
    # Binarize (threshold) to black and white
    # Convert to numpy array for thresholding
    img_array = np.array(img)
    
    # Use Otsu's method for automatic thresholding or adaptive threshold
    # For simplicity, use a fixed threshold - adjust based on typical image brightness
    threshold = 128  # Middle gray
    img_array = np.where(img_array > threshold, 255, 0).astype(np.uint8)
    
    # Convert back to PIL Image
    img = Image.fromarray(img_array)
    
    # Use pytesseract with optimized config for better accuracy
    # --psm 6: Assume a single uniform block of text  
    # Don't use whitelist - it can cause spacing issues
    # Instead use PSM 6 which preserves spacing better
    custom_config = r'--oem 3 --psm 6'
    
    try:
        text = pytesseract.image_to_string(img, config=custom_config)
        # Post-process: ensure spaces are preserved around colons and X multipliers
        # Add space after colon if missing: "A1:GOOD" -> "A1: GOOD"
        text = re.sub(r'([A-E]\d*):([A-Z])', r'\1: \2', text)
        # Add space before X when followed by number: "GOODX10" -> "GOOD X10"
        text = re.sub(r'([A-Za-z])X(\d)', r'\1 X\2', text)
        return text
    except Exception:
        # Fallback to default config if custom config fails
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
RE_DISTANCE = re.compile(r"(?P<d1>\d+)(?:[\-–](?P<d2>\d+))?\s*(m|meter|meters|km|mi|mile|miles)\b", re.I)
RE_REPS_RANGE = re.compile(r"(?P<rmin>\d+)\s*[\-–]\s*(?P<rmax>\d+)\s*reps?", re.I)
RE_REPS_AFTER_X = re.compile(r"[x×]\s*(?P<rmin>\d+)\s*[\-–]\s*(?P<rmax>\d+)\b", re.I)
RE_REPS_PLAIN_X = re.compile(r"[x×]\s*(?P<reps>\d+)\b", re.I)
RE_LABELED = re.compile(r"^[A-E](?:[0-9A-Za-z]+)?[:\-]?\s*(.*)", re.I)
RE_LETTER_START = re.compile(r"^['\"]?[A-E]", re.I)  # Check if line starts with A-E (with optional quote)
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
    # But NOT if it starts with a labeled exercise pattern (A1:, B2:, etc.)
    if re.match(r"^[A-E](?:[0-9A-Za-z]+)?[:\-]?\s*", ln):
        return False  # This is a labeled exercise, not a header
    
    # Don't treat instruction lines as headers (they contain numbers and specific words)
    if re.search(r"\d+\s*(rounds?|sets?|mins?|secs?|rest)", ln.lower()):
        return False
    
    if len(ln) <= 28 and ln.replace("/", " ").isupper() and re.search(r"[A-Z]{3}", ln):
        return True
    return False

def _is_junk(ln: str) -> bool:
    # Skip very short or mostly punctuation / OCR gunk
    if len(ln) < 4:
        return True
    
    # Skip Instagram UI elements - but only if the line doesn't contain exercise content
    instagram_words = ['like', 'dislike', 'share', 'comment', 'follow', 'followers', 'following']
    ln_lower = ln.lower()
    # Check if line contains Instagram words AND no exercise indicators
    exercise_indicators = ['x', ':', 'kg', 'kb', 'db', 'rep', 'set', 'round', 'meter', 'm', 'squat', 'press', 'push', 'pull', 'carry', 'sled', 'swing', 'burpee', 'jump']
    has_exercise_content = any(indicator in ln_lower for indicator in exercise_indicators) or re.search(r'[A-E]\d*:', ln)
    
    if any(instagram_word in ln_lower for instagram_word in instagram_words) and not has_exercise_content:
        return True
    
    # Skip lines that are just numbers (like Instagram like counts: "4", "0")
    # But not if they're part of an exercise line or rep count
    if re.match(r'^\d+$', ln.strip()) and len(ln.strip()) <= 3:
        return True
    
    # Skip lines that look like "block 1", "block 2", etc. (OCR artifacts)
    if re.match(r'^block\s+\d+$', ln.lower()):
        return True
    
    # Skip lines that are just single letters or weird patterns like "q:Ry"
    if re.match(r'^[a-z]:[A-Z][a-z]+$', ln):
        return True
    
    # Count letters and common punctuation
    letters = re.sub(r"[^A-Za-z]", "", ln)
    if len(letters) <= 2:
        return True
    
    # Skip lines with excessive backslashes or weird characters (OCR artifacts)
    if ln.count('\\') > 2 or ln.count('|') > 2:
        return True
    
    # Skip lines that look like corrupted text (mix of letters, numbers, symbols in weird patterns)
    # Pattern: starts with backslash, has weird spacing, or looks like OCR gibberish
    if re.match(r'^\\\s*[a-z]\.\s*[a-z]', ln.lower()):
        return True
    
    # Skip lines with too many single characters separated by spaces/punctuation
    single_chars = re.findall(r'\b[a-z]\b', ln.lower())
    if len(single_chars) > len(ln.split()) * 0.5:  # More than 50% single characters
        return True
    
    # Skip lines that don't contain any recognizable exercise-related words
    exercise_words = ['press', 'squat', 'deadlift', 'row', 'pull', 'push', 'curl', 'extension', 
                     'flexion', 'raise', 'lift', 'hold', 'plank', 'burpee', 'jump', 'run', 
                     'walk', 'bike', 'swim', 'ski', 'erg', 'meter', 'rep', 'set', 'round',
                     'goodmorning', 'sled', 'drag', 'carry', 'farmer', 'hand', 'release',
                     'kb', 'db', 'dual', 'alternating', 'broad', 'swing', 'skier']
    ln_lower = ln.lower()
    has_exercise_word = any(word in ln_lower for word in exercise_words)
    
    # If it's a short line without exercise words and has weird characters, skip it
    if len(ln) < 20 and not has_exercise_word and re.search(r'[\\|\.]{2,}', ln):
        return True
    
    return False

def parse_free_text_to_workout(text: str, source: Optional[str] = None) -> Workout:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    
    # Clean up OCR artifacts before processing
    cleaned_lines = []
    for ln in lines:
        # Clean up common OCR artifacts - be very aggressive
        ln = ln.replace("'", "").replace("'", "").replace("'", "")  # Remove all types of quotes
        # Fix specific OCR issues FIRST, before general patterns
        ln = re.sub(r"^Ax:", "A1:", ln)  # Fix "Ax:" -> "A1:"
        ln = re.sub(r"^Az:", "A2:", ln)  # Fix "Az:" -> "A2:"
        ln = re.sub(r"^A3:", "A3:", ln)  # Ensure A3 is properly formatted
        # Then apply general patterns
        ln = re.sub(r"^([A-E])[a-z]+:", r"\1:", ln)  # Fix other "Ax:" -> "A:", "Az:" -> "A:"
        ln = re.sub(r"oS OFF", "90S OFF", ln)  # Fix "oS OFF" -> "90S OFF"
        # Handle any remaining quote issues
        ln = re.sub(r"^'([A-E])", r"\1", ln)  # Remove leading quotes from letters
        cleaned_lines.append(ln)
    
    blocks: List[Block] = []
    current = Block(label="Block 1")
    wk_title = None
    current_superset: List[Exercise] = []  # Track exercises for current superset
    superset_letter = None  # Track current superset letter (A, B, C, D, etc.)

    for ln in cleaned_lines:
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
            # Finish current superset if any
            if current_superset:
                current.supersets.append(Superset(exercises=current_superset.copy()))
                current_superset.clear()
            
            if current.exercises or current.supersets:
                blocks.append(current)
            # Normalize a few known variants to nicer labels
            lbl = ln.title()
            if re.search(r"muscular\s+endurance", ln, re.I):
                lbl = "Muscular Endurance"
            if re.search(r"metabolic|conditioning", ln, re.I):
                lbl = "Metabolic Conditioning"
            current = Block(label=lbl)
            superset_letter = None  # Reset superset tracking
            # Inline structure / default reps in header
            m_struct = RE_ROUNDS_SETS.search(ln)
            if m_struct:
                current.structure = f"{m_struct.group('n')} {m_struct.group('kind').lower()}"
                # Store the number of rounds/sets as default sets for exercises in this block
                current.default_sets = _to_int(m_struct.group("n"))
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
                # Store the number of rounds/sets as default sets for exercises in this block
                current.default_sets = _to_int(m_s.group("n"))
            if m_r:
                current.rest_between_sec = _to_int(m_r.group("rest"))
            if m_range_only:
                current.default_reps_range = f"{m_range_only.group('rmin')}-{m_range_only.group('rmax')}"
            continue

        # Ski Erg special: set timed block config but don't misread distance lines like "200m ski"
        # Only trigger if it's NOT a labeled exercise (labeled exercises are handled later)
        if RE_SKI.search(ln) and not RE_LETTER_START.match(ln):
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
        # Store the full line for the exercise name, before stripping the label
        full_line_for_name = ln
        
        # Check if line starts with A-E (more flexible for OCR artifacts)
        letter_match = RE_LETTER_START.match(ln)
        exercise_letter = None
        if letter_match:
            # Extract the actual letter, skipping any leading quotes
            letter_part = letter_match.group(0)
            exercise_letter = letter_part[-1].upper()  # Get the last character (the actual letter)
            # Try to extract exercise name after the letter
            m_lab = RE_LABELED.match(ln)
            if m_lab:
                ln = m_lab.group(1)
            else:
                # If labeled regex doesn't match, just remove the first character and colon
                ln = re.sub(r"^[A-E][:\-]?\s*", "", ln)

        # Check for interval/timed exercise pattern: "60S ON 90S OFF X3"
        # This should extract: work time, rest time, and sets
        interval_pattern = re.compile(r'(?P<work>\d+)S?\s+ON\s+(?P<rest>\d+)S?\s+OFF(?:\s+X(?P<sets>\d+))?', re.I)
        m_interval = interval_pattern.search(ln)
        
        time_work_sec = None
        rest_sec = None
        sets = None
        
        if m_interval:
            # This is a timed/interval exercise (like Ski Erg)
            time_work_sec = _to_int(m_interval.group("work"))
            rest_sec = _to_int(m_interval.group("rest"))
            sets = _to_int(m_interval.group("sets"))
            # Set block-level timing if not already set
            if current.label and "Metabolic Conditioning" in current.label:
                current.time_work_sec = current.time_work_sec or time_work_sec
                current.rest_between_sec = current.rest_between_sec or rest_sec
        
        # distance (check before reps, as distance-based exercises don't have reps)
        distance_m = None
        distance_range = None
        m_dist = RE_DISTANCE.search(ln)
        if m_dist:
            d1, d2 = m_dist.group("d1"), m_dist.group("d2")
            if d2:
                distance_range = f"{d1}-{d2}m"
            else:
                distance_m = _to_int(d1)
        
        # reps-range (x6-10 or 6-10 reps), single reps (x10)
        # Don't parse reps if it's an interval exercise OR if distance is found (distance-based exercises)
        reps_range = None
        reps = None
        
        # Only parse reps if it's not an interval exercise and no distance is found
        if not m_interval and not m_dist:
            m_rr = RE_REPS_RANGE.search(ln) or RE_REPS_AFTER_X.search(ln)
            if m_rr:
                reps_range = f"{m_rr.group('rmin')}-{m_rr.group('rmax')}"
            else:
                m_rx = RE_REPS_PLAIN_X.search(ln)
                if m_rx:
                    reps = _to_int(m_rx.group("reps"))

        # inherit reps_range from header if none on line (but not for distance-based exercises)
        if not reps and not reps_range and not distance_m and not distance_range and current.default_reps_range:
            reps_range = current.default_reps_range

        # classify: interval if it has time_work_sec/rest_sec, strength if it has reps/reps_range or distance
        if time_work_sec or rest_sec:
            ex_type = "interval"
        else:
            ex_type = "strength" if (reps or reps_range or distance_m or distance_range) else "interval"

        # Clean and validate exercise name
        exercise_name = full_line_for_name.strip(" .")
        
        # Remove Instagram UI text that may appear at the end of exercise names
        # Be careful not to remove valid exercise content - only remove if it's clearly Instagram UI
        # Remove standalone Instagram words at the end
        exercise_name = re.sub(r'\s+(Dislike|Share|Like|Comment|Follow|Followers|Following)$', '', exercise_name, flags=re.I)
        # Remove single lowercase letters at the end (like "a", "s") that are likely OCR artifacts
        exercise_name = re.sub(r'\s+([a-z])$', '', exercise_name)
        exercise_name = exercise_name.strip()
        
        # Additional validation for exercise names
        if _is_junk(exercise_name):
            continue  # Skip this line entirely
        
        # Skip names that are clearly OCR artifacts
        if re.search(r'^\\\s*[a-z]\.\s*[a-z]', exercise_name.lower()):
            continue
        
        # Use default_sets from block if sets not already set
        exercise_sets = sets if sets is not None else current.default_sets
        
        exercise = Exercise(
            name=exercise_name,
            sets=exercise_sets,  # Use parsed sets or default from block structure
            reps=reps if not m_interval else None,  # Don't use reps for interval exercises
            reps_range=reps_range,
            duration_sec=time_work_sec,
            rest_sec=rest_sec,
            distance_m=distance_m,
            distance_range=distance_range,
            type=ex_type
        )
        
        # Handle supersets vs individual exercises
        if exercise_letter:
            # Special case: METABOLIC CONDITIONING E exercises should be individual, not supersets
            if current.label and "Metabolic Conditioning" in current.label and exercise_letter == "E":
                # Finish current superset if any
                if current_superset:
                    current.supersets.append(Superset(exercises=current_superset.copy()))
                    current_superset.clear()
                    superset_letter = None
                # Add as individual exercise
                current.exercises.append(exercise)
            # For most blocks, group all exercises into one superset
            # Exception: MUSCULAR ENDURANCE has multiple supersets (C1,C2 and D1,D2)
            elif current.label and "Muscular Endurance" in current.label:
                # Special case: MUSCULAR ENDURANCE has multiple supersets
                if superset_letter != exercise_letter:
                    # Finish previous superset if any
                    if current_superset:
                        current.supersets.append(Superset(exercises=current_superset.copy()))
                    # Start new superset
                    current_superset = [exercise]
                    superset_letter = exercise_letter
                else:
                    # Add to current superset
                    current_superset.append(exercise)
            else:
                # For all other blocks, group all exercises into one superset
                if not current_superset:
                    # Start the superset for this block
                    current_superset = [exercise]
                    superset_letter = exercise_letter
                else:
                    # Add to the existing superset for this block
                    current_superset.append(exercise)
        else:
            # Finish current superset if any (unlabeled exercise starts)
            if current_superset:
                current.supersets.append(Superset(exercises=current_superset.copy()))
                current_superset.clear()
                superset_letter = None
            # Add as individual exercise
            current.exercises.append(exercise)

    # Finish any remaining superset
    if current_superset:
        current.supersets.append(Superset(exercises=current_superset.copy()))
    
    if current.exercises or current.supersets:
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
        
        # Render supersets
        for si, superset in enumerate(b.supersets):
            if len(b.supersets) > 1:
                lines.append(f"### Superset {si + 1}")
            for e in superset.exercises:
                parts = [e.name]
                if e.sets: parts.append(f"{e.sets} sets")
                if e.reps_range: parts.append(f"{e.reps_range} reps")
                elif e.reps: parts.append(f"{e.reps} reps")
                if e.distance_range: parts.append(e.distance_range)
                elif e.distance_m: parts.append(f"{e.distance_m}m")
                if b.time_work_sec and not e.reps and not e.reps_range and not e.distance_m and not e.distance_range:
                    parts.append(f"{b.time_work_sec}s")
                lines.append("• " + " — ".join(parts))
            if superset.rest_between_sec:
                lines.append(f"Rest: {superset.rest_between_sec}s between exercises")
        
        # Render individual exercises
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