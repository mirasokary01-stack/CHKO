import streamlit as st
import json
import unicodedata
from datetime import datetime
from groq import Groq
import re
import os
from difflib import SequenceMatcher, get_close_matches
from typing import Optional, Dict, Any, List, Tuple
from langsmith import traceable
import time

# ═══════════════════════════════════════════════════════════════════════════════
# ENV SETUP
# ═══════════════════════════════════════════════════════════════════════════════
os.environ["GROQ_API_KEY"]          = st.secrets["GROQ_API_KEY"]
os.environ["LANGCHAIN_API_KEY"]     = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_PROJECT"]     = st.secrets["LANGCHAIN_PROJECT"]
os.environ["LANGCHAIN_TRACING_V2"]  = st.secrets["LANGCHAIN_TRACING_V2"]

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
STUDENT_ID_REGEX = re.compile(r'\b(202\d{5,})\b')

DAY_AR   = {0:"الاثنين",1:"الثلاثاء",2:"الاربعاء",3:"الخميس",4:"الجمعة",5:"السبت",6:"الاحد"}
TIME_ORDER = {"الاولى":1,"الأولى":1,"الثانية":2,"الثالثة":3,"الرابعة":4,"الخامسة":5}

GRADE_KEYWORDS    = ["درجة","درجه","نتيجة","نتيجه","علامة","علامه","grade","نمرة","نمره","درجات","درجاتي","علامتي","نتيجتي"]
SCHEDULE_KEYWORDS = ["جدول","النهارده","اليوم","بكره","بكرا","غدا","محاضرات","محاضرة","سكشن","موعد","دكتور","د.","مادة","درس","حصة","الجدول","مواعيد"]

GRADE_PATTERNS    = [re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE) for kw in GRADE_KEYWORDS]
SCHEDULE_PATTERNS = [re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE) for kw in SCHEDULE_KEYWORDS]

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_json_file(filename: str) -> Any:
    paths = [
        filename,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), filename),
        os.path.join(os.getcwd(), filename),
    ]
    for path in paths:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                return json.load(f)
    return []

@st.cache_data
def load_schedule() -> List[Dict]:
    return load_json_file("output.json")

@st.cache_data
def load_grades() -> Dict[str, Dict]:
    data = load_json_file("grades.json")
    return {str(item.get("id","")).strip(): item for item in data if item.get("id")}

@st.cache_data
def load_rl() -> Dict[str, Dict]:
    data = load_json_file("rl.json")
    return {str(item.get("id","")).strip(): item for item in data if item.get("id")}

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════
def fix(text: Optional[str]) -> str:
    return unicodedata.normalize('NFKC', str(text or "")).strip()

def norm(text: Optional[str]) -> str:
    if not text: return ""
    text = fix(text)
    for k,v in {'ة':'ه','أ':'ا','إ':'ا','آ':'ا','ى':'ي','ئ':'ي','ؤ':'و'}.items():
        text = text.replace(k,v)
    return text.strip()

def sort_time(t: str) -> int:
    for k,v in TIME_ORDER.items():
        if k in t: return v
    return 99

def today_ar()    -> str: return DAY_AR.get(datetime.now().weekday(),"")
def tomorrow_ar() -> str: return DAY_AR.get((datetime.now().weekday()+1)%7,"")

def fuzzy_match(text: str, keywords: List[str], threshold: float = 0.75) -> bool:
    for kw in keywords:
        if SequenceMatcher(None, text, norm(kw)).ratio() > threshold:
            return True
    return False

def extract_student_id(text: str) -> Optional[str]:
    m = STUDENT_ID_REGEX.search(text)
    return m.group(1) if m else None

def get_last_student_id(messages: List[Dict]) -> Optional[str]:
    for msg in reversed(messages):
        sid = extract_student_id(msg["content"])
        if sid: return sid
    return None

# ═══════════════════════════════════════════════════════════════════════════════
# DATA PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════════
def get_records(data: List[Dict]) -> List[Dict]:
    records = []
    for student in data:
        sid  = fix(student.get("student_id"))
        name = fix(student.get("name"))
        prog = fix(student.get("program"))
        for course in student.get("courses", []):
            records.append({
                "id":sid, "name":name, "program":prog,
                "course": fix(course.get("course_name")),
                "type":   fix(course.get("type")),
                "day":    fix(course.get("day")),
                "time":   fix(course.get("time")),
                "group":  fix(course.get("section")),
                "room":   fix(course.get("room")),
                "doctor": fix(course.get("instructor")),
            })
    return records

# ═══════════════════════════════════════════════════════════════════════════════
# SEARCH
# ═══════════════════════════════════════════════════════════════════════════════
def search_by_student(records, sid):
    return [r for r in records if r["id"] == sid]

def search_by_name(records, name):
    q = norm(name)
    exact = [r for r in records if q in norm(r["name"])]
    if exact: return exact
    all_names = list({r["name"] for r in records})
    close = get_close_matches(q, [norm(n) for n in all_names], n=3, cutoff=0.6)
    return [r for r in records if norm(r["name"]) in close] if close else []

def search_by_doctor(records, doctor_name):
    q = norm(doctor_name)
    exact = [r for r in records if q in norm(r["doctor"])]
    if exact: return exact
    all_docs = list({r["doctor"] for r in records if r["doctor"]})
    close = get_close_matches(q, [norm(d) for d in all_docs], n=2, cutoff=0.55)
    return [r for r in records if norm(r["doctor"]) in close]

def search_by_course(records, course_name):
    q = norm(course_name)
    exact = [r for r in records if q in norm(r["course"])]
    if exact: return exact
    all_courses = list({r["course"] for r in records if r["course"]})
    close = get_close_matches(q, [norm(c) for c in all_courses], n=2, cutoff=0.55)
    return [r for r in records if norm(r["course"]) in close]

# ═══════════════════════════════════════════════════════════════════════════════
# GRADE
# ═══════════════════════════════════════════════════════════════════════════════
def get_grade(sid, grades, rl):
    if not sid or len(sid) < 8: return None
    for store in (rl, grades):
        if sid in store:
            try: return float(store[sid].get("grade"))
            except: pass
    return None

# ═══════════════════════════════════════════════════════════════════════════════
# FORMATTERS
# ═══════════════════════════════════════════════════════════════════════════════
def format_day_schedule(rows, day):
    filtered = sorted(
        [r for r in rows if norm(day) in norm(r["day"])],
        key=lambda x: sort_time(x["time"])
    )
    if not filtered:
        return f"مفيش محاضرات يوم {day} 😊"
    lines = [f"📅 **يوم {day}**\n"]
    for r in filtered:
        t   = "محاضرة" if r["type"] == "Lecture" else "سكشن"
        doc = f"\n    👨‍🏫 {r['doctor']}" if r['doctor'] else ""
        lines.append(f"🕐 **{r['time']}**\n📚 {r['course']} — {t} | 🏛️ {r['room']}{doc}\n")
    return "\n".join(lines)

def format_full_schedule(rows, name, sid):
    lines = [f"👤 **{name}**\n🆔 `{sid}` | {rows[0]['program']}\n"]
    for day in ["السبت","الاحد","الاثنين","الثلاثاء","الاربعاء","الخميس"]:
        day_rows = sorted([r for r in rows if norm(r["day"])==norm(day)], key=lambda x: sort_time(x["time"]))
        if day_rows:
            lines.append(f"**📅 {day}**")
            for r in day_rows:
                tp  = "محاضرة" if "lecture" in r["type"].lower() else "سكشن"
                doc = f" | {r['doctor']}" if r["doctor"] else ""
                lines.append(f"  🕐 {r['time']} — {r['course']} ({tp}) | {r['room']}{doc}")
            lines.append("")
    return "\n".join(lines)

def format_doctor_schedule(rows):
    if not rows: return "مش لاقي الدكتور ده في الجدول 🤔"
    doctor_name = rows[0]["doctor"]
    lines = [f"👨‍🏫 **جدول {doctor_name}**\n"]
    by_day = {}
    for r in rows: by_day.setdefault(r["day"],[]).append(r)
    for day in ["السبت","الاحد","الاثنين","الثلاثاء","الاربعاء","الخميس"]:
        if day in by_day:
            lines.append(f"**📅 {day}**")
            for r in sorted(by_day[day], key=lambda x: sort_time(x["time"])):
                t = "محاضرة" if "lecture" in r["type"].lower() else "سكشن"
                lines.append(f"  🕐 {r['time']} — {r['course']} ({t}) | 🏛️ {r['room']}")
            lines.append("")
    return "\n".join(lines)

def format_course_info(rows, course_name):
    if not rows: return f"مش لاقي مادة '{course_name}' 🤔"
    lines = [f"📚 **{rows[0]['course']}**\n"]
    seen = set()
    for r in rows:
        key = (r["day"],r["time"],r["type"],r["doctor"])
        if key in seen: continue
        seen.add(key)
        t   = "محاضرة" if "lecture" in r["type"].lower() else "سكشن"
        doc = f" | 👨‍🏫 {r['doctor']}" if r["doctor"] else ""
        lines.append(f"📅 {r['day']} 🕐 {r['time']} ({t}) | 🏛️ {r['room']}{doc}")
    return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════════════════
# CLASSIFIERS
# ═══════════════════════════════════════════════════════════════════════════════
def is_grade_question(text):
    t = norm(text)
    return any(p.search(t) for p in GRADE_PATTERNS) or fuzzy_match(t, GRADE_KEYWORDS, 0.75)

def is_schedule_question(text, messages):
    t = norm(text)
    if any(p.search(t) for p in SCHEDULE_PATTERNS): return True
    if STUDENT_ID_REGEX.search(text): return True
    if get_last_student_id(messages):
        return any(c in t for c in [norm(w) for w in ["بكره","اليوم","النهارده","محاضرات","جدول","بكرا","غدا"]])
    return False

# ═══════════════════════════════════════════════════════════════════════════════
# SCHEDULE HANDLER
# ═══════════════════════════════════════════════════════════════════════════════
def handle_schedule_query(text, records, messages):
    t_norm = norm(text)
    sid    = extract_student_id(text) or get_last_student_id(messages)

    if sid:
        rows = search_by_student(records, sid)
        if not rows:
            return f"الرقم **{sid}** مش موجود في الجدول 🤔"
        name = rows[0]["name"]
        if any(w in t_norm for w in [norm(w) for w in ["بكره","بكرا","غدا","غداً"]]):
            return f"👤 **{name}**\n\n" + format_day_schedule(rows, tomorrow_ar())
        for day in DAY_AR.values():
            if norm(day) in t_norm:
                return f"👤 **{name}**\n\n" + format_day_schedule(rows, day)
        if any(w in t_norm for w in [norm(w) for w in ["النهارده","اليوم","دلوقتي","محاضرات"]]):
            return f"👤 **{name}**\n\n" + format_day_schedule(rows, today_ar())
        return format_full_schedule(rows, name, sid)

    SKIP = {norm(w) for w in ["جدول","اليوم","بكره","محاضرات","سكشن","مادة","درس","موعد","ايه","هو","هي","في","من","علي","على","ده","دي","ال","و","ب"]}
    for word in [w for w in text.split() if len(norm(w)) >= 3 and norm(w) not in SKIP]:
        rows = search_by_name(records, word)
        if rows:
            name = rows[0]["name"]
            if any(w in t_norm for w in [norm(w) for w in ["بكره","بكرا","غدا"]]):
                return f"👤 **{name}**\n\n" + format_day_schedule(rows, tomorrow_ar())
            for day in DAY_AR.values():
                if norm(day) in t_norm:
                    return f"👤 **{name}**\n\n" + format_day_schedule(rows, day)
            return format_full_schedule(rows, name, rows[0]["id"])

    doc_match = re.search(r'(?:دكتور[ه]?|د\.?)\s+([\w\s]+)', text)
    if doc_match:
        rows = search_by_doctor(records, doc_match.group(1).strip().split()[0])
        if rows: return format_doctor_schedule(rows)

    course_match = re.search(r'(?:ماد[ةه]|درس|مادة)\s+([\w\s]+)', text)
    if course_match:
        rows = search_by_course(records, course_match.group(1).strip())
        if rows: return format_course_info(rows, course_match.group(1).strip())

    return None

# ═══════════════════════════════════════════════════════════════════════════════
# AI FALLBACK
# ═══════════════════════════════════════════════════════════════════════════════
@traceable(name="chatbot_response")
def ask_groq(messages, context_data=""):
    system_prompt = f"""أنت "شيكو"، مساعد جامعي ودود بتساعد طلاب كلية الحاسوب والذكاء الاصطناعي، جامعة المنوفية الأهلية.

أسلوبك:
- عامية مصرية خفيفة وودودة زي صاحب بيساعد صاحبه
- ردودك مختصرة ومفيدة
- emoji باعتدال
- لو سألوا عن جدول أو درجة قولهم يكتبوا رقمهم الأكاديمي
- لو سؤال أكاديمي جاوب بدقة
- متكتبش كود غير لو طلبوا صراحةً
- اسمك شيكو دايماً

{f'معلومات: {context_data}' if context_data else ''}"""

    groq_msgs = [{"role":"system","content":system_prompt}] + \
                [{"role":m["role"],"content":m["content"]} for m in messages[-12:]]
    for attempt in range(3):
        try:
            res = Groq(api_key=st.secrets["GROQ_API_KEY"]).chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=groq_msgs,
                max_tokens=600,
                temperature=0.3
            )
            return res.choices[0].message.content
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                m = re.search(r'try again in (\d+\.?\d*)s', err, re.IGNORECASE)
                time.sleep(float(m.group(1))+1 if m else 10)
            else:
                return f"حصل خطأ 😅 {err}"
    return "الخدمة مشغولة دلوقتي، جرب تاني 🙏"

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(
        page_title="شيكو — مساعدك الجامعي",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700;900&display=swap');
* { font-family: 'Cairo', sans-serif !important; }

/* ── Dark palette ── */
:root {
    --bg:        #0f1117;
    --surface:   #1a1d27;
    --surface2:  #22263a;
    --border:    #2e3248;
    --text:      #e8eaf6;
    --muted:     #6b7099;
    --accent:    #7c9ef8;
    --accent-d:  #3d5bd9;
    --accent-l:  #1e2a4a;
    --green:     #4ecba8;
    --green-l:   #0f2d24;
    --yellow-l:  #1e1a0a;
    --yellow-b:  #4a3a00;
    --yellow-t:  #f0c040;
}

.stApp { background: var(--bg); color: var(--text); }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.2rem !important; max-width: 680px; }

/* ── header ── */
.shiko-header {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 18px 20px;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 14px;
    direction: rtl;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}
.shiko-avatar {
    width: 52px; height: 52px;
    background: var(--accent-l);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.6rem;
    flex-shrink: 0;
    border: 2px solid var(--accent-d);
}
.shiko-name {
    font-size: 1.15rem; font-weight: 900;
    color: var(--text); margin: 0; line-height: 1.2;
}
.shiko-name em { color: var(--accent); font-style: normal; }
.shiko-sub { font-size: 0.76rem; color: var(--muted); margin: 2px 0 0; }
.online { color: var(--green); font-weight: 700; }
.day-tag {
    margin-right: auto;
    background: var(--accent-l);
    color: var(--accent);
    border-radius: 20px;
    padding: 4px 13px;
    font-size: 0.74rem;
    font-weight: 700;
    white-space: nowrap;
    border: 1px solid var(--border);
}

/* ── tip ── */
.tip {
    background: var(--yellow-l);
    border: 1px solid var(--yellow-b);
    border-radius: 12px;
    padding: 9px 14px;
    font-size: 0.79rem;
    color: var(--yellow-t);
    direction: rtl;
    margin-bottom: 14px;
    line-height: 1.8;
}

/* ── bubbles ── */
.user-b {
    background: var(--accent-d);
    color: #fff;
    padding: 10px 15px;
    border-radius: 18px 18px 4px 18px;
    margin: 5px 0 5px auto;
    max-width: 68%;
    width: fit-content;
    font-size: 0.87rem;
    line-height: 1.65;
    direction: rtl;
}
.bot-b {
    background: var(--surface);
    color: var(--text);
    padding: 11px 15px;
    border-radius: 18px 18px 18px 4px;
    margin: 5px auto 5px 0;
    max-width: 78%;
    font-size: 0.87rem;
    line-height: 1.75;
    border: 1px solid var(--border);
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    direction: rtl;
}

/* ── grade ── */
.grade-card {
    text-align: center;
    padding: 18px 24px;
    background: var(--green-l);
    border: 1.5px solid #1e5a47;
    border-radius: 16px;
    margin: 4px auto;
    max-width: 230px;
}
.grade-num { font-size: 3rem; font-weight: 900; color: var(--green); line-height: 1; }
.grade-lbl { font-size: 0.78rem; color: #4ecba8; margin-top: 3px; opacity: 0.8; }

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-left: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.2rem 1rem !important; }
[data-testid="stSidebar"] * { color: var(--text) !important; }

.sb-sec {
    font-size: 0.7rem; font-weight: 700;
    color: var(--muted) !important; letter-spacing: 1.1px;
    text-transform: uppercase;
    margin: 14px 0 7px; direction: rtl;
}
.ql {
    display: flex; align-items: center; gap: 9px;
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 10px; padding: 9px 12px; margin-bottom: 7px;
    text-decoration: none !important; color: var(--text) !important;
    font-size: 0.81rem; direction: rtl; transition: all 0.15s;
}
.ql:hover {
    background: var(--accent-l);
    border-color: var(--accent-d);
    color: var(--accent) !important;
}

/* ── expander (exam schedule) ── */
[data-testid="stExpander"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] summary {
    color: var(--text) !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
}

/* ── streamlit elements dark override ── */
.stButton button {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-family: 'Cairo', sans-serif !important;
}
.stButton button:hover {
    background: var(--accent-l) !important;
    border-color: var(--accent-d) !important;
    color: var(--accent) !important;
}

/* ── chat input ── */
[data-testid="stChatInput"] textarea {
    background: var(--surface) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 14px !important;
    color: var(--text) !important;
    font-family: 'Cairo', sans-serif !important;
    font-size: 0.9rem !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--accent-d) !important;
    box-shadow: 0 0 0 3px rgba(61,91,217,0.2) !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: var(--muted) !important;
}

/* ── spinner ── */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── divider ── */
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

    # ── Load ─────────────────────────────────────────────────────
    raw_data = load_schedule()
    records  = get_records(raw_data)
    grades   = load_grades()
    rl_data  = load_rl()

    if not records:
        st.error(f"❌ ملف output.json مش موجود! المجلد: {os.getcwd()}")
        st.stop()

    # ── Sidebar ──────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🤖 شيكو")
        st.caption("مساعدك الجامعي — كلية الحاسوب")
        st.divider()

        st.markdown('<p class="sb-sec">روابط سريعة</p>', unsafe_allow_html=True)
        for icon, label, url in [
            ("📚", "منصة المواد (LMS)", "https://mnulms.menofia.education/login/index.php"),
            ("✅", "نظام الحضور",       "https://mnulms.menofia.education/attendance"),
            ("📁", "درايف الملخصات",    "https://drive.google.com/drive/mobile/folders/1MZ079RA9Pj2l7J81O0InWJelhITMPNox"),
        ]:
            st.markdown(
                f'<a class="ql" href="{url}" target="_blank"><span>{icon}</span><span>{label}</span></a>',
                unsafe_allow_html=True
            )

        # ── Exam schedule — hidden behind expander ────────────────
        if os.path.exists("page-1.jpg"):
            st.divider()
            with st.expander("📋 جدول الامتحانات", expanded=False):
                st.image("page-1.jpg", use_container_width=True)

        st.divider()
        if st.button("🗑️ مسح المحادثة", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # ── Header ───────────────────────────────────────────────────
    st.markdown(f"""
    <div class="shiko-header">
        <div class="shiko-avatar">🤖</div>
        <div>
            <p class="shiko-name"><em>شيكو</em> — مساعدك الجامعي</p>
            <p class="shiko-sub">
                <span class="online">● متاح</span> &nbsp;·&nbsp;
                كلية الحاسوب والذكاء الاصطناعي
            </p>
        </div>
        <span class="day-tag">📅 {today_ar()}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Tip ──────────────────────────────────────────────────────
    st.markdown("""
    <div class="tip">
    💡 اكتب <b>رقمك الأكاديمي</b> لعرض جدولك &nbsp;·&nbsp;
       <b>رقمك + درجتي</b> لعرض درجتك &nbsp;·&nbsp;
       اسأل عن <b>دكتور</b> أو <b>مادة</b> بالاسم
    </div>
    """, unsafe_allow_html=True)

    # ── Init ─────────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "مرحباً! 👋 أنا شيكو، مساعدك الدراسي.\nأساعدك إزاي؟ 😊"
        }]

    # ── Render messages ──────────────────────────────────────────
    for msg in st.session_state.messages:
        cls = "user-b" if msg["role"] == "user" else "bot-b"
        st.markdown(f'<div class="{cls}">{msg["content"]}</div>', unsafe_allow_html=True)

    # ── Input ────────────────────────────────────────────────────
    if prompt := st.chat_input("اكتب رسالتك هنا...", key="chat_input"):
        st.session_state.messages.append({"role":"user","content":prompt})

        with st.spinner("شيكو بيفكر... 🤔"):
            sid      = extract_student_id(prompt) or get_last_student_id(st.session_state.messages)
            response = None

            # 1️⃣ Grade
            if is_grade_question(prompt) and sid:
                grade = get_grade(sid, grades, rl_data)
                if grade is not None:
                    rows = search_by_student(records, sid)
                    name = rows[0]["name"] if rows else ""
                    name_line = f'<div style="font-size:.8rem;color:#4ecba8;margin-bottom:5px;">👤 {name}</div>' if name else ""
                    response = f'<div class="grade-card">{name_line}<div class="grade-num">{grade}</div><div class="grade-lbl">من 20 درجة</div></div>'
                else:
                    response = "مش لاقي درجة لرقمك ده 🤔\nتأكد إن الرقم صح."

            # 2️⃣ Schedule
            elif is_schedule_question(prompt, st.session_state.messages):
                response = handle_schedule_query(prompt, records, st.session_state.messages)
                if not response:
                    response = "مش فاهم السؤال 🤔\nجرب تكتب رقمك الأكاديمي، اسم الدكتور، أو اسم المادة."

            # 3️⃣ AI
            else:
                response = ask_groq(st.session_state.messages)

        st.session_state.messages.append({"role":"assistant","content":response})
        st.rerun()

if __name__ == "__main__":
    main()