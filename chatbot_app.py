import streamlit as st
import json
import unicodedata
from datetime import datetime
from groq import Groq
import re
import os
from difflib import SequenceMatcher, get_close_matches
from typing import Optional, Dict, Any, List
from langsmith import traceable
import time

# ═══════════════════════════════════════════════════════════════════════════════
# ENV SETUP
# ═══════════════════════════════════════════════════════════════════════════════
os.environ["GROQ_API_KEY"]         = st.secrets["GROQ_API_KEY"]
os.environ["LANGCHAIN_API_KEY"]    = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_PROJECT"]    = st.secrets["LANGCHAIN_PROJECT"]
os.environ["LANGCHAIN_TRACING_V2"] = st.secrets["LANGCHAIN_TRACING_V2"]

@st.cache_resource
def get_groq_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
STUDENT_ID_REGEX = re.compile(r'\b(202\d{5,})\b')

DAY_AR     = {0:"الاثنين",1:"الثلاثاء",2:"الاربعاء",3:"الخميس",4:"الجمعة",5:"السبت",6:"الاحد"}
TIME_ORDER = {"الاولى":1,"الأولى":1,"الثانية":2,"الثالثة":3,"الرابعة":4,"الخامسة":5}

GRADE_KEYWORDS    = ["درجة","درجه","نتيجة","نتيجه","علامة","علامه","grade",
                     "نمرة","نمره","درجات","درجاتي","علامتي","نتيجتي"]
SCHEDULE_KEYWORDS = ["جدول","النهارده","اليوم","بكره","بكرا","غدا","محاضرات",
                     "محاضرة","سكشن","موعد","دكتور","د.","مادة","درس","حصة",
                     "الجدول","مواعيد"]

GRADE_PATTERNS    = [re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE) for kw in GRADE_KEYWORDS]
SCHEDULE_PATTERNS = [re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE) for kw in SCHEDULE_KEYWORDS]

# ═══════════════════════════════════════════════════════════════════════════════
# SHORTCUTS
# ═══════════════════════════════════════════════════════════════════════════════
SHORTCUTS: Dict[str, Dict] = {
    "/ml": {
        "emoji": "🤖", "hint": "/ml xgboost", "label": "شرح ML",
        "template": """اشرح المفهوم التالي من مجال Machine Learning بالعربية البسيطة.\nاجعل الشرح منظماً:\n1. التعريف\n2. كيف يعمل (بمثال بسيط)\n3. المميزات\n4. العيوب\n5. مثال تطبيقي حقيقي\n\nالمفهوم: {query}""",
    },
    "/code": {
        "emoji": "💻", "hint": "/code حساب المتوسط", "label": "كود Python",
        "template": """اكتب كود Python نظيف للمهمة التالية.\nالمتطلبات:\n- مناسب للمبتدئين\n- كل سطر معلّق بـ comment بالعربية\n- أضف مثال للاستخدام\n\nالمهمة: {query}""",
    },
    "/sum": {
        "emoji": "📝", "hint": "/sum [النص هنا]", "label": "تلخيص",
        "template": """لخّص النص التالي في نقاط مرتبة.\n- bullet points واضحة\n- الأفكار الرئيسية فقط\n- لغة بسيطة\n- لا تتجاوز 8 نقاط\n\nالنص:\n{query}""",
    },
    "/compare": {
        "emoji": "⚖️", "hint": "/compare KMeans vs DBSCAN", "label": "مقارنة",
        "template": """قارن بين الشيئين التاليين بجدول منظم بالعربية يشمل:\n- التعريف\n- متى نستخدم كل منهم\n- المميزات والعيوب\n- مثال\n\nالمقارنة: {query}""",
    },
    "/explain": {
        "emoji": "💡", "hint": "/explain overfitting", "label": "شرح مبسط",
        "template": """اشرح المفهوم التالي بأسلوب بسيط لطالب سنة أولى.\n- ابدأ بمثال من الحياة اليومية\n- ثم الشرح العلمي\n- ثم مثال تطبيقي في البرمجة\n\nالمفهوم: {query}""",
    },
    "/steps": {
        "emoji": "📋", "hint": "/steps بناء نموذج ML", "label": "خطوات",
        "template": """اشرح خطوات تنفيذ المهمة التالية بشكل مرتب ومفصل.\n- رقّم كل خطوة\n- اشرح سبب كل خطوة\n- نبّه على الأخطاء الشائعة\n\nالمهمة: {query}""",
    },
    "/quiz": {
        "emoji": "❓", "hint": "/quiz neural networks", "label": "مراجعة",
        "template": """اعمل 5 أسئلة مراجعة للموضوع التالي مع إجاباتها.\n- تنوع بين MCQ ومقالي قصير\n- رتّب من السهل للصعب\n- ضع الإجابة بعد كل سؤال\n\nالموضوع: {query}""",
    },
    "/debug": {
        "emoji": "🐛", "hint": "/debug xgboost overfitting", "label": "تشخيص مشكلة",
        "template": """حلّل المشكلة التالية خطوة بخطوة بأسلوب مطوّر متخصص.
اشرح:
1. الأسباب المحتملة
2. كيف تكتشف المشكلة (indicators)
3. الحلول المقترحة بالترتيب من الأبسط للأعقد
4. كود أو إعدادات عملية لحل المشكلة

المشكلة: {query}""",
    },
    "/جدولي":  {"emoji": "📅", "hint": "/جدولي",  "label": "جدولي",   "template": "__SCHEDULE_TODAY__"},
    "/درجتي":  {"emoji": "🎯", "hint": "/درجتي",  "label": "درجتي",   "template": "__GRADE__"},
    "/help":   {"emoji": "🆘", "hint": "/help",   "label": "المساعدة", "template": "__HELP__"},
}

SHORTCUT_DISPLAY = [(c,i) for c,i in SHORTCUTS.items() if not i["template"].startswith("__")]
SHORTCUT_SPECIAL = [(c,i) for c,i in SHORTCUTS.items() if     i["template"].startswith("__")]

# ═══════════════════════════════════════════════════════════════════════════════
# RATE LIMITER
# ═══════════════════════════════════════════════════════════════════════════════
RATE_LIMIT_PER_MINUTE = 10
RATE_COOLDOWN_SEC     = 2

def init_rate_limiter():
    if "rl_timestamps" not in st.session_state:
        st.session_state.rl_timestamps   = []
    if "rl_last_request" not in st.session_state:
        st.session_state.rl_last_request = 0.0

def check_rate_limit() -> tuple:
    init_rate_limiter()
    now = time.time()
    since_last = now - st.session_state.rl_last_request
    if since_last < RATE_COOLDOWN_SEC:
        wait = round(RATE_COOLDOWN_SEC - since_last, 1)
        return False, "cooldown", wait
    st.session_state.rl_timestamps = [
        t for t in st.session_state.rl_timestamps if now - t < 60
    ]
    if len(st.session_state.rl_timestamps) >= RATE_LIMIT_PER_MINUTE:
        oldest = st.session_state.rl_timestamps[0]
        wait   = round(60 - (now - oldest), 1)
        return False, "window", wait
    return True, "ok", 0.0

def record_request():
    now = time.time()
    st.session_state.rl_last_request = now
    st.session_state.rl_timestamps.append(now)

def requests_left() -> int:
    now    = time.time()
    recent = [t for t in st.session_state.get("rl_timestamps", []) if now - t < 60]
    return max(0, RATE_LIMIT_PER_MINUTE - len(recent))

# ═══════════════════════════════════════════════════════════════════════════════
# SHORTCUT EXPANSION
# ═══════════════════════════════════════════════════════════════════════════════
def expand_shortcut(prompt: str) -> tuple:
    s = prompt.strip()
    if not s.startswith("/"):
        return s, None, False

    matched_cmd = matched_info = None
    for cmd in sorted(SHORTCUTS.keys(), key=len, reverse=True):
        if s.lower().startswith(cmd.lower()):
            matched_cmd  = cmd
            matched_info = SHORTCUTS[cmd]
            break

    if matched_info is None:
        return s, None, False

    query    = s[len(matched_cmd):].strip()
    template = matched_info["template"]

    if template.startswith("__"):
        return template, matched_cmd, True

    return template.format(query=query or "(غير محدد)"), matched_cmd, False

# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY
# ═══════════════════════════════════════════════════════════════════════════════
def init_memory() -> Dict:
    return {
        "student_id":   None,
        "student_name": None,
        "last_course":  None,
        "last_doctor":  None,
        "last_day":     None,
        "last_intent":  None,
    }

def get_memory() -> Dict:
    if "memory" not in st.session_state:
        st.session_state.memory = init_memory()
    return st.session_state.memory

def update_memory(**kwargs):
    mem = get_memory()
    for k, v in kwargs.items():
        if v is not None:
            mem[k] = v

def memory_context_str() -> str:
    mem   = get_memory()
    parts = []
    if mem["student_id"]:
        parts.append(f"الطالب الحالي: {mem['student_name'] or ''} (رقم {mem['student_id']})")
    if mem["last_course"]:
        parts.append(f"آخر مادة: {mem['last_course']}")
    if mem["last_doctor"]:
        parts.append(f"آخر دكتور: {mem['last_doctor']}")
    if mem["last_day"]:
        parts.append(f"آخر يوم: {mem['last_day']}")
    return " | ".join(parts) if parts else ""

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
    q     = norm(name)
    exact = [r for r in records if q in norm(r["name"])]
    if exact: return exact
    all_names = list({r["name"] for r in records})
    close = get_close_matches(q, [norm(n) for n in all_names], n=3, cutoff=0.6)
    return [r for r in records if norm(r["name"]) in close] if close else []

def search_by_doctor(records, doctor_name):
    q     = norm(doctor_name)
    exact = [r for r in records if q in norm(r["doctor"])]
    if exact: return exact
    all_docs = list({r["doctor"] for r in records if r["doctor"]})
    close = get_close_matches(q, [norm(d) for d in all_docs], n=2, cutoff=0.55)
    return [r for r in records if norm(r["doctor"]) in close]

def search_by_course(records, course_name):
    q     = norm(course_name)
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
        day_rows = sorted(
            [r for r in rows if norm(r["day"])==norm(day)],
            key=lambda x: sort_time(x["time"])
        )
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
    lines  = [f"👨‍🏫 **جدول {doctor_name}**\n"]
    by_day = {}
    for r in rows:
        by_day.setdefault(r["day"],[]).append(r)
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
    seen  = set()
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
def is_grade_question(text: str) -> bool:
    t = norm(text)
    return any(p.search(t) for p in GRADE_PATTERNS) or fuzzy_match(t, GRADE_KEYWORDS, 0.75)

def is_schedule_question(text: str) -> bool:
    t = norm(text)
    if any(p.search(t) for p in SCHEDULE_PATTERNS): return True
    if STUDENT_ID_REGEX.search(text): return True
    if get_memory().get("student_id"):
        day_words = [norm(w) for w in ["بكره","اليوم","النهارده","محاضرات","جدول","بكرا","غدا"]]
        if any(w in t for w in day_words):
            return True
    return False

# ═══════════════════════════════════════════════════════════════════════════════
# SCHEDULE HANDLER
# ═══════════════════════════════════════════════════════════════════════════════
def handle_schedule_query(text: str, records: List[Dict]) -> Optional[str]:
    t_norm = norm(text)
    mem    = get_memory()
    sid    = extract_student_id(text) or mem.get("student_id")

    if sid:
        rows = search_by_student(records, sid)
        if not rows:
            return f"الرقم **{sid}** مش موجود في الجدول 🤔"
        name = rows[0]["name"]
        update_memory(student_id=sid, student_name=name, last_intent="schedule")
        if any(w in t_norm for w in [norm(w) for w in ["بكره","بكرا","غدا","غداً"]]):
            day = tomorrow_ar()
            update_memory(last_day=day)
            return f"👤 **{name}**\n\n" + format_day_schedule(rows, day)
        for day in DAY_AR.values():
            if norm(day) in t_norm:
                update_memory(last_day=day)
                return f"👤 **{name}**\n\n" + format_day_schedule(rows, day)
        if any(w in t_norm for w in [norm(w) for w in ["النهارده","اليوم","دلوقتي","محاضرات"]]):
            day = today_ar()
            update_memory(last_day=day)
            return f"👤 **{name}**\n\n" + format_day_schedule(rows, day)
        return format_full_schedule(rows, name, sid)

    SKIP = {norm(w) for w in ["جدول","اليوم","بكره","محاضرات","سكشن","مادة",
                               "درس","موعد","ايه","هو","هي","في","من","علي",
                               "على","ده","دي","ال","و","ب"]}
    for word in [w for w in text.split() if len(norm(w)) >= 3 and norm(w) not in SKIP]:
        rows = search_by_name(records, word)
        if rows:
            name = rows[0]["name"]
            update_memory(student_name=name, last_intent="schedule")
            if any(w in t_norm for w in [norm(w) for w in ["بكره","بكرا","غدا"]]):
                return f"👤 **{name}**\n\n" + format_day_schedule(rows, tomorrow_ar())
            for day in DAY_AR.values():
                if norm(day) in t_norm:
                    return f"👤 **{name}**\n\n" + format_day_schedule(rows, day)
            return format_full_schedule(rows, name, rows[0]["id"])

    doc_match = re.search(r'(?:دكتور[ه]?|د\.?)\s+([\w\s]+)', text)
    if doc_match:
        doc_name = doc_match.group(1).strip().split()[0]
        rows = search_by_doctor(records, doc_name)
        if rows:
            update_memory(last_doctor=rows[0]["doctor"], last_intent="schedule")
            return format_doctor_schedule(rows)

    course_match = re.search(r'(?:ماد[ةه]|درس|مادة)\s+([\w\s]+)', text)
    if course_match:
        course_name = course_match.group(1).strip()
        rows = search_by_course(records, course_name)
        if rows:
            update_memory(last_course=rows[0]["course"], last_intent="schedule")
            return format_course_info(rows, course_name)

    return None

# ═══════════════════════════════════════════════════════════════════════════════
# AI FALLBACK
# ═══════════════════════════════════════════════════════════════════════════════
@traceable(name="chatbot_response")
def ask_groq(messages: List[Dict], extra_context: str = "") -> str:
    mem_ctx = memory_context_str()
    context = " | ".join(filter(None, [mem_ctx, extra_context]))

    system_prompt = f"""أنت "شيكو"، مساعد جامعي ودود بتساعد طلاب كلية الحاسوب والذكاء الاصطناعي، جامعة المنوفية الأهلية.

أسلوبك:
- عامية مصرية خفيفة وودودة زي صاحب بيساعد صاحبه
- ردودك مختصرة ومفيدة
- emoji باعتدال
- لو سألوا عن جدول أو درجة قولهم يكتبوا رقمهم الأكاديمي
- لو سؤال أكاديمي جاوب بدقة
- متكتبش كود غير لو طلبوا صراحةً
- اسمك شيكو دايماً
- استخدم المعلومات الموجودة في سياق المحادثة للرد بشكل شخصي

{f'سياق المحادثة الحالية: {context}' if context else ''}"""

    groq_msgs = [{"role":"system","content":system_prompt}] + \
                [{"role":m["role"],"content":m["content"]} for m in messages[-12:]]

    client  = get_groq_client()
    backoff = [1, 5, 15]

    for attempt in range(3):
        try:
            res = client.chat.completions.create(
                model       = "llama-3.3-70b-versatile",
                messages    = groq_msgs,
                max_tokens  = 600,
                temperature = 0.3,
            )
            return res.choices[0].message.content
        except Exception as e:
            err = str(e)
            is_rate_limit = "429" in err or "rate_limit" in err.lower()
            if is_rate_limit:
                hint = re.search(r'try again in (\d+\.?\d*)s', err, re.IGNORECASE)
                wait = float(hint.group(1)) + 1 if hint else backoff[attempt]
                time.sleep(wait)
            else:
                return f"حصل خطأ غير متوقع 😅\n`{err}`"

    return "الـ API مشغولة دلوقتي 🙏 جرب تاني بعد شوية."

# ═══════════════════════════════════════════════════════════════════════════════
# CORE PROMPT PROCESSOR  ← ✅ الجديد: منفصل عن الـ UI
# ═══════════════════════════════════════════════════════════════════════════════
def process_prompt(prompt: str, records, grades, rl_data):
    """
    يعالج أي prompt (من chat_input أو shortcut button) ويضيف الرد للـ messages.
    """
    # ── Rate limit ──────────────────────────────────────────────
    allowed, reason, wait_secs = check_rate_limit()
    if not allowed:
        if reason == "cooldown":
            st.markdown(
                f'<div class="rl-warn">⏳ استنى {wait_secs} ثانية بين كل رسالة</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="rl-error">🚫 وصلت الحد الأقصى ({RATE_LIMIT_PER_MINUTE} طلبات/دقيقة)'
                f' — استنى {wait_secs} ثانية</div>',
                unsafe_allow_html=True
            )
        return  # لا نكمل

    record_request()

    # ── Expansion ────────────────────────────────────────────────
    expanded_prompt, matched_cmd, is_special = expand_shortcut(prompt)

    # استخرج الـ sid من الرسالة الأصلية
    sid_in_prompt = extract_student_id(prompt)
    if sid_in_prompt:
        update_memory(student_id=sid_in_prompt)

    # أضف رسالة المستخدم الأصلية للـ history
    st.session_state.messages.append({"role":"user","content":prompt})

    sid = sid_in_prompt or get_memory().get("student_id")

    with st.spinner("شيكو بيفكر... 🤔"):
        response = None

        # ── Special tokens ──────────────────────────────────────
        if is_special and expanded_prompt == "__HELP__":
            cmds_html = "".join(
                f'<div style="margin:3px 0;direction:rtl;">'
                f'<b style="color:#7c9ef8;">{cmd}</b> — {info["label"]} '
                f'<span style="color:#6b7099;font-size:0.75rem;">({info["hint"]})</span></div>'
                for cmd, info in SHORTCUTS.items()
            )
            response = (
                '<div style="direction:rtl;line-height:2;">' +
                '<b style="color:#4ecba8;">⚡ الأوامر المتاحة:</b><br>' +
                cmds_html + "</div>"
            )

        elif is_special and expanded_prompt == "__GRADE__":
            if sid:
                grade = get_grade(sid, grades, rl_data)
                update_memory(last_intent="grade")
                if grade is not None:
                    rows = search_by_student(records, sid)
                    name = rows[0]["name"] if rows else ""
                    if name: update_memory(student_name=name)
                    name_line = (
                        f'<div style="font-size:.8rem;color:#4ecba8;margin-bottom:5px;">👤 {name}</div>'
                        if name else ""
                    )
                    response = (
                        f'<div class="grade-card">{name_line}'
                        f'<div class="grade-num">{grade}</div>'
                        f'<div class="grade-lbl">من 20 درجة</div></div>'
                    )
                else:
                    response = "مش لاقي درجة لرقمك ده 🤔\nتأكد إن الرقم صح."
            else:
                response = "اكتب رقمك الأكاديمي الأول 📝\nمثال: 20231234 /درجتي"

        elif is_special and expanded_prompt == "__SCHEDULE_TODAY__":
            if sid:
                rows = search_by_student(records, sid)
                if rows:
                    name = rows[0]["name"]
                    update_memory(student_name=name, last_intent="schedule")
                    response = f"👤 **{name}**\n\n" + format_day_schedule(rows, today_ar())
                else:
                    response = "مش لاقي جدولك 🤔"
            else:
                response = "اكتب رقمك الأكاديمي الأول 📝"

        # ── Grade ───────────────────────────────────────────────
        elif is_grade_question(prompt) and sid:
            grade = get_grade(sid, grades, rl_data)
            update_memory(last_intent="grade")
            if grade is not None:
                rows = search_by_student(records, sid)
                name = rows[0]["name"] if rows else ""
                if name: update_memory(student_name=name)
                name_line = (
                    f'<div style="font-size:.8rem;color:#4ecba8;margin-bottom:5px;">👤 {name}</div>'
                    if name else ""
                )
                response = (
                    f'<div class="grade-card">{name_line}'
                    f'<div class="grade-num">{grade}</div>'
                    f'<div class="grade-lbl">من 20 درجة</div></div>'
                )
            else:
                response = "مش لاقي درجة لرقمك ده 🤔\nتأكد إن الرقم صح."

        # ── Schedule ────────────────────────────────────────────
        elif is_schedule_question(prompt):
            response = handle_schedule_query(prompt, records)
            if not response:
                response = ("مش فاهم السؤال 🤔\n"
                            "جرب تكتب رقمك الأكاديمي، اسم الدكتور، أو اسم المادة.")

        # ── AI fallback ─────────────────────────────────────────
        else:
            update_memory(last_intent="ai")
            is_debug_cmd = (matched_cmd == "/debug")
            t_start = time.time()

            temp_messages = st.session_state.messages.copy()
            temp_messages[-1] = {"role":"user","content":expanded_prompt}
            response = ask_groq(temp_messages)

            elapsed = round(time.time() - t_start, 2)

            if is_debug_cmd:
                mem_snap  = get_memory()
                mem_lines = "".join(
                    f'<div><span class="dbg-key">{k:15s}</span> : '
                    f'<span class="dbg-val">{v}</span></div>'
                    for k, v in mem_snap.items()
                )
                ep_display = (expanded_prompt
                              .replace("&","&amp;")
                              .replace("<","&lt;")
                              .replace(">","&gt;"))
                debug_panel = f"""
<div class="debug-panel">
  <div class="dbg-title">🐛 DEBUG MODE — شيكو Expansion Layer</div>
  <div><span class="dbg-key">shortcut       </span> : <span class="dbg-val">{matched_cmd}</span></div>
  <div><span class="dbg-key">llm_response_ms</span> : <span class="dbg-val">{elapsed}s</span></div>
  <div><span class="dbg-key">model          </span> : <span class="dbg-val">llama-3.3-70b-versatile</span></div>
  <br>
  <div class="dbg-title">📦 Memory Snapshot</div>
  {mem_lines}
  <br>
  <div class="dbg-title">📤 Expanded Prompt → LLM</div>
  <div class="dbg-prompt">{ep_display}</div>
</div>"""
                response = debug_panel + response

    st.session_state.messages.append({"role":"assistant","content":response})
    # نخلي الـ injected_prompt فاضي بعد المعالجة
    st.session_state.injected_prompt = None

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
    --red-muted: #ff6b6b;
}

.stApp { background: var(--bg); color: var(--text); }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.2rem !important; max-width: 680px; }

[data-testid="collapsedControl"],
section[data-testid="stSidebar"] { display: none !important; }

[data-testid="stHorizontalBlock"] { gap: 8px !important; align-items: center !important; }

.shiko-header {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 16px 20px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 14px;
    direction: rtl;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}
.shiko-avatar {
    width: 50px; height: 50px;
    background: var(--accent-l);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem; flex-shrink: 0;
    border: 2px solid var(--accent-d);
}
.shiko-name { font-size: 1.1rem; font-weight: 900; color: var(--text); margin: 0; line-height: 1.2; }
.shiko-name em { color: var(--accent); font-style: normal; }
.shiko-sub { font-size: 0.74rem; color: var(--muted); margin: 2px 0 0; }
.online { color: var(--green); font-weight: 700; }
.day-tag {
    margin-right: auto;
    background: var(--accent-l); color: var(--accent);
    border-radius: 20px; padding: 4px 12px;
    font-size: 0.73rem; font-weight: 700; white-space: nowrap;
    border: 1px solid var(--border);
}

.links-row { display: flex; align-items: center; gap: 7px; direction: rtl; }
.ql {
    display: inline-flex; align-items: center; gap: 5px;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 6px 12px;
    text-decoration: none !important; color: var(--text) !important;
    font-size: 0.78rem; font-weight: 600; transition: all 0.15s; white-space: nowrap;
}
.ql:hover { background: var(--accent-l); border-color: var(--accent-d); color: var(--accent) !important; }

/* ✅ زرار المسح */
.stButton > button {
    background: var(--surface2) !important; color: var(--muted) !important;
    border: 1px solid var(--border) !important; border-radius: 10px !important;
    font-family: 'Cairo', sans-serif !important; font-size: 0.76rem !important;
    padding: 6px 12px !important; height: 36px !important;
    white-space: nowrap !important; width: 100% !important; transition: all 0.15s !important;
}
.stButton > button:hover {
    color: var(--red-muted) !important; border-color: var(--red-muted) !important;
    background: var(--surface2) !important;
}

/* ✅ أزرار الـ shortcuts — override للـ style فوق */
div[data-testid="stHorizontalBlock"] div.sc-btn button {
    height: auto !important;
    padding: 8px 6px !important;
    font-size: 0.78rem !important;
    color: var(--text) !important;
    border-color: var(--border) !important;
    background: var(--surface) !important;
    border-radius: 10px !important;
    line-height: 1.5 !important;
    text-align: right !important;
}
div[data-testid="stHorizontalBlock"] div.sc-btn button:hover {
    background: var(--accent-l) !important;
    border-color: var(--accent-d) !important;
    color: var(--accent) !important;
}

[data-testid="stExpander"] {
    background: var(--surface) !important; border: 1px solid var(--border) !important;
    border-radius: 12px !important; margin-bottom: 10px !important;
}
[data-testid="stExpander"] summary {
    color: var(--text) !important; font-size: 0.82rem !important;
    font-weight: 600 !important; direction: rtl !important;
}

.tip {
    background: var(--yellow-l); border: 1px solid var(--yellow-b);
    border-radius: 12px; padding: 8px 14px; font-size: 0.77rem;
    color: var(--yellow-t); direction: rtl; margin-bottom: 12px; line-height: 1.8;
}

.mem-badge {
    display: inline-block;
    background: var(--accent-l);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 3px 10px;
    font-size: 0.72rem;
    color: var(--accent);
    direction: rtl;
    margin-bottom: 8px;
}

.user-b {
    background: var(--accent-d); color: #fff;
    padding: 10px 15px; border-radius: 18px 18px 4px 18px;
    margin: 5px 0 5px auto; max-width: 68%; width: fit-content;
    font-size: 0.87rem; line-height: 1.65; direction: rtl;
}
.bot-b {
    background: var(--surface); color: var(--text);
    padding: 11px 15px; border-radius: 18px 18px 18px 4px;
    margin: 5px auto 5px 0; max-width: 78%;
    font-size: 0.87rem; line-height: 1.75;
    border: 1px solid var(--border);
    box-shadow: 0 2px 8px rgba(0,0,0,0.3); direction: rtl;
}

.grade-card {
    text-align: center; padding: 18px 24px;
    background: var(--green-l); border: 1.5px solid #1e5a47;
    border-radius: 16px; margin: 4px auto; max-width: 230px;
}
.grade-num { font-size: 3rem; font-weight: 900; color: var(--green); line-height: 1; }
.grade-lbl { font-size: 0.78rem; color: #4ecba8; margin-top: 3px; opacity: 0.8; }

[data-testid="stChatInput"] textarea {
    background: var(--surface) !important; border: 1.5px solid var(--border) !important;
    border-radius: 14px !important; color: var(--text) !important;
    font-family: 'Cairo', sans-serif !important; font-size: 0.9rem !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--accent-d) !important;
    box-shadow: 0 0 0 3px rgba(61,91,217,0.2) !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: var(--muted) !important; }

.stSpinner > div { border-top-color: var(--accent) !important; }
hr { border-color: var(--border) !important; }

.rl-bar {
    display: flex; align-items: center; gap: 8px;
    direction: rtl; font-size: 0.72rem;
    color: var(--muted); margin-bottom: 6px;
}
.rl-pips { display: flex; gap: 3px; }
.rl-pip {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--accent); opacity: 0.85;
}
.rl-pip.used { background: var(--border); opacity: 0.4; }
.rl-warn {
    background: rgba(250,204,21,0.08);
    border: 1px solid rgba(250,204,21,0.3);
    border-radius: 10px; padding: 7px 14px;
    font-size: 0.8rem; color: #f0c040;
    direction: rtl; margin-bottom: 8px; text-align: center;
}
.rl-error {
    background: rgba(255,107,107,0.08);
    border: 1px solid rgba(255,107,107,0.3);
    border-radius: 10px; padding: 7px 14px;
    font-size: 0.8rem; color: #ff6b6b;
    direction: rtl; margin-bottom: 8px; text-align: center;
}

.debug-panel {
    background: #0d1f0d; border: 1px solid #1a5c1a;
    border-radius: 14px; padding: 14px 18px;
    font-family: 'Courier New', monospace; font-size: 0.74rem;
    color: #4ade80; direction: ltr; margin-bottom: 10px; line-height: 1.9;
}
.debug-panel .dbg-title { color: #86efac; font-weight: 700; font-size: 0.8rem; margin-bottom: 6px; letter-spacing: 1px; }
.debug-panel .dbg-key   { color: #6b7099; }
.debug-panel .dbg-val   { color: #4ade80; }
.debug-panel .dbg-prompt {
    background: #0a150a; border: 1px solid #1a4a1a;
    border-radius: 8px; padding: 8px 10px; margin-top: 6px;
    color: #86efac; white-space: pre-wrap; word-break: break-word;
}
</style>
""", unsafe_allow_html=True)

    # ── Load data ────────────────────────────────────────────────
    raw_data = load_schedule()
    records  = get_records(raw_data)
    grades   = load_grades()
    rl_data  = load_rl()

    if not records:
        st.error(f"❌ ملف output.json مش موجود! المجلد: {os.getcwd()}")
        st.stop()

    # ── Init session state ───────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "مرحباً! 👋 أنا شيكو، مساعدك الدراسي.\nأساعدك إزاي؟ 😊"
        }]
    if "memory" not in st.session_state:
        st.session_state.memory = init_memory()
    if "show_exam" not in st.session_state:
        st.session_state.show_exam = False
    # ✅ الجديد: injected_prompt للـ shortcut buttons
    if "injected_prompt" not in st.session_state:
        st.session_state.injected_prompt = None

    init_rate_limiter()

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

    # ── Memory badge ─────────────────────────────────────────────
    mem = get_memory()
    if mem.get("student_id"):
        label = mem.get("student_name") or mem["student_id"]
        st.markdown(
            f'<div class="mem-badge">🧠 محفوظ: {label}</div>',
            unsafe_allow_html=True
        )

    # ── Rate-limit pip bar ───────────────────────────────────────
    left = requests_left()
    pips_html = "".join(
        f'<div class="rl-pip{"" if i < left else " used"}"></div>'
        for i in range(RATE_LIMIT_PER_MINUTE)
    )
    st.markdown(
        f'<div class="rl-bar"><div class="rl-pips">{pips_html}</div>'
        f'<span>{left} / {RATE_LIMIT_PER_MINUTE} طلب متاح في الدقيقة</span></div>',
        unsafe_allow_html=True
    )

    # ── Toolbar ──────────────────────────────────────────────────
    col_links, col_btn = st.columns([5, 1])

    with col_links:
        st.markdown("""
        <div class="links-row">
            <a class="ql" href="https://mnulms.menofia.education/login/index.php" target="_blank">📚 منصة المواد</a>
            <a class="ql" href="https://mnulms.menofia.education/attendance" target="_blank">✅ الحضور</a>
            <a class="ql" href="https://drive.google.com/drive/mobile/folders/1MZ079RA9Pj2l7J81O0InWJelhITMPNox" target="_blank">📁 الملخصات</a>
        </div>
        """, unsafe_allow_html=True)

    with col_btn:
        if st.button("🗑️ مسح"):
            st.session_state.messages        = []
            st.session_state.memory          = init_memory()
            st.session_state.injected_prompt = None
            st.rerun()

    # ── Exam toggle ──────────────────────────────────────────────
    if os.path.exists("page-1.jpg"):
        exam_label = ("📋 جدول الامتحانات ▲ — إخفاء"
                      if st.session_state.show_exam
                      else "📋 جدول الامتحانات ▼ — اضغط للعرض")
        if st.button(exam_label, key="exam_toggle"):
            st.session_state.show_exam = not st.session_state.show_exam
            st.rerun()
        if st.session_state.show_exam:
            st.image("page-1.jpg", use_container_width=True)

    # ── Tip ──────────────────────────────────────────────────────
    st.markdown("""
    <div class="tip">
    💡 اكتب <b>رقمك الأكاديمي</b> لعرض جدولك &nbsp;·&nbsp;
       <b>رقمك + درجتي</b> لعرض درجتك &nbsp;·&nbsp;
       اسأل عن <b>دكتور</b> أو <b>مادة</b> &nbsp;·&nbsp;
       جرب <b>/ml</b> أو <b>/code</b> أو <b>/sum</b> لردود احترافية ⚡
    </div>
    """, unsafe_allow_html=True)

    # ── Shortcuts panel ──────────────────────────────────────────
    # ✅ كل زرار دلوقتي st.button حقيقي يعمل inject للـ prompt
    with st.expander("⚡ Prompt Shortcuts — اضغط على أمر لتفعيله مباشرةً", expanded=False):
        st.markdown("""
        <div style="direction:rtl;font-size:0.78rem;color:#94a3b8;margin-bottom:8px;">
        اضغط على أي أمر وشيكو هيرد فوراً 🚀
        </div>""", unsafe_allow_html=True)

        # AI shortcuts
        st.markdown(
            '<div style="font-size:0.75rem;color:#7dd4fc;font-weight:700;'
            'margin:4px 0 6px;direction:rtl;">🎓 أوامر أكاديمية</div>',
            unsafe_allow_html=True
        )
        cols = st.columns(4)
        for i, (cmd, info) in enumerate(SHORTCUT_DISPLAY):
            with cols[i % 4]:
                btn_label = f"{info['emoji']} {cmd}\n{info['label']}"
                if st.button(btn_label, key=f"sc_{cmd}"):
                    st.session_state.injected_prompt = cmd
                    st.rerun()

        # University shortcuts
        st.markdown(
            '<div style="font-size:0.75rem;color:#7dd4fc;font-weight:700;'
            'margin:8px 0 6px;direction:rtl;">🏫 أوامر الجامعة</div>',
            unsafe_allow_html=True
        )
        ucols = st.columns(3)
        for i, (cmd, info) in enumerate(SHORTCUT_SPECIAL):
            with ucols[i % 3]:
                btn_label = f"{info['emoji']} {cmd}\n{info['label']}"
                if st.button(btn_label, key=f"sc_{cmd}"):
                    st.session_state.injected_prompt = cmd
                    st.rerun()

    # ── Render chat history ──────────────────────────────────────
    for msg in st.session_state.messages:
        cls = "user-b" if msg["role"] == "user" else "bot-b"
        st.markdown(f'<div class="{cls}">{msg["content"]}</div>',
                    unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # ✅ INJECTION HANDLER — shortcut button ضغط عليه
    # يشتغل قبل st.chat_input عشان يعالج الـ injected prompt
    # ══════════════════════════════════════════════════════════════
    if st.session_state.injected_prompt:
        injected = st.session_state.injected_prompt
        process_prompt(injected, records, grades, rl_data)
        st.rerun()

    # ── Chat input ───────────────────────────────────────────────
    if prompt := st.chat_input("اكتب رسالتك هنا...  (/help للأوامر)", key="chat_input"):
        process_prompt(prompt, records, grades, rl_data)
        st.rerun()


if __name__ == "__main__":
    main()
