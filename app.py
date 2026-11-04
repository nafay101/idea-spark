import io
import json
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from google import genai
from google.genai import errors, types

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Idea Spark AI", page_icon="✨", layout="wide")

# Models are tried in this order. If one is unavailable (404), rate-limited (429)
# or overloaded (503), the app moves on to the next one.
# Google renames models often: see https://ai.google.dev/gemini-api/docs/models
MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-flash-latest",
]

MAX_FILE_MB = 15

# Speed setting -> how long the model "thinks" before answering
SPEEDS = {"⚡ Fast": "low", "⚖️ Balanced": "medium", "🧠 Deep thinking": "high"}
CHAT_SYSTEM = (
    "You are Spark, the friendly AI assistant inside the Idea Spark AI app. "
    "If someone asks your name, say you are Spark. If asked which model powers you, "
    "say Google Gemini. Be helpful, warm and clear."
)
MAX_HISTORY = 20  # only the last 20 chat messages are sent, which keeps replies quick

# ----------------------------------------------------------------------------
# Writing tools: name shown in the dropdown -> instruction sent to the AI
# Add your own by adding one more line.
# ----------------------------------------------------------------------------
TASKS = {
    "📝 Summarize": "Summarize the text in a few short, clear bullet points.",
    "✨ Rewrite professionally": "Rewrite the text in a polished, professional tone. Keep the meaning.",
    "🧒 Explain like I'm 12": "Explain the text in very simple words a 12-year-old can understand.",
    "✅ Fix grammar": "Fix all grammar, spelling and word-choice mistakes. Keep the meaning and style. "
                      "Return the corrected text first, then a short list of the main corrections.",
    "🎓 Explain at expert level (with references)": "Explain the topic of the text at an advanced, expert level. "
        "At the end, suggest real books, papers or organizations the reader can search for. "
        "Do NOT invent links or exact citations.",
    "🧠 Explain with analogies": "Explain the ideas in the text using clear everyday analogies.",
    "📚 Make study notes": "Turn the text into well-organized study notes with headings and bullet points.",
    "🃏 Make flashcards": "Create 8 flashcards from the text in the form 'Q: ... / A: ...'.",
    "🧩 Make a quiz (5 questions)": "Create a 5-question multiple-choice quiz from the text. "
                                    "Put the answer key at the end.",
    "🗺️ Mind-map outline": "Turn the text into a mind-map style nested outline.",
    "❓ Questions to think about": "Write 7 thought-provoking discussion questions based on the text.",
    "🤯 Interesting facts & trivia": "Share surprising, interesting facts and trivia related to the text's topic.",
    "📖 Turn into a short story": "Turn the ideas in the text into a short, engaging story.",
    "🎭 Turn into a poem": "Turn the ideas in the text into a poem.",
    "📱 Social media post": "Write a catchy social media post based on the text, with a few hashtags.",
    "💼 LinkedIn post": "Write a professional, engaging LinkedIn post based on the text.",
    "📧 Draft an email": "Draft a clear, polite email based on the text, with a subject line.",
    "🎤 Presentation script": "Write a short presentation script (about 2 minutes) based on the text.",
    "🏷️ Title & headline ideas": "Suggest 10 creative titles or headlines for the text.",
    "🎯 Key takeaways & action items": "List the key takeaways and clear action items from the text.",
    "⚖️ Pros and cons": "List the pros and cons related to the text in two clear sections.",
    "🔍 Critique & counter-arguments": "Critique the text fairly and give strong counter-arguments.",
    "🌍 Translate": "Translate the text into {language}. Return only the translation.",
}

# ----------------------------------------------------------------------------
# Light, colorful styling
# ----------------------------------------------------------------------------
# Cute background doodles (inline SVG, no image files needed)
DOODLE_SVG = """<svg xmlns='http://www.w3.org/2000/svg' width='440' height='440' viewBox='0 0 440 440' fill='none' stroke-linecap='round' stroke-linejoin='round'>
<g stroke='%23B9ABF7' stroke-width='2.5' opacity='.55'>
<path d='M50 50 q8 -18 16 0 q8 18 16 0 q8 -18 16 0'/>
<circle cx='330' cy='70' r='14' stroke-dasharray='4 7'/>
<path d='M205 125 v22 M194 136 h22'/>
<path d='M60 380 q10 -14 20 0 q10 14 20 0'/>
</g>
<g stroke='%23F6A6D2' stroke-width='2.5' opacity='.6'>
<path transform='translate(360 270) scale(.7)' d='M0 -4 C-14 -22 -34 -2 0 24 C34 -2 14 -22 0 -4z'/>
<path d='M150 250 l5 -11 l5 11 l11 5 l-11 5 l-5 11 l-5 -11 l-11 -5z'/>
<path d='M300 380 l3 -7 l3 7 l7 3 l-7 3 l-3 7 l-3 -7 l-7 -3z'/>
</g>
<g stroke='%23FFBE94' stroke-width='2.5' opacity='.65'>
<circle cx='95' cy='200' r='15'/>
<path d='M88 218 h14 M90 225 h10'/>
<path d='M95 170 v-8 M72 178 l-6 -5 M118 178 l6 -5'/>
<path d='M250 40 q10 -10 20 0 q10 10 20 0'/>
</g>
<g stroke='%239ADBC5' stroke-width='2.5' opacity='.65'>
<path d='M380 150 l4 -9 l4 9 l9 4 l-9 4 l-4 9 l-4 -9 l-9 -4z'/>
<circle cx='230' cy='330' r='7'/>
<path d='M390 390 v16 M382 398 h16'/>
</g>
<g fill='%23D9CFFF' opacity='.7'><circle cx='180' cy='60' r='4'/><circle cx='20' cy='290' r='4'/><circle cx='420' cy='330' r='4'/></g>
<g fill='%23FFD3E8' opacity='.8'><circle cx='320' cy='190' r='4'/><circle cx='120' cy='120' r='4'/></g>
</svg>"""
DOODLE_URI = 'url("data:image/svg+xml;utf8,' + DOODLE_SVG.replace("\n", "") + '")'

CSS = """
<style>
.stApp {
    background-image: __DOODLE__, linear-gradient(135deg, #F4EFFF 0%, #FFF1F8 55%, #FFF5EC 100%);
    background-size: 440px 440px, cover;
    background-repeat: repeat, no-repeat;
    background-attachment: fixed, fixed;
}
.stApp, .stApp p, .stApp li, .stApp label, .stApp span, .stApp div,
.stApp h2, .stApp h3, .stApp h4 { color: #2D2A4A; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, rgba(244,239,255,.95), rgba(255,240,248,.95)); border-right: 1px solid #E6DEFF; }
[data-testid="stSidebar"] > div { background: transparent; }
[data-testid="stSidebar"] * { color: #2D2A4A; }
[data-testid="stHeader"] { background: transparent; }

.hero { padding: 6px 0 14px 0; }
.hero h1 { font-size: 2.4rem; margin: 0; background: linear-gradient(90deg, #7C6AF0, #E879B9);
    -webkit-background-clip: text; background-clip: text; color: transparent !important;
    -webkit-text-fill-color: transparent; }
.hero p { margin: 4px 0 0 0; color: #6B6890 !important; }

button[data-baseweb="tab"] { background: transparent; font-weight: 600; color: #7A7699 !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: #7C6AF0 !important; }

[data-testid="stChatMessage"] {
    background: rgba(255,255,255,.82); border: 1px solid #EADFFF; border-radius: 18px;
    padding: 12px 16px; box-shadow: 0 4px 14px rgba(139, 124, 246, 0.08);
}
.card { background: rgba(255,255,255,.82); border: 1px solid #EADFFF; border-radius: 18px;
        padding: 18px; box-shadow: 0 4px 14px rgba(139, 124, 246, 0.08); }

.stTextArea textarea, .stTextInput input {
    background: rgba(255,255,255,.85) !important; color: #2D2A4A !important;
    border: 1px solid #DDD3FF !important; border-radius: 14px !important;
}
div[data-baseweb="select"] > div { background: rgba(255,255,255,.85) !important; border-radius: 14px !important; border-color: #DDD3FF !important; }
div[data-baseweb="select"] * { color: #2D2A4A !important; }
ul[role="listbox"], div[data-baseweb="popover"] > div { background: #FFFFFF !important; }
ul[role="listbox"] li { color: #2D2A4A !important; }
ul[role="listbox"] li:hover { background: #F1ECFF !important; }

/* Chat bar: soft glass instead of a big white strip */
[data-testid="stBottom"], [data-testid="stBottom"] > div,
[data-testid="stBottomBlockContainer"] { background: transparent !important; }
[data-testid="stChatInput"] { background: rgba(255,255,255,.78) !important; border: 1px solid #DDD3FF !important;
    border-radius: 20px !important; box-shadow: 0 6px 20px rgba(139,124,246,.14); }
[data-testid="stChatInput"] > div, [data-testid="stChatInput"] textarea { background: transparent !important; color: #2D2A4A !important; }
[data-testid="stChatInput"] button { background: linear-gradient(135deg, #8B7CF6, #F08CC8) !important; border-radius: 12px; }
[data-testid="stChatInput"] button svg { fill: #FFFFFF !important; color: #FFFFFF !important; }

[data-testid="stFileUploaderDropzone"] { background: rgba(255,255,255,.7); border: 2px dashed #CFC4FF; border-radius: 16px; }
[data-testid="stFileUploaderDropzone"] * { color: #2D2A4A !important; }

.stButton > button, [data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(90deg, #8B7CF6, #F08CC8); color: #FFFFFF !important;
    border: none; border-radius: 14px; padding: 0.55rem 1.3rem; font-weight: 600;
    box-shadow: 0 4px 12px rgba(139,124,246,.25);
}
.stButton > button:hover { filter: brightness(1.06); }
.stButton > button p, [data-testid="stFileUploaderDropzone"] button * { color: #FFFFFF !important; }

.thinking { display: inline-flex; align-items: center; gap: 10px; padding: 8px 16px;
    background: rgba(255,255,255,.85); border: 1px solid #DDD3FF; border-radius: 999px;
    color: #7C6AF0; font-weight: 600; animation: glow 1.8s ease-in-out infinite; }
.thinking .dot { width: 8px; height: 8px; border-radius: 50%; background: #8B7CF6;
    animation: bounce 1.2s infinite ease-in-out; }
.thinking .dot:nth-child(2) { animation-delay: 0.15s; background: #F08CC8; }
.thinking .dot:nth-child(3) { animation-delay: 0.3s; background: #FFB27D; }
@keyframes bounce { 0%, 80%, 100% { transform: translateY(0); opacity: .5; }
                    40% { transform: translateY(-6px); opacity: 1; } }
@keyframes glow { 0%, 100% { box-shadow: 0 0 0 0 rgba(139,124,246,0.25); }
                  50% { box-shadow: 0 0 0 8px rgba(139,124,246,0); } }

/* Always light: no black/dark parts, whatever the browser or system theme is */
:root, html, body, .stApp { color-scheme: light !important; }
[data-testid="stMainMenuPopover"], [data-testid="stMainMenuList"],
[data-testid="stMainMenuPopover"] * { background: #FFFFFF !important; color: #2D2A4A !important; }
[data-testid="stToolbar"] *, [data-testid="stHeader"] button { color: #2D2A4A !important; }
[data-testid="stTooltipContent"] { background: #FFFFFF !important; color: #2D2A4A !important; }
[data-testid="stCode"] pre { background: #F4F0FF !important; }
[data-testid="stCode"] code, [data-testid="stCode"] code span { color: #2D2A4A !important; }
[data-testid="stExpander"] { background: rgba(255,255,255,.7); border: 1px solid #EADFFF; border-radius: 14px; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { color: #2D2A4A !important; }
[data-testid="stDataFrame"] { border: 1px solid #EADFFF; border-radius: 12px; overflow: hidden; }
.stDownloadButton > button { background: #FFFFFF; color: #7C6AF0 !important; border: 1px solid #DDD3FF;
    border-radius: 14px; font-weight: 600; }
.stDownloadButton > button p { color: #7C6AF0 !important; }
</style>
"""
st.markdown(CSS.replace("__DOODLE__", DOODLE_URI), unsafe_allow_html=True)


def pill(text: str) -> str:
    return (
        f'<div class="thinking"><span>{text}</span>'
        '<span class="dot"></span><span class="dot"></span><span class="dot"></span></div>'
    )


# ----------------------------------------------------------------------------
# Gemini helpers
# ----------------------------------------------------------------------------
class FriendlyError(Exception):
    """An error whose message is safe to show to the user."""

    def __init__(self, message, detail=""):
        super().__init__(message)
        self.detail = detail


def make_config(temperature, system, thinking, json_mode=False):
    kwargs = dict(temperature=temperature, system_instruction=system)
    if json_mode:
        kwargs["response_mime_type"] = "application/json"
    if thinking:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking)
    return types.GenerateContentConfig(**kwargs)


ROUNDS = 5  # how many times all models are tried before giving up


def stream_gemini(client, contents, temperature, model_list, system=None,
                  thinking="low", on_wait=None, json_mode=False):
    """Yield text chunks without bothering the user.

    Busy or rate-limited? Jump to the next model right away. If every model is
    busy, wait a little (longer each round) and go around again.
    """
    config = make_config(temperature, system, thinking, json_mode)
    dead = set()  # models that do not exist (404) are skipped for this request
    last_detail = ""
    final_problems = set()

    for round_no in range(ROUNDS):
        live = [m for m in model_list if m not in dead]
        if not live:
            break
        problems = set()

        for model in live:
            for _ in range(2):  # 2nd pass only if we must drop the speed setting
                got_text = False
                try:
                    for chunk in client.models.generate_content_stream(
                        model=model, contents=contents, config=config
                    ):
                        if chunk.text:
                            got_text = True
                            yield chunk.text
                    return
                except errors.APIError as e:
                    code = getattr(e, "code", None)
                    last_detail = f"{model} -> {code}: {str(e)[:300]}"
                    if got_text:
                        raise FriendlyError("The answer was cut off. Please try again.", last_detail)
                    if code == 400 and "thinking" in str(e).lower() and config.thinking_config:
                        config = make_config(temperature, system, None, json_mode)
                        continue  # same model, without the speed setting
                    if code in (400, 401, 403):
                        if code in (401, 403) or "api key" in str(e).lower():
                            raise FriendlyError("Your API key was rejected. Please check it in the sidebar.", last_detail)
                        raise FriendlyError("Google could not process this request.", last_detail)
                    if code == 404:
                        dead.add(model)
                    elif code == 429:
                        problems.add("limit")
                    elif code in (500, 502, 503, 504):
                        problems.add("busy")
                    else:
                        raise FriendlyError("Unexpected error from Google.", last_detail)
                    break  # next model

        final_problems = problems
        if not problems or round_no == ROUNDS - 1:
            break
        if problems == {"limit"}:
            wait, msg = 5 * (round_no + 1), "⏳ Free-tier limit reached, waiting a moment"
        else:
            wait, msg = 2 * (round_no + 1), "🔄 Google is busy, trying again"
        if on_wait:
            on_wait(msg)
        time.sleep(wait)

    if "limit" in final_problems:
        raise FriendlyError("Free-tier limit reached. Please wait about a minute and send again.", last_detail)
    if "busy" in final_problems:
        raise FriendlyError("Google's servers are still busy. Please send your message again in a moment.", last_detail)
    raise FriendlyError(
        "None of the models were available. Type a current model name in the sidebar "
        "(see ai.google.dev/gemini-api/docs/models).", last_detail,
    )


def run_stream(client, contents, temperature, model_list, thinking_text,
               system=None, thinking="low"):
    """Show the thinking pill, stream the answer, return the final text (or None on error)."""
    pill_box = st.empty()
    out_box = st.empty()
    detail_box = st.empty()
    pill_box.markdown(pill(thinking_text), unsafe_allow_html=True)

    def on_wait(msg):
        pill_box.markdown(pill(msg), unsafe_allow_html=True)

    text = ""
    try:
        for piece in stream_gemini(client, contents, temperature, model_list,
                                   system, thinking, on_wait):
            pill_box.empty()
            text += piece
            out_box.markdown(text + "▌")
        pill_box.empty()
        out_box.markdown(text)
        return text
    except FriendlyError as e:
        pill_box.empty()
        out_box.warning(str(e))
        if e.detail:
            with detail_box.expander("Technical details"):
                st.code(e.detail)
        return None
    except Exception as e:  # last-resort safety net
        pill_box.empty()
        out_box.error(f"Something went wrong: {e}")
        return None


def read_upload(uploaded):
    """Return (text_or_None, Part_or_None, error_or_None) for an uploaded file."""
    data = uploaded.getvalue()
    if not data:
        return None, None, "That file is empty."
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        return None, None, f"That file is larger than {MAX_FILE_MB} MB."

    name = uploaded.name.lower()
    try:
        if name.endswith(".pdf"):
            return None, types.Part.from_bytes(data=data, mime_type="application/pdf"), None
        if name.endswith((".png", ".jpg", ".jpeg")):
            mime = "image/png" if name.endswith(".png") else "image/jpeg"
            return None, types.Part.from_bytes(data=data, mime_type=mime), None
        if name.endswith(".docx"):
            from docx import Document

            doc = Document(io.BytesIO(data))
            parts = [p.text for p in doc.paragraphs if p.text.strip()]
            for table in doc.tables:
                for row in table.rows:
                    parts.append(" | ".join(c.text.strip() for c in row.cells))
            text = "\n".join(parts).strip()
            return (text, None, None) if text else (None, None, "No readable text in that Word file.")
        text = data.decode("utf-8", errors="ignore").strip()
        return (text, None, None) if text else (None, None, "No readable text in that file.")
    except Exception as e:
        return None, None, f"Could not read that file: {e}"


# ----------------------------------------------------------------------------
# Graph helpers
# ----------------------------------------------------------------------------
CHART_TYPES = {
    "✨ Auto (AI chooses)": None,
    "📊 Bar": "bar",
    "📏 Horizontal bar": "horizontal_bar",
    "📈 Line": "line",
    "🏔️ Area": "area",
    "🥧 Pie": "pie",
    "🔵 Scatter": "scatter",
    "📶 Histogram": "histogram",
}
VALID_KINDS = {"bar", "horizontal_bar", "line", "area", "pie", "scatter", "histogram"}
PALETTE = ["#8B7CF6", "#F08CC8", "#FFB27D", "#7DD3B0", "#6CB4F5", "#F5D76E"]

GRAPH_SYSTEM = """You turn ideas, notes, text and tables into chart data.
Reply with ONLY one valid JSON object (no markdown, no explanation) with these keys:
{"title": str, "chart_type": "bar"|"horizontal_bar"|"line"|"area"|"pie"|"scatter"|"histogram",
 "x_label": str, "y_label": str, "categories": [labels or x numbers],
 "series": [{"name": str, "values": [numbers]}], "summary": "1-2 sentence takeaway",
 "estimated": true|false}
Rules: at most 30 categories and 5 series. Every series must have exactly as many values as
categories (for a histogram, categories can be empty and the values are the raw numbers).
Values must be plain numbers (no units, no % signs). Pick the chart type that fits the data best
unless one is requested. If the user gives no numbers, create reasonable illustrative estimates
and set estimated to true. Otherwise use the given numbers exactly and set estimated to false."""


def to_num(v):
    try:
        return float(str(v).replace(",", "").replace("%", "").strip())
    except (ValueError, TypeError):
        return None


def parse_json(text):
    """Parse the model's JSON, tolerating code fences or extra words around it."""
    text = text.strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    a, b = text.find("{"), text.rfind("}")
    if a != -1 and b > a:
        return json.loads(text[a:b + 1])
    raise ValueError("No JSON found")


def normalize_spec(raw, forced_kind=None):
    """Clean up the AI's chart data so it is always safe to draw."""
    if not isinstance(raw, dict):
        raise ValueError("Chart data was not an object")
    kind = forced_kind or raw.get("chart_type")
    if kind not in VALID_KINDS:
        kind = "bar"

    cats = list(raw.get("categories") or [])[:30]
    series = []
    for s_ in (raw.get("series") or [])[:5]:
        if not isinstance(s_, dict):
            continue
        vals = [to_num(v) for v in (s_.get("values") or [])]
        series.append({"name": str(s_.get("name") or "Value"), "values": vals})
    series = [s_ for s_ in series if any(v is not None for v in s_["values"])]
    if not series:
        raise ValueError("No numbers to plot")

    if kind != "histogram":
        n = len(cats) if cats else max(len(s_["values"]) for s_ in series)
        if not cats:
            cats = [str(i + 1) for i in range(n)]
        for s_ in series:  # shorter series get empty gaps instead of cutting data off
            s_["values"] = (s_["values"] + [None] * n)[:n]
        if kind == "scatter":
            nums = [to_num(c) for c in cats]
            cats = nums if all(v is not None for v in nums) else list(range(1, n + 1))
        else:
            cats = [str(c) for c in cats]

    return {
        "title": str(raw.get("title") or "Your graph"),
        "chart_type": kind,
        "x_label": str(raw.get("x_label") or ""),
        "y_label": str(raw.get("y_label") or ""),
        "categories": cats,
        "series": series,
        "summary": str(raw.get("summary") or ""),
        "estimated": bool(raw.get("estimated")),
    }


PLAN_SYSTEM = """You choose how to chart a table. You are given the column names, their types
and the first rows. Reply with ONLY one valid JSON object (no markdown):
{"title": str, "chart_type": "bar"|"horizontal_bar"|"line"|"area"|"pie"|"scatter"|"histogram",
 "x_column": exact column name or null, "y_columns": [exact numeric column names, at most 5],
 "x_label": str, "y_label": str, "aggregate": "none"|"sum"|"mean"|"count",
 "sort": "none"|"asc"|"desc", "top_n": integer or null}
Rules: use ONLY column names from the list, spelled exactly. Use the user's request to decide
what to show. line/area suit time or ordered data, bar suits categories, pie suits shares of
one total (one y column), scatter needs two numeric columns, histogram needs one numeric
column and x_column null. If the x column repeats (for example several rows per region) use
aggregate sum or mean. Use count to count rows per category when there is no number to plot."""


def clean_number_text(series):
    return (series.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
                  .str.replace(r"(?i)rs\.?|usd|[,$£€%\s]", "", regex=True))


def numeric_clean(df):
    """Turn text columns like '1,200' or '45%' into real numbers when most cells are numeric."""
    out = df.copy()
    for c in out.columns:
        if not pd.api.types.is_numeric_dtype(out[c]) and not pd.api.types.is_datetime64_any_dtype(out[c]):
            raw = out[c].astype(str).str.strip()
            filled = ~raw.isin(["", "nan", "None", "NaN", "NaT"])
            if not filled.any():
                continue
            nums = pd.to_numeric(clean_number_text(raw), errors="coerce")
            if nums[filled].notna().mean() >= 0.8:
                out[c] = nums
    return out


def frame_from_rows(rows):
    """Build a DataFrame from rows of cell text (Word or PDF tables)."""
    rows = [[("" if c is None else str(c).replace("\n", " ").strip()) for c in r] for r in rows if r]
    rows = [r for r in rows if any(r)]
    if len(rows) < 2:
        return None
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    first = rows[0]
    numeric_cells = sum(to_num(x) is not None for x in first)
    if numeric_cells <= len(first) / 2:
        header, body = first, rows[1:]
    else:
        header, body = [f"Column {i + 1}" for i in range(width)], rows
    names, seen = [], {}
    for i, h in enumerate(header):
        h = h or f"Column {i + 1}"
        seen[h] = seen.get(h, 0) + 1
        names.append(h if seen[h] == 1 else f"{h} ({seen[h]})")
    df = pd.DataFrame(body, columns=names)
    df = df.loc[:, (df != "").any(axis=0)]
    df = df[(df != "").any(axis=1)]
    if df.empty:
        return None
    return numeric_clean(df.reset_index(drop=True))


@st.cache_data(show_spinner=False, max_entries=20)
def load_graph_file(name, data):
    """Read any supported file. Returns tables found, plain text and an error (if any)."""
    lower = name.lower()
    result = {"tables": [], "text": "", "mime": None, "error": None}
    try:
        if lower.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(data))
            result["tables"].append(("Data", numeric_clean(df.head(5000))))
        elif lower.endswith(".xlsx"):
            sheets = pd.read_excel(io.BytesIO(data), sheet_name=None)
            for sheet, df in sheets.items():
                df = df.dropna(how="all").dropna(axis=1, how="all")
                if not df.empty:
                    result["tables"].append((f"Sheet: {sheet}", numeric_clean(df.head(5000))))
        elif lower.endswith(".docx"):
            from docx import Document

            doc = Document(io.BytesIO(data))
            for i, table in enumerate(doc.tables, 1):
                df = frame_from_rows([[c.text for c in r.cells] for r in table.rows])
                if df is not None:
                    result["tables"].append((f"Word table {i}", df))
            result["text"] = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        elif lower.endswith(".pdf"):
            result["mime"] = "application/pdf"
            try:
                import pdfplumber

                texts, count = [], 0
                with pdfplumber.open(io.BytesIO(data)) as pdf:
                    for page_no, page in enumerate(pdf.pages[:30], 1):
                        for table in page.extract_tables():
                            df = frame_from_rows(table)
                            if df is not None:
                                count += 1
                                result["tables"].append((f"PDF table {count} (page {page_no})", df))
                        texts.append(page.extract_text() or "")
                result["text"] = "\n".join(texts)
            except ImportError:
                pass
        elif lower.endswith((".txt", ".md")):
            result["text"] = data.decode("utf-8", errors="ignore")
        elif lower.endswith((".png", ".jpg", ".jpeg")):
            result["mime"] = "image/png" if lower.endswith(".png") else "image/jpeg"
        else:
            result["error"] = "That file type is not supported."
    except Exception as e:
        result["error"] = f"Could not read that file: {e}"
    return result


def table_brief(df):
    lines = [f"Rows: {len(df)}", "Columns:"]
    for c in df.columns:
        kind = "number" if pd.api.types.is_numeric_dtype(df[c]) else "text"
        lines.append(f"- {c} ({kind})")
    lines.append("First rows (CSV):")
    lines.append(df.head(25).to_csv(index=False))
    return "\n".join(lines)


def plan_to_spec(df, plan, forced=None):
    """Build chart data from the REAL table, so every number is exact."""
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    lookup = {c.strip().lower(): c for c in df.columns}

    def find(name):
        return lookup.get(str(name).strip().lower()) if name not in (None, "") else None

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    x = find(plan.get("x_column"))
    ys = []
    for n in plan.get("y_columns") or []:
        c = find(n)
        if c and c != x and c in num_cols and c not in ys:
            ys.append(c)
    ys = ys[:5]
    kind = forced or plan.get("chart_type")
    if kind not in VALID_KINDS:
        kind = "bar"
    agg = plan.get("aggregate") if plan.get("aggregate") in ("sum", "mean", "count") else "none"
    sort = plan.get("sort") if plan.get("sort") in ("asc", "desc") else "none"
    try:
        top_n = int(plan.get("top_n")) if plan.get("top_n") else None
    except (ValueError, TypeError):
        top_n = None

    if x is None and kind != "histogram":
        text_cols = [c for c in df.columns if c not in num_cols]
        x = text_cols[0] if text_cols else None
    if not ys:
        ys = [c for c in num_cols if c != x][:3]

    spec = {"chart_type": kind, "summary": "", "estimated": False, "note": "", "source": "table"}

    if kind == "histogram":
        if not ys:
            raise ValueError("No numeric column to plot")
        col = ys[0]
        vals = [float(v) for v in df[col].dropna()]
        spec.update(title=str(plan.get("title") or f"Distribution of {col}"),
                    x_label=str(plan.get("x_label") or col), y_label="Count", categories=[],
                    series=[{"name": col, "values": vals}])
        return spec

    if not ys and agg != "count":
        raise ValueError("No numeric columns to plot")

    if x is None:
        work = df[ys].copy()
        work.insert(0, "Row", range(1, len(work) + 1))
        x = "Row"
    else:
        work = df[[x] + ys].copy()
    if pd.api.types.is_datetime64_any_dtype(work[x]):
        work[x] = work[x].dt.strftime("%Y-%m-%d")

    if agg == "count" or not ys:
        g = work.groupby(x, sort=False).size().reset_index(name="Count")
        ys = ["Count"]
    elif agg in ("sum", "mean"):
        g = work.groupby(x, sort=False)[ys].agg(agg).reset_index()
    else:
        g = work.dropna(subset=ys, how="all")

    if top_n and sort == "none":
        sort = "desc"
    if sort != "none":
        g = g.sort_values(ys[0], ascending=(sort == "asc"), kind="stable")
    if top_n:
        g = g.head(top_n)
    limit = 400 if kind in ("line", "area", "scatter") else 40
    if len(g) > limit:
        spec["note"] = f"Showing the first {limit} of {len(g)} rows to keep the graph readable."
        g = g.head(limit)

    if kind == "scatter" and pd.api.types.is_numeric_dtype(g[x]):
        cats = [float(v) for v in g[x]]
    elif kind == "scatter":
        cats = list(range(1, len(g) + 1))
    else:
        cats = [str(v) for v in g[x]]
    series = [{"name": c, "values": [None if pd.isna(v) else float(v) for v in g[c]]} for c in ys]
    spec.update(
        title=str(plan.get("title") or f"{', '.join(ys)} by {x}"),
        x_label=str(plan.get("x_label") or x),
        y_label=str(plan.get("y_label") or (ys[0] if len(ys) == 1 else "")),
        categories=cats, series=series,
    )
    return spec


def fmt_num(v):
    return f"{v:,.2f}".rstrip("0").rstrip(".")


def quick_facts(spec):
    """Simple facts worked out by the app itself (never by the AI), so they are exact."""
    first = spec["series"][0]
    kind = spec["chart_type"]
    if kind == "histogram":
        vals = [v for v in first["values"] if v is not None]
        if not vals:
            return ""
        return (f"📌 **Count:** {len(vals)} · **Lowest:** {fmt_num(min(vals))} · "
                f"**Highest:** {fmt_num(max(vals))} · **Average:** {fmt_num(sum(vals) / len(vals))}")
    pairs = [(c, v) for c, v in zip(spec["categories"], first["values"]) if v is not None]
    if not pairs:
        return ""
    hi = max(pairs, key=lambda p: p[1])
    lo = min(pairs, key=lambda p: p[1])
    total = sum(v for _, v in pairs)
    who = f" ({first['name']})" if len(spec["series"]) > 1 else ""
    if kind == "scatter":
        return (f"📌 **Points:** {len(pairs)} · **Lowest{who}:** {fmt_num(lo[1])} · "
                f"**Highest{who}:** {fmt_num(hi[1])}")
    return (f"📌 **Highest{who}:** {hi[0]} ({fmt_num(hi[1])}) · **Lowest:** {lo[0]} ({fmt_num(lo[1])}) · "
            f"**Total:** {fmt_num(total)} · **Average:** {fmt_num(total / len(pairs))}")


def build_figure(spec):
    kind, cats, series = spec["chart_type"], spec["categories"], spec["series"]
    many = len(cats) > 12
    fig = go.Figure()
    for i, s_ in enumerate(series):
        color = PALETTE[i % len(PALETTE)]
        vals = s_["values"]
        if kind == "pie":
            fig.add_trace(go.Pie(labels=cats, values=vals, hole=0.4, name=s_["name"], sort=False,
                                 marker=dict(colors=PALETTE * 6, line=dict(color="white", width=2)),
                                 textinfo="label+percent"))
            break
        if kind == "histogram":
            fig.add_trace(go.Histogram(x=vals, name=s_["name"], opacity=0.9,
                                       marker=dict(color=color, line=dict(color="white", width=1))))
        elif kind == "scatter":
            fig.add_trace(go.Scatter(x=cats, y=vals, name=s_["name"], mode="markers",
                                     marker=dict(size=12, color=color, line=dict(width=1.5, color="white"))))
        elif kind == "line":
            fig.add_trace(go.Scatter(x=cats, y=vals, name=s_["name"],
                                     mode="lines" if len(cats) > 40 else "lines+markers",
                                     line=dict(width=3, color=color), marker=dict(size=8)))
        elif kind == "area":
            fig.add_trace(go.Scatter(x=cats, y=vals, name=s_["name"], mode="lines",
                                     line=dict(width=2, color=color), fill="tozeroy"))
        elif kind == "horizontal_bar":
            extra = {} if many else dict(texttemplate="%{x:,.4~g}", textposition="outside", cliponaxis=False)
            fig.add_trace(go.Bar(x=vals, y=cats, name=s_["name"], orientation="h",
                                 marker=dict(color=color), **extra))
        else:
            extra = {} if many else dict(texttemplate="%{y:,.4~g}", textposition="outside", cliponaxis=False)
            fig.add_trace(go.Bar(x=cats, y=vals, name=s_["name"], marker=dict(color=color), **extra))

    x_title, y_title = spec["x_label"], spec["y_label"]
    if kind == "horizontal_bar":
        x_title, y_title = y_title, x_title
    fig.update_layout(
        title=dict(text=spec["title"], x=0.01, xanchor="left", font=dict(size=22, color="#2D2A4A")),
        template="plotly_white", barmode="group", colorway=PALETTE,
        font=dict(family="Segoe UI, Inter, sans-serif", color="#2D2A4A", size=13),
        paper_bgcolor="rgba(255,255,255,0)", plot_bgcolor="rgba(255,255,255,0.7)",
        height=480, margin=dict(l=50, r=30, t=80, b=60),
        showlegend=(len(series) > 1 or kind == "pie"),
        legend=dict(orientation="h", y=-0.22, x=0),
        hoverlabel=dict(bgcolor="white", font_size=13),
    )
    if kind in ("line", "area"):
        fig.update_layout(hovermode="x unified")
    if kind != "pie":
        fig.update_xaxes(title_text=x_title, gridcolor="#EEE9FF", zeroline=False, automargin=True)
        fig.update_yaxes(title_text=y_title, gridcolor="#EEE9FF", zeroline=False, automargin=True)
        if many and kind in ("bar", "line", "area"):
            fig.update_xaxes(tickangle=-35)
    return fig


def spec_to_frame(spec):
    if spec["chart_type"] == "histogram":
        return pd.DataFrame({s_["name"]: pd.Series(s_["values"]) for s_ in spec["series"]})
    data = {"label": spec["categories"]}
    for s_ in spec["series"]:
        data[s_["name"]] = s_["values"]
    return pd.DataFrame(data)


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    api_key = ""
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        pass
    if not api_key:
        api_key = st.text_input("Gemini API key", type="password",
                                help="Free key at aistudio.google.com")

    model_choice = st.selectbox("Model", MODELS)
    custom_model = st.text_input("Custom model name (optional)",
                                 placeholder="e.g. gemini-3.6-flash")
    temperature = st.slider("Creativity", 0.0, 2.0, 1.0, 0.1)
    speed = st.selectbox(
        "Speed",
        list(SPEEDS.keys()),
        help="Fast answers quickly. Deep thinking is slower but better for hard questions.",
    )
    thinking_level = SPEEDS[speed]

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

    st.caption("Get a free key at aistudio.google.com. Never share it publicly.")

# Chosen model first, then the rest as backups
first = custom_model.strip() or model_choice
model_list = [first] + [m for m in MODELS if m != first]

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    '<div class="hero"><h1>✨ Idea Spark AI</h1>'
    "<p>Spark new ideas, chat with AI and use handy writing tools, powered by Gemini Flash.</p></div>",
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def get_client(key):
    return genai.Client(api_key=key)


client = get_client(api_key) if api_key else None
tab_chat, tab_tools, tab_graph = st.tabs(["💬 Chat", "🛠️ Writing Tools", "📊 Graphs"])

# ----------------------------------------------------------------------------
# Chat tab
# ----------------------------------------------------------------------------
with tab_chat:
    chat_area = st.container()  # messages live here, above the input box
    prompt = st.chat_input("Ask me anything...")

    with chat_area:
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        if prompt:
            if not client:
                st.info("🔑 Add your Gemini API key in the sidebar to start.")
            else:
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                history = [
                    types.Content(
                        role="user" if m["role"] == "user" else "model",
                        parts=[types.Part.from_text(text=m["content"])],
                    )
                    for m in st.session_state.messages[-MAX_HISTORY:]
                ]
                with st.chat_message("assistant"):
                    reply = run_stream(client, history, temperature, model_list,
                                       "✨ Spark is thinking", system=CHAT_SYSTEM,
                                       thinking=thinking_level)
                if reply:
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                else:
                    st.session_state.messages.pop()  # drop the unanswered question

# ----------------------------------------------------------------------------
# Writing tools tab
# ----------------------------------------------------------------------------
with tab_tools:
    st.markdown("#### Pick a tool, add your text or file, and press Run")
    task_name = st.selectbox("What would you like to do?", list(TASKS.keys()))

    language = ""
    if task_name == "🌍 Translate":
        language = st.text_input("Translate into which language?", placeholder="e.g. Urdu, French")

    if "Expert level" in task_name or "expert level" in task_name:
        st.caption("⚠️ AI can get references wrong. Always verify them before relying on them.")

    source = st.radio("Input", ["✍️ Paste text", "📎 Upload file"], horizontal=True,
                      label_visibility="collapsed")

    text_in, file_in = "", None
    if source == "✍️ Paste text":
        text_in = st.text_area("Your text", height=220, placeholder="Paste your text here...")
    else:
        file_in = st.file_uploader(
            f"Upload a file (up to {MAX_FILE_MB} MB)",
            type=["pdf", "docx", "txt", "md", "csv", "png", "jpg", "jpeg"],
        )

    if st.button("🚀 Run"):
        if not client:
            st.info("🔑 Add your Gemini API key in the sidebar to start.")
        elif task_name == "🌍 Translate" and not language.strip():
            st.warning("Please type the language you want to translate into.")
        else:
            instruction = TASKS[task_name].format(language=language.strip() or "English")
            contents = None

            if source == "✍️ Paste text":
                if not text_in.strip():
                    st.warning("Please paste some text first.")
                else:
                    contents = [types.Part.from_text(text=f"{instruction}\n\nText:\n{text_in}")]
            else:
                if not file_in:
                    st.warning("Please upload a file first.")
                else:
                    file_text, file_part, err = read_upload(file_in)
                    if err:
                        st.warning(err)
                    elif file_part is not None:
                        contents = [types.Part.from_text(text=instruction + "\n\nThe content is in the attached file."),
                                    file_part]
                    else:
                        contents = [types.Part.from_text(text=f"{instruction}\n\nText:\n{file_text}")]

            if contents:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                run_stream(
                    client, contents, temperature, model_list, "✨ Crafting your result",
                    system="Follow the task exactly. Output only the result, formatted in clean Markdown.",
                    thinking=thinking_level,
                )
                st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Graphs tab
# ----------------------------------------------------------------------------
with tab_graph:
    st.markdown("#### Turn your data into clear, accurate graphs")
    st.caption("Upload Excel, CSV, Word, PDF or an image, or just describe an idea. "
               "Tables inside your files are charted with their exact numbers.")
    g_source = st.radio("Where is your data?", ["📎 Upload a file", "✍️ Type an idea or paste data"],
                        horizontal=True, label_visibility="collapsed")

    g_file, g_text, raw_bytes = None, "", b""
    if g_source.startswith("📎"):
        g_file = st.file_uploader(
            f"Upload a file (up to {MAX_FILE_MB} MB)",
            type=["xlsx", "csv", "docx", "pdf", "txt", "md", "png", "jpg", "jpeg"], key="graph_file",
        )
        g_text = st.text_input("What should the graph show? (optional)",
                               placeholder="e.g. sales by region, or compare 2023 with 2024")
    else:
        g_text = st.text_area(
            "Describe your idea or paste your information", height=150,
            placeholder="e.g. Compare monthly sales: Product A 120, 150, 170 / Product B 90, 110, 95 "
                        "(Jan to Mar)\nor: Show how a student could split a 24-hour day.",
        )
    g_kind_name = st.selectbox("Chart type", list(CHART_TYPES.keys()))

    source, table_df, file_problem = None, None, None
    if g_file:
        raw_bytes = g_file.getvalue()
        if not raw_bytes:
            file_problem = "That file is empty."
        elif len(raw_bytes) > MAX_FILE_MB * 1024 * 1024:
            file_problem = f"That file is larger than {MAX_FILE_MB} MB."
        else:
            source = load_graph_file(g_file.name, raw_bytes)
            file_problem = source["error"]
        if file_problem:
            st.warning(file_problem)
        elif source["tables"]:
            labels = [f"{label} ({len(df)} rows × {len(df.columns)} columns)"
                      for label, df in source["tables"]]
            pick = 0
            if len(labels) > 1:
                pick = labels.index(st.selectbox("Which table should I chart?", labels, key="graph_table"))
            table_df = source["tables"][pick][1]
            with st.expander("👀 Preview of the table I found"):
                st.dataframe(table_df.head(50), use_container_width=True)
        else:
            st.info("I did not find a table in this file, so AI will read the numbers from its content. "
                    "Please check the result against your original.")

    if st.button("📊 Create graph"):
        forced = CHART_TYPES[g_kind_name]
        if not client:
            st.info("🔑 Add your Gemini API key in the sidebar to start.")
        elif not g_file and not g_text.strip():
            st.warning("Please upload a file or describe your idea first.")
        elif file_problem:
            st.warning("Please fix the file problem above first.")
        else:
            box = st.empty()
            box.markdown(pill("📊 Drawing your graph"), unsafe_allow_html=True)

            def wait_msg(msg):
                box.markdown(pill(msg), unsafe_allow_html=True)

            hint = f"\nUse a {forced.replace('_', ' ')} chart." if forced else ""
            request = g_text.strip() or "Choose the most useful chart for this data."
            try:
                if table_df is not None:
                    # Exact path: the AI only picks WHAT to chart, the app reads the real numbers.
                    prompt = f"User request: {request}{hint}\n\n{table_brief(table_df)}"
                    plan, ai_note = {}, ""
                    try:
                        text = "".join(stream_gemini(
                            client, [types.Part.from_text(text=prompt)], 0.2, model_list,
                            system=PLAN_SYSTEM, thinking=thinking_level, on_wait=wait_msg, json_mode=True))
                        plan = parse_json(text)
                    except (FriendlyError, ValueError):
                        ai_note = "AI was busy, so a simple chart was chosen automatically. Try again for a smarter one."
                    spec = plan_to_spec(table_df, plan if isinstance(plan, dict) else {}, forced)
                    if ai_note:
                        spec["note"] = (spec["note"] + " " + ai_note).strip()
                else:
                    # Reading path: AI reads numbers from text, a PDF or an image.
                    parts = [types.Part.from_text(text=f"{request}{hint}")]
                    if source and source["text"]:
                        parts.append(types.Part.from_text(text="Document text:\n" + source["text"][:30000]))
                    if source and source["mime"] and not source["tables"]:
                        parts.append(types.Part.from_bytes(data=raw_bytes, mime_type=source["mime"]))
                    text = "".join(stream_gemini(
                        client, parts, 0.3, model_list, system=GRAPH_SYSTEM, thinking=thinking_level,
                        on_wait=wait_msg, json_mode=True))
                    spec = normalize_spec(parse_json(text), forced)
                    spec["source"] = "ai"
                st.session_state.graph = spec
                box.empty()
            except FriendlyError as e:
                box.empty()
                st.warning(str(e))
                if e.detail:
                    with st.expander("Technical details"):
                        st.code(e.detail)
            except (ValueError, KeyError, TypeError, AttributeError):
                box.empty()
                st.warning("I could not turn that into a graph. Check that your data has some numbers, "
                           "or tell me what you want to compare.")

    spec = st.session_state.get("graph")
    if spec:
        st.plotly_chart(build_figure(spec), use_container_width=True,
                        config={"displaylogo": False,
                                "toImageButtonOptions": {"format": "png", "scale": 3,
                                                         "filename": "idea-spark-graph"}})
        facts = quick_facts(spec)
        if facts:
            st.markdown(facts)
        if spec.get("source") == "table":
            st.caption("✅ Every value in this graph comes straight from your table.")
        else:
            st.caption("⚠️ AI read these numbers from your text or document"
                       + (" and some are estimates" if spec.get("estimated") else "")
                       + ". Please compare them with your original in the data table below.")
        if spec.get("note"):
            st.caption("ℹ️ " + spec["note"])
        frame = spec_to_frame(spec)
        with st.expander("🔢 See the data used in this graph"):
            st.dataframe(frame, use_container_width=True)
        d1, d2 = st.columns(2)
        with d1:
            st.download_button("⬇️ Download data (CSV)", frame.to_csv(index=False),
                               file_name="idea-spark-data.csv", mime="text/csv")
        with d2:
            st.download_button("⬇️ Download graph (HTML)",
                               build_figure(spec).to_html(include_plotlyjs="cdn"),
                               file_name="idea-spark-graph.html", mime="text/html")
        st.caption("Tip: hover over the graph and click the camera icon to save a sharp PNG image.")
