"""
SpamDetect — revamped SMS spam classifier UI
Dark terminal aesthetic | portfolio-ready
"""
import re
import streamlit as st
from src.predict import predict_sms, models_loaded
from src.preprocess import extract_signals


def _md(html: str) -> None:
    """Collapse whitespace then render as HTML via st.markdown.

    Streamlit's markdown parser treats lines starting with 4+ spaces as code
    blocks. Any multiline indented f-string would trigger that and dump raw
    HTML into the UI. Stripping excess whitespace between tags before handing
    off to st.markdown prevents this entirely.
    """
    clean = re.sub(r">\s+<", "><", html.strip())   # whitespace between tags
    clean = re.sub(r"\n\s+", " ", clean)            # indented continuation lines
    st.markdown(clean, unsafe_allow_html=True)

# ─── Page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SpamDetect",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ─────────────────────────────────────────────────────────────

_md("""
<style>
  /* ── Base ── */
  html, body, [data-testid="stApp"]                { background: #111210 !important; }
  [data-testid="stSidebar"]                        { background: #0f0f0d !important; border-right: 0.5px solid #2e2e2a; }
  [data-testid="stMain"]                           { background: #111210 !important; }

  /* ── Typography ── */
  *, p, div, span, label                           { font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace !important; }
  h1, h2, h3                                       { color: #d4d3cc !important; }

  /* ── Reduce block spacing ── */
  .block-container                                 { padding: 1.2rem 2rem 2rem !important; max-width: 1200px !important; }
  [data-testid="stVerticalBlock"] > div            { gap: 0.4rem !important; }

  /* ── Inputs ── */
  textarea, .stTextArea textarea {
    background: #161614 !important;
    border: 0.5px solid #2e2e2a !important;
    border-radius: 6px !important;
    color: #d4d3cc !important;
    font-size: 13px !important;
    line-height: 1.6 !important;
    caret-color: #EF9F27;
    padding: 10px 12px !important;
  }
  textarea:focus { border-color: #EF9F27 !important; box-shadow: none !important; }

  /* ── Button ── */
  .stButton button {
    background: #1c1c1a !important;
    border: 0.5px solid #3e3e38 !important;
    color: #d4d3cc !important;
    border-radius: 6px !important;
    font-size: 12px !important;
    width: 100% !important;
    padding: 7px 14px !important;
    transition: border-color .2s, color .2s !important;
  }
  .stButton button:hover {
    border-color: #EF9F27 !important;
    color: #EF9F27 !important;
    background: #1c1c1a !important;
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    background: #1a1a18 !important;
    border-bottom: 0.5px solid #2e2e2a !important;
    gap: 0 !important;
  }
  /* Target the actual button elements so pointer-events work */
  .stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"] {
    background: transparent !important;
    background-color: transparent !important;
    color: #5a5a54 !important;
    font-size: 11px !important;
    letter-spacing: .04em !important;
    font-family: 'SF Mono','Fira Code','Consolas',monospace !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 10px 20px !important;
    cursor: pointer !important;
    pointer-events: all !important;
    box-shadow: none !important;
    outline: none !important;
  }
  /* Selected tab — use box-shadow instead of border-bottom (more reliable) */
  .stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"][aria-selected="true"] {
    background: transparent !important;
    background-color: transparent !important;
    color: #EF9F27 !important;
    border-bottom: 2px solid #EF9F27 !important;
    box-shadow: none !important;
  }
  .stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"]:hover {
    color: #d4d3cc !important;
    background: transparent !important;
  }
  /* Remove the default blue focus ring Streamlit adds */
  .stTabs [data-baseweb="tab-list"] button:focus,
  .stTabs [data-baseweb="tab-list"] button:focus-visible {
    outline: none !important;
    box-shadow: none !important;
  }
  .stTabs [data-baseweb="tab-panel"] { padding: 1rem 0 0 !important; }

  /* ── Progress / spinner ── */
  .stSpinner > div                                 { border-top-color: #EF9F27 !important; }
  .stProgress > div > div > div                    { background: #EF9F27 !important; }

  /* ── Misc ── */
  [data-testid="stMarkdownContainer"] a            { color: #EF9F27 !important; }
  .stDownloadButton button                         { color: #1D9E75 !important; border-color: #1D9E75 !important; }
  footer { display: none !important; }
  #MainMenu { visibility: hidden !important; }
  [data-testid="stDecoration"] { display: none !important; }
  [data-testid="stHeader"] { background: transparent !important; }
</style>
""")

# ─── Shared HTML components ──────────────────────────────────────────────────

def _label_tag(text: str, color: str, bg: str) -> str:
    return (
        f'<span style="display:inline-block;background:{bg};color:{color};'
        f'border:0.5px solid {color};border-radius:4px;padding:2px 8px;'
        f'font-size:10px;letter-spacing:.07em;margin-right:5px">{text}</span>'
    )


def _section_label(text: str) -> str:
    return (
        f'<p style="font-size:10px;color:#5a5a54;letter-spacing:.08em;'
        f'text-transform:uppercase;margin:0 0 8px">{text}</p>'
    )


def _signal_chips(signals: dict) -> str:
    chips = []
    defs = [
        ("has_url",      "URL detected",   "#C84A1A", "#221205"),
        ("has_phone",    "Phone no.",       "#BA7517", "#231800"),
        ("has_currency", "Currency",        "#BA7517", "#231800"),
        ("has_email",    "Email addr.",     "#BA7517", "#231800"),
    ]
    for key, label, color, bg in defs:
        active = signals.get(key, False)
        dim = "opacity:.35" if not active else ""
        chips.append(
            f'<span style="background:{bg if active else "#1c1c1a"};'
            f'border:0.5px solid {"#3a1e0a" if active else "#2e2e2a"};'
            f'border-radius:4px;padding:4px 10px;font-size:11px;'
            f'color:{"#C84A1A" if active else "#5a5a54"};{dim};'
            f'display:inline-flex;align-items:center;gap:5px">'
            f'<span style="width:5px;height:5px;border-radius:50%;'
            f'background:currentColor;display:inline-block"></span>'
            f'{label}</span>'
        )
    # caps ratio chip
    caps = signals.get("caps_ratio", 0)
    if caps > 0.3:
        chips.append(
            f'<span style="background:#221205;border:0.5px solid #3a1e0a;'
            f'border-radius:4px;padding:4px 10px;font-size:11px;color:#C84A1A;'
            f'display:inline-flex;align-items:center;gap:5px">'
            f'<span style="width:5px;height:5px;border-radius:50%;'
            f'background:currentColor;display:inline-block"></span>'
            f'Shouting ({int(caps*100)}% caps)</span>'
        )
    return " ".join(chips)


def _highlighted_message(message: str, anatomy: list) -> str:
    """Wrap known spam trigger tokens in the message with highlight spans."""
    if not anatomy:
        return f'<span style="color:#d4d3cc;font-size:13px;line-height:1.7">{_escape(message)}</span>'

    trigger_tokens = {item["token"] for item in anatomy if not item["token"].startswith("[")}
    # Sort by length descending to avoid partial replacements
    sorted_tokens = sorted(trigger_tokens, key=len, reverse=True)

    html = _escape(message)
    for tok in sorted_tokens:
        if len(tok) < 2:
            continue
        pattern = re.compile(re.escape(tok), re.IGNORECASE)
        html = pattern.sub(
            lambda m: (
                f'<mark style="background:#2a1a05;color:#EF9F27;'
                f'border-radius:2px;padding:0 2px">{_escape(m.group())}</mark>'
            ),
            html,
        )
    return f'<span style="color:#d4d3cc;font-size:13px;line-height:1.7">{html}</span>'


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _verdict_panel(result: dict):
    label     = result["label"]
    spam_prob = int(result["spam_prob"] * 100)
    anatomy   = result["anatomy"]

    is_spam   = label == "SPAM"
    v_color   = "#EF9F27" if is_spam else "#1D9E75"
    v_bg      = "#221205" if is_spam else "#0d1e16"
    v_border  = "#3a1e0a" if is_spam else "#0f3020"
    icon      = "⚠" if is_spam else "✓"
    bar_color = "#EF9F27" if is_spam else "#1D9E75"

    # ── Each section is a single-line HTML string. ───────────────────────────
    # IMPORTANT: never pass multiline indented strings to st.markdown —
    # Streamlit's markdown parser treats lines with 4+ leading spaces as
    # code blocks, which is what caused the raw HTML dump in the UI.

    # Header row
    st.markdown(
        f'<div style="background:{v_bg};border:0.5px solid {v_border};border-radius:6px 6px 0 0;'
        f'padding:10px 14px;display:flex;align-items:center;justify-content:space-between;margin-top:14px">'
        f'<span style="font-size:10px;color:#6b4020;letter-spacing:.08em;text-transform:uppercase">classification</span>'
        f'<span style="font-size:15px;color:{v_color};font-weight:500;letter-spacing:.05em">{icon} {label}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Confidence bar
    st.markdown(
        f'<div style="background:#161614;border-left:0.5px solid #2e2e2a;border-right:0.5px solid #2e2e2a;'
        f'padding:10px 14px;display:flex;align-items:center;gap:10px">'
        f'<span style="font-size:10px;color:#5a5a54;text-transform:uppercase;letter-spacing:.06em;white-space:nowrap">Spam probability</span>'
        f'<div style="flex:1;height:4px;background:#2a2a26;border-radius:2px;overflow:hidden">'
        f'<div style="height:100%;width:{spam_prob}%;background:{bar_color};border-radius:2px"></div>'
        f'</div>'
        f'<span style="font-size:12px;color:{v_color};min-width:38px;text-align:right">{spam_prob}%</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Anatomy panel
    if anatomy:
        tags_html = "".join(
            f'<span style="background:{"#2a1a05" if item["weight"] > 0.05 else "#221205"};'
            f'border:0.5px solid {"#EF9F27" if item["weight"] > 0.05 else "#3a1e0a"};'
            f'border-radius:4px;padding:3px 8px;font-size:11px;'
            f'color:{"#EF9F27" if item["weight"] > 0.05 else "#C84A1A"};'
            f'display:inline-flex;align-items:center;gap:5px">'
            f'{_escape(item["token"])}'
            f'<span style="font-size:10px;color:#6b4020">+{item["weight"]:.3f}</span>'
            f'</span>'
            for item in anatomy[:8]
        )
        st.markdown(
            f'<div style="background:#161614;border:0.5px solid #2e2e2a;border-top:none;'
            f'border-radius:0 0 6px 6px;padding:10px 14px">'
            f'<p style="font-size:10px;color:#5a5a54;letter-spacing:.08em;text-transform:uppercase;margin:0 0 8px">spam anatomy — top triggers</p>'
            f'<div style="display:flex;flex-wrap:wrap;gap:6px">{tags_html}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        # Close the card border
        st.markdown(
            '<div style="background:#161614;border-left:0.5px solid #2e2e2a;'
            'border-right:0.5px solid #2e2e2a;border-bottom:0.5px solid #2e2e2a;'
            'border-radius:0 0 6px 6px;height:4px"></div>',
            unsafe_allow_html=True,
        )


# ─── Session state ───────────────────────────────────────────────────────────

def _init_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    # message_buffer: the source of truth for the textarea value.
    # Example buttons write here; text_area reads from here via value=.
    # We NEVER write to st.session_state["message_input"] (the widget key)
    # after the widget is instantiated — that causes the StreamlitAPIException.
    if "message_buffer" not in st.session_state:
        st.session_state.message_buffer = ""
    if "last_result" not in st.session_state:
        st.session_state.last_result = None


# ─── Sidebar ─────────────────────────────────────────────────────────────────

def _sidebar():
    with st.sidebar:
        st.markdown(
            '<p style="font-size:16px;font-weight:500;color:#d4d3cc;margin:0 0 4px">SpamDetect</p>'
            '<p style="font-size:10px;color:#5a5a54;letter-spacing:.06em;margin:0 0 20px">'
            'SMS CLASSIFIER · 138K TRAINED</p>',
            unsafe_allow_html=True,
        )

        history = st.session_state.history
        total   = len(history)
        n_spam  = sum(1 for r in history if r["label"] == "SPAM")
        spam_rate = int(n_spam / max(total, 1) * 100)
        avg_conf  = int(sum(r["confidence"] for r in history) / max(total, 1) * 100)

        _md(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:14px">
          <div style="background:#161614;border-radius:6px;padding:10px 12px">
            <p style="font-size:18px;color:#EF9F27;font-weight:500;margin:0;line-height:1">{total}</p>
            <p style="font-size:10px;color:#5a5a54;text-transform:uppercase;letter-spacing:.06em;margin:3px 0 0">analysed</p>
          </div>
          <div style="background:#161614;border-radius:6px;padding:10px 12px">
            <p style="font-size:18px;color:{"#EF9F27" if n_spam else "#d4d3cc"};font-weight:500;margin:0;line-height:1">{spam_rate}%</p>
            <p style="font-size:10px;color:#5a5a54;text-transform:uppercase;letter-spacing:.06em;margin:3px 0 0">spam rate</p>
          </div>
          <div style="background:#161614;border-radius:6px;padding:10px 12px;grid-column:1/-1">
            <p style="font-size:18px;color:#d4d3cc;font-weight:500;margin:0;line-height:1">{avg_conf}%</p>
            <p style="font-size:10px;color:#5a5a54;text-transform:uppercase;letter-spacing:.06em;margin:3px 0 0">avg confidence</p>
          </div>
        </div>
        """)

        # History list
        if history:
            st.markdown(_section_label("session history"), unsafe_allow_html=True)
            for r in reversed(history[-20:]):
                is_spam  = r["label"] == "SPAM"
                dot_col  = "#EF9F27" if is_spam else "#1D9E75"
                excerpt  = r["message"][:38] + "…" if len(r["message"]) > 38 else r["message"]
                conf_pct = int(r["confidence"] * 100)
                _md(f"""
                <div style="display:flex;align-items:center;gap:6px;padding:5px 8px;
                            background:#161614;border-radius:4px;border:0.5px solid #2a2a26;margin-bottom:5px">
                  <span style="width:6px;height:6px;border-radius:50%;background:{dot_col};
                               flex-shrink:0;display:inline-block"></span>
                  <span style="font-size:11px;color:#6b6a64;overflow:hidden;text-overflow:ellipsis;
                               white-space:nowrap;flex:1" title="{_escape(r['message'])}">{_escape(excerpt)}</span>
                  <span style="font-size:10px;color:#4a4a44">{conf_pct}%</span>
                </div>
                """)

        # Clear button
        if history:
            st.markdown("<div style='margin-top:10px'>", unsafe_allow_html=True)
            if st.button("Clear history", key="clear_hist"):
                st.session_state.history = []
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        # Model info
        st.markdown("<div style='margin-top:20px'>", unsafe_allow_html=True)
        st.markdown(_section_label("model"), unsafe_allow_html=True)
        _md("""
        <div style="font-size:11px;color:#5a5a54;line-height:1.8">
          LinearSVC + CalibratedCV<br>
          TF-IDF word (1-3) · char (3-5)<br>
          + 7 metadata features<br>
          SMOTE balanced · threshold-tuned<br>
          Dataset: 138k messages
        </div>
        """)
        st.markdown("</div>", unsafe_allow_html=True)


# ─── Tab: Analyse ────────────────────────────────────────────────────────────

EXAMPLES = [
    ("Prize scam",    "Congratulations! You've WON a FREE iPhone 15. Click http://bit.ly/cl4im-pr1ze to claim your reward NOW before it expires at midnight!"),
    ("Bank phishing", "ALERT: Your SBI account has been suspended. Verify your KYC immediately at http://sbi-verify.co/update or lose access within 24hrs."),
    ("OTP fraud",     "Your OTP for bank transaction is 847291. NEVER share this with anyone. If not initiated by you call 1800-XXXX immediately."),
    ("Ham – friend",  "Hey! Are you free for lunch today? There's a new place on MG Road I've been wanting to try."),
    ("Ham – delivery","Your Amazon package #405-1234 has been delivered to your door. If you didn't receive it, contact support."),
]


def _analyse_tab():
    st.markdown(
        '<p style="font-size:19px;font-weight:500;color:#d4d3cc;margin:0 0 2px">'
        '🛡 SpamDetect</p>'
        '<p style="font-size:12px;color:#5a5a54;margin:0 0 18px">'
        'Paste any SMS — the model classifies it and explains why.</p>',
        unsafe_allow_html=True,
    )

    col_main, col_right = st.columns([5, 2], gap="large")

    with col_right:
        # ── Example buttons ──────────────────────────────────────────────────
        # FIX: write to st.session_state.message_buffer (NOT the widget key).
        # Setting st.session_state["message_input"] after the widget is
        # rendered raises StreamlitAPIException. The buffer is read back via
        # value= on the text_area below, which is safe.
        st.markdown(_section_label("try an example"), unsafe_allow_html=True)
        for label, example in EXAMPLES:
            is_spam_eg = any(k in label.lower() for k in ["scam", "phishing", "fraud", "otp"])
            dot = "🟠" if is_spam_eg else "🟢"
            if st.button(f"{dot} {label}", key=f"eg_{label}"):
                st.session_state.message_buffer = example
                st.session_state.last_result = None   # clear old result
                st.rerun()

        st.markdown(
            '<div style="margin-top:20px;padding-top:16px;'
            'border-top:0.5px solid #2a2a26">',
            unsafe_allow_html=True,
        )
        st.markdown(_section_label("signal legend"), unsafe_allow_html=True)
        _md("""
        <div style="font-size:11px;color:#5a5a54;line-height:2.1">
          <span style="color:#C84A1A">■</span>&nbsp; URL detected<br>
          <span style="color:#BA7517">■</span>&nbsp; Phone / currency<br>
          <span style="color:#EF9F27">■</span>&nbsp; Spam trigger word<br>
          <span style="color:#1D9E75">■</span>&nbsp; Ham (safe)
        </div>
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_main:
        # ── Text area ────────────────────────────────────────────────────────
        # Use value= to populate from the buffer. Do NOT use key= together
        # with manual session_state writes — that's what caused the crash.
        message = st.text_area(
            label="Message input",
            label_visibility="collapsed",
            placeholder="Paste or type an SMS message here…",
            value=st.session_state.message_buffer,
            height=120,
        )
        # Sync buffer so it survives reruns (user edits are preserved)
        st.session_state.message_buffer = message

        # ── Live signal chips — only if there's text ─────────────────────────
        if message.strip():
            sigs = extract_signals(message)
            char_info = (
                f'<span style="font-size:11px;color:#4a4a44;margin-left:4px">'
                f'{sigs["char_count"]} chars · {sigs["word_count"]} words</span>'
            )
            st.markdown(
                f'<div style="display:flex;align-items:center;flex-wrap:wrap;'
                f'gap:6px;margin:8px 0 12px">'
                f'{_signal_chips(sigs)}{char_info}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown("<div style='margin-bottom:14px'></div>", unsafe_allow_html=True)

        classify_btn = st.button("▶  analyse message →", key="classify_btn")

        # ── Validate ─────────────────────────────────────────────────────────
        if classify_btn and not message.strip():
            st.markdown(
                '<div style="background:#1e1a0e;border:0.5px solid #3a3010;'
                'border-radius:6px;padding:10px 14px;font-size:12px;color:#9a8a40;'
                'margin-top:8px">⚠ Please enter a message first.</div>',
                unsafe_allow_html=True,
            )
            return

        # ── Classify ─────────────────────────────────────────────────────────
        if classify_btn and message.strip():
            if not models_loaded():
                st.error("No trained model found. Run `python -m src.train` first.")
                return
            with st.spinner(""):
                result = predict_sms(message)
                result["message"] = message
                st.session_state.history.append(result)
                st.session_state.last_result = result

        # ── Show result (persists until next message) ─────────────────────────
        result = st.session_state.last_result
        if result:
            st.markdown(
                '<hr style="border:none;border-top:0.5px solid #2a2a26;margin:14px 0 12px">',
                unsafe_allow_html=True,
            )
            st.markdown(_section_label("message preview — highlighted triggers"), unsafe_allow_html=True)
            st.markdown(
                f'<div style="background:#161614;border:0.5px solid #2a2a26;'
                f'border-radius:6px;padding:12px 14px;margin-bottom:2px;line-height:1.8">'
                f'{_highlighted_message(result["message"], result["anatomy"])}</div>',
                unsafe_allow_html=True,
            )
            _verdict_panel(result)


# ─── Tab: Batch ──────────────────────────────────────────────────────────────

def _batch_tab():
    st.markdown(
        '<p style="font-size:13px;color:#5a5a54;margin:0 0 12px">'
        'One message per line. Max 200 at a time.</p>',
        unsafe_allow_html=True,
    )

    raw = st.text_area(
        label="Batch input",
        label_visibility="collapsed",
        placeholder="Message 1\nMessage 2\nMessage 3\n...",
        height=200,
        key="batch_input",
    )

    if st.button("▶  classify all →", key="batch_btn"):
        if not raw.strip():
            st.warning("Enter at least one message.")
            return
        if not models_loaded():
            st.error("No trained model found. Run `python -m src.train` first.")
            return

        lines = [l.strip() for l in raw.splitlines() if l.strip()][:200]
        results = []
        progress = st.progress(0, text="Classifying…")
        for i, line in enumerate(lines):
            r = predict_sms(line)
            r["message"] = line
            results.append(r)
            progress.progress((i + 1) / len(lines))
        progress.empty()

        n_spam = sum(1 for r in results if r["label"] == "SPAM")
        _md(f"""
        <div style="display:flex;gap:12px;margin-bottom:14px">
          <div style="background:#161614;border-radius:6px;padding:10px 16px;flex:1">
            <p style="font-size:18px;color:#d4d3cc;font-weight:500;margin:0">{len(results)}</p>
            <p style="font-size:10px;color:#5a5a54;text-transform:uppercase;margin:3px 0 0">total</p>
          </div>
          <div style="background:#161614;border-radius:6px;padding:10px 16px;flex:1">
            <p style="font-size:18px;color:#EF9F27;font-weight:500;margin:0">{n_spam}</p>
            <p style="font-size:10px;color:#5a5a54;text-transform:uppercase;margin:3px 0 0">spam</p>
          </div>
          <div style="background:#161614;border-radius:6px;padding:10px 16px;flex:1">
            <p style="font-size:18px;color:#1D9E75;font-weight:500;margin:0">{len(results)-n_spam}</p>
            <p style="font-size:10px;color:#5a5a54;text-transform:uppercase;margin:3px 0 0">ham</p>
          </div>
        </div>
        """)

        # Results table
        for r in results:
            is_spam  = r["label"] == "SPAM"
            dot_col  = "#EF9F27" if is_spam else "#1D9E75"
            conf_pct = int(r["confidence"] * 100)
            excerpt  = r["message"][:90] + "…" if len(r["message"]) > 90 else r["message"]
            _md(f"""
            <div style="display:flex;align-items:center;gap:10px;padding:8px 12px;
                        background:#161614;border:0.5px solid #2a2a26;border-radius:5px;margin-bottom:5px">
              <span style="width:8px;height:8px;border-radius:50%;background:{dot_col};flex-shrink:0;display:inline-block"></span>
              <span style="font-size:11px;color:#6b6a64;flex:1">{_escape(excerpt)}</span>
              <span style="font-size:11px;color:{dot_col};min-width:45px;text-align:right">{r["label"]}</span>
              <span style="font-size:10px;color:#4a4a44;min-width:32px;text-align:right">{conf_pct}%</span>
            </div>
            """)

        # CSV export
        import io, csv
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["message", "label", "spam_prob", "confidence"])
        for r in results:
            writer.writerow([r["message"], r["label"], r["spam_prob"], r["confidence"]])
        st.download_button(
            label="⬇  Download results CSV",
            data=buf.getvalue().encode(),
            file_name="spam_results.csv",
            mime="text/csv",
            key="dl_csv",
        )


# ─── Tab: Model stats ────────────────────────────────────────────────────────

def _stats_tab():
    import os, json, joblib
    models_dir = os.path.join(os.path.dirname(__file__), "models")

    st.markdown(
        '<p style="font-size:13px;color:#5a5a54;margin:0 0 16px">'
        'Technical overview of the trained model.</p>',
        unsafe_allow_html=True,
    )

    cols = st.columns(2)
    with cols[0]:
        st.markdown(_section_label("architecture"), unsafe_allow_html=True)
        _md("""
        <table style="width:100%;font-size:12px;border-collapse:collapse">
          <tr><td style="color:#5a5a54;padding:5px 0">Model</td>
              <td style="color:#d4d3cc;text-align:right">LinearSVC + CalibratedCV</td></tr>
          <tr><td style="color:#5a5a54;padding:5px 0;border-top:0.5px solid #2a2a26">Text features</td>
              <td style="color:#d4d3cc;text-align:right">TF-IDF word (1-3g) + char (3-5g)</td></tr>
          <tr><td style="color:#5a5a54;padding:5px 0;border-top:0.5px solid #2a2a26">Metadata</td>
              <td style="color:#d4d3cc;text-align:right">7 handcrafted features</td></tr>
          <tr><td style="color:#5a5a54;padding:5px 0;border-top:0.5px solid #2a2a26">Balancing</td>
              <td style="color:#d4d3cc;text-align:right">SMOTE + class_weight=balanced</td></tr>
          <tr><td style="color:#5a5a54;padding:5px 0;border-top:0.5px solid #2a2a26">Training data</td>
              <td style="color:#d4d3cc;text-align:right">138k messages (arxiv:2210.10451)</td></tr>
        </table>
        """)

    with cols[1]:
        st.markdown(_section_label("metadata features"), unsafe_allow_html=True)
        features = [
            ("char_count",        "Normalised message length"),
            ("word_count",        "Normalised word count"),
            ("caps_ratio",        "Fraction of uppercase letters"),
            ("exclamation_count", "Number of ! characters"),
            ("has_url",           "Contains a URL (0/1)"),
            ("has_phone",         "Contains phone number (0/1)"),
            ("has_currency",      "Contains currency symbol (0/1)"),
        ]
        for name, desc in features:
            _md(f"""
            <div style="display:flex;justify-content:space-between;font-size:11px;
                        padding:5px 0;border-top:0.5px solid #2a2a26">
              <span style="color:#EF9F27;font-family:monospace">{name}</span>
              <span style="color:#5a5a54">{desc}</span>
            </div>
            """)

    t_path = os.path.join(models_dir, "threshold.json")
    if os.path.exists(t_path):
        with open(t_path) as f:
            t = json.load(f)["threshold"]
        _md(f"""
        <div style="background:#161614;border:0.5px solid #2a2a26;border-radius:6px;
                    padding:12px 14px;margin-top:16px;font-size:12px">
          <span style="color:#5a5a54">Tuned decision threshold: </span>
          <span style="color:#EF9F27;font-weight:500">{t:.4f}</span>
          <span style="color:#5a5a54;margin-left:12px">
            (optimised for F1-spam on held-out validation set)
          </span>
        </div>
        """)

    # Top spam and ham tokens (requires coefficients)
    coef_path = os.path.join(models_dir, "coefficients.pkl")
    names_path = os.path.join(models_dir, "feature_names.pkl")
    if os.path.exists(coef_path) and os.path.exists(names_path):
        coefs = joblib.load(coef_path)
        names = joblib.load(names_path)

        # Filter to word-level only
        word_mask = [n.startswith("word_tfidf__") for n in names]
        word_names = [n.replace("word_tfidf__", "") for n, m in zip(names, word_mask) if m]
        word_coefs = [c for c, m in zip(coefs, word_mask) if m]

        import numpy as np
        coef_arr = np.array(word_coefs)
        top_spam_idx = coef_arr.argsort()[-20:][::-1]
        top_ham_idx  = coef_arr.argsort()[:20]

        st.markdown("<div style='margin-top:20px'>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(_section_label("top spam signals"), unsafe_allow_html=True)
            for i in top_spam_idx:
                bar_w = min(int(coef_arr[i] / max(coef_arr) * 100), 100)
                _md(f"""
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px">
                  <span style="font-size:11px;color:#d4d3cc;min-width:110px;
                               font-family:monospace">{word_names[i][:18]}</span>
                  <div style="flex:1;height:3px;background:#2a2a26;border-radius:2px">
                    <div style="height:100%;width:{bar_w}%;background:#EF9F27;border-radius:2px"></div>
                  </div>
                  <span style="font-size:10px;color:#EF9F27;min-width:44px;text-align:right">
                    +{coef_arr[i]:.3f}</span>
                </div>
                """)
        with c2:
            st.markdown(_section_label("top ham signals"), unsafe_allow_html=True)
            for i in top_ham_idx:
                bar_w = min(int(abs(coef_arr[i]) / max(abs(coef_arr[top_ham_idx])) * 100), 100)
                _md(f"""
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px">
                  <span style="font-size:11px;color:#d4d3cc;min-width:110px;
                               font-family:monospace">{word_names[i][:18]}</span>
                  <div style="flex:1;height:3px;background:#2a2a26;border-radius:2px">
                    <div style="height:100%;width:{bar_w}%;background:#1D9E75;border-radius:2px"></div>
                  </div>
                  <span style="font-size:10px;color:#1D9E75;min-width:44px;text-align:right">
                    {coef_arr[i]:.3f}</span>
                </div>
                """)
        st.markdown("</div>", unsafe_allow_html=True)


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    _init_state()
    _sidebar()

    tabs = st.tabs(["Analyse", "Batch classifier", "Model stats"])
    with tabs[0]:
        _analyse_tab()
    with tabs[1]:
        _batch_tab()
    with tabs[2]:
        _stats_tab()


if __name__ == "__main__":
    main()