"""SpamDetect v2 — multi-model SMS spam classifier"""
import re
import streamlit as st
from src.predict import predict_sms, predict_all, get_all_metrics, MODEL_INFO
from src.preprocess import extract_signals

st.set_page_config(page_title="SpamDetect", page_icon="🛡", layout="wide",
                   initial_sidebar_state="expanded")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""<style>
html,body,[data-testid="stApp"]{background:#111210!important}
[data-testid="stSidebar"]{background:#0f0f0d!important;border-right:.5px solid #2e2e2a}
[data-testid="stMain"]{background:#111210!important}
*,p,div,span,label{font-family:'SF Mono','Fira Code','Consolas',monospace!important}
.block-container{padding:1.2rem 2rem 2rem!important;max-width:1200px!important}
[data-testid="stVerticalBlock"]>div{gap:.35rem!important}
textarea,.stTextArea textarea{background:#161614!important;border:.5px solid #2e2e2a!important;
  border-radius:6px!important;color:#d4d3cc!important;font-size:13px!important;
  line-height:1.6!important;caret-color:#EF9F27;padding:10px 12px!important}
textarea:focus{border-color:#EF9F27!important;box-shadow:none!important}
.stButton button{background:#1c1c1a!important;border:.5px solid #3e3e38!important;
  color:#d4d3cc!important;border-radius:6px!important;font-size:12px!important;
  width:100%!important;padding:7px 14px!important;transition:border-color .2s,color .2s!important}
.stButton button:hover{border-color:#EF9F27!important;color:#EF9F27!important;background:#1c1c1a!important}
.stTabs [data-baseweb="tab-list"]{background:#1a1a18!important;border-bottom:.5px solid #2e2e2a!important;gap:0!important}
.stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"]{background:transparent!important;
  background-color:transparent!important;color:#5a5a54!important;font-size:11px!important;
  letter-spacing:.04em!important;border:none!important;border-bottom:2px solid transparent!important;
  padding:10px 20px!important;cursor:pointer!important;pointer-events:all!important;
  box-shadow:none!important;outline:none!important}
.stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"][aria-selected="true"]{
  background:transparent!important;background-color:transparent!important;
  color:#EF9F27!important;border-bottom:2px solid #EF9F27!important;box-shadow:none!important}
.stTabs [data-baseweb="tab-list"] button[data-baseweb="tab"]:hover{color:#d4d3cc!important;background:transparent!important}
.stTabs [data-baseweb="tab-list"] button:focus,.stTabs [data-baseweb="tab-list"] button:focus-visible{outline:none!important;box-shadow:none!important}
.stTabs [data-baseweb="tab-panel"]{padding:1rem 0 0!important}
.stSpinner>div{border-top-color:#EF9F27!important}
.stProgress>div>div>div{background:#EF9F27!important}
[data-testid="stMarkdownContainer"] a{color:#EF9F27!important}
.stDownloadButton button{color:#1D9E75!important;border-color:#1D9E75!important}
footer,#MainMenu,[data-testid="stDecoration"]{display:none!important}
[data-testid="stHeader"]{background:transparent!important}
</style>""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def _md(html):
    clean = re.sub(r">\s+<", "><", html.strip())
    clean = re.sub(r"\n\s+", " ", clean)
    st.markdown(clean, unsafe_allow_html=True)

def _e(t):
    return t.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

def _sec(t):
    return f'<p style="font-size:10px;color:#5a5a54;letter-spacing:.08em;text-transform:uppercase;margin:0 0 8px">{t}</p>'

# ── Session state ─────────────────────────────────────────────────────────────
def _init():
    for k, v in [("history", []), ("message_buffer", ""),
                 ("last_result", None), ("active_model", "ensemble"),
                 ("compare_results", None)]:
        if k not in st.session_state:
            st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────────────────────
def _sidebar():
    with st.sidebar:
        _md('<p style="font-size:16px;font-weight:500;color:#d4d3cc;margin:0 0 2px">SpamDetect</p>'
            '<p style="font-size:10px;color:#5a5a54;letter-spacing:.06em;margin:0 0 18px">SMS CLASSIFIER · 4 MODELS · 138K TRAINED</p>')

        h = st.session_state.history
        n = len(h); ns = sum(1 for r in h if r["label"]=="SPAM")
        ac = int(sum(r["confidence"] for r in h)/max(n,1)*100)
        _md(f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:14px">'
            f'<div style="background:#161614;border-radius:6px;padding:10px 12px">'
            f'<p style="font-size:18px;color:#EF9F27;font-weight:500;margin:0;line-height:1">{n}</p>'
            f'<p style="font-size:10px;color:#5a5a54;text-transform:uppercase;letter-spacing:.06em;margin:3px 0 0">analysed</p></div>'
            f'<div style="background:#161614;border-radius:6px;padding:10px 12px">'
            f'<p style="font-size:18px;color:{"#EF9F27" if ns else "#d4d3cc"};font-weight:500;margin:0;line-height:1">{int(ns/max(n,1)*100)}%</p>'
            f'<p style="font-size:10px;color:#5a5a54;text-transform:uppercase;letter-spacing:.06em;margin:3px 0 0">spam rate</p></div>'
            f'<div style="background:#161614;border-radius:6px;padding:10px 12px;grid-column:1/-1">'
            f'<p style="font-size:18px;color:#d4d3cc;font-weight:500;margin:0;line-height:1">{ac}%</p>'
            f'<p style="font-size:10px;color:#5a5a54;text-transform:uppercase;letter-spacing:.06em;margin:3px 0 0">avg confidence</p></div></div>')

        if h:
            _md(_sec("session history"))
            for r in reversed(h[-20:]):
                dc = "#EF9F27" if r["label"]=="SPAM" else "#1D9E75"
                ex = r["message"][:36]+"…" if len(r["message"])>36 else r["message"]
                _md(f'<div style="display:flex;align-items:center;gap:6px;padding:5px 8px;background:#161614;'
                    f'border-radius:4px;border:.5px solid #2a2a26;margin-bottom:5px">'
                    f'<span style="width:6px;height:6px;border-radius:50%;background:{dc};flex-shrink:0;display:inline-block"></span>'
                    f'<span style="font-size:11px;color:#6b6a64;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1" '
                    f'title="{_e(r["message"])}">{_e(ex)}</span>'
                    f'<span style="font-size:10px;color:#4a4a44">{int(r["confidence"]*100)}%</span></div>')
            if st.button("Clear history", key="clear_hist"):
                st.session_state.history = []; st.rerun()

        _md('<div style="margin-top:16px">' + _sec("active model") +
            f'<p style="font-size:12px;color:#EF9F27;margin:0">'
            f'{MODEL_INFO[st.session_state.active_model]["label"]}</p>'
            f'<p style="font-size:11px;color:#5a5a54;margin:2px 0 0">'
            f'{MODEL_INFO[st.session_state.active_model]["tagline"]}</p></div>')

# ── Model selector pills ──────────────────────────────────────────────────────
def _model_selector():
    metrics = get_all_metrics()
    cols = st.columns(4, gap="small")
    model_ids = list(MODEL_INFO.keys())
    for i, mid in enumerate(model_ids):
        info = MODEL_INFO[mid]
        m    = metrics.get(mid, {})
        acc  = int(m.get("accuracy",0)*100)
        rec  = int(m.get("recall",0)*100)
        active = st.session_state.active_model == mid
        border = "#EF9F27" if active else "#2e2e2a"
        bg     = "#1c1200" if active else "#161614"
        tc     = "#EF9F27" if active else "#d4d3cc"
        rec_c  = "#1D9E75" if mid in ("naive_bayes","ensemble") else "#BA7517"
        badge  = "★ recommended" if mid == "ensemble" else ""
        with cols[i]:
            _md(f'<div style="background:{bg};border:.5px solid {border};border-radius:8px;'
                f'padding:10px 12px;cursor:pointer;min-height:80px">'
                f'{"<p style=\"font-size:9px;color:#EF9F27;margin:0 0 4px;letter-spacing:.06em\">"+badge+"</p>" if badge else "<div style=height:13px></div>"}'
                f'<p style="font-size:12px;font-weight:500;color:{tc};margin:0 0 2px">{info["label"]}</p>'
                f'<p style="font-size:10px;color:#5a5a54;margin:0 0 8px">{info["tagline"]}</p>'
                f'<span style="font-size:10px;color:#d4d3cc">acc {acc}%</span>'
                f'<span style="font-size:10px;color:{rec_c};margin-left:8px">rec {rec}%</span>'
                f'</div>')
            if st.button("select", key=f"sel_{mid}"):
                st.session_state.active_model = mid
                st.session_state.last_result   = None
                st.session_state.compare_results = None
                st.rerun()

# ── Confidence arc gauge ──────────────────────────────────────────────────────
def _arc_gauge(spam_prob: float, is_spam: bool):
    pct   = int(spam_prob * 100)
    color = "#EF9F27" if is_spam else "#1D9E75"
    # Arc: half-circle, radius 54, stroke 10
    # Full half-circle arc length ≈ π×54 ≈ 169.6; fill = pct/100 * 169.6
    fill  = spam_prob * 169.6
    _md(f'<div style="display:flex;flex-direction:column;align-items:center;padding:10px 0 4px">'
        f'<svg width="130" height="72" viewBox="0 0 130 72">'
        f'<path d="M 11 65 A 54 54 0 0 1 119 65" fill="none" stroke="#2a2a26" stroke-width="10" stroke-linecap="round"/>'
        f'<path d="M 11 65 A 54 54 0 0 1 119 65" fill="none" stroke="{color}" stroke-width="10" stroke-linecap="round"'
        f' stroke-dasharray="{fill:.1f} 169.6" style="transition:stroke-dasharray .6s ease"/>'
        f'<text x="65" y="60" text-anchor="middle" font-size="22" font-weight="500" fill="{color}"'
        f' font-family="SF Mono,Fira Code,Consolas,monospace">{pct}%</text>'
        f'</svg>'
        f'<p style="font-size:10px;color:#5a5a54;letter-spacing:.06em;text-transform:uppercase;margin:0">spam probability</p>'
        f'</div>')

# ── Highlighted message ───────────────────────────────────────────────────────
def _highlight(message, anatomy):
    triggers = sorted({a["token"] for a in anatomy if not a["token"].startswith("[")}, key=len, reverse=True)
    html = _e(message)
    for tok in triggers:
        if len(tok) < 2: continue
        html = re.compile(re.escape(tok), re.I).sub(
            lambda m: f'<mark style="background:#2a1a05;color:#EF9F27;border-radius:2px;padding:0 2px">{_e(m.group())}</mark>',
            html)
    return f'<span style="color:#d4d3cc;font-size:13px;line-height:1.8">{html}</span>'

# ── Signal chips ──────────────────────────────────────────────────────────────
def _chips(sigs):
    defs = [("has_url","URL","#C84A1A","#221205"),("has_phone","Phone","#BA7517","#231800"),
            ("has_currency","Currency","#BA7517","#231800"),("has_email","Email","#BA7517","#231800")]
    parts = []
    for key, lbl, c, bg in defs:
        on = sigs.get(key, False)
        parts.append(f'<span style="background:{"" if not on else bg};border:.5px solid {"#2e2e2a" if not on else c};'
                     f'border-radius:4px;padding:4px 10px;font-size:11px;color:{"#5a5a54" if not on else c};'
                     f'{"opacity:.4;" if not on else ""}display:inline-flex;align-items:center;gap:5px">'
                     f'<span style="width:5px;height:5px;border-radius:50%;background:currentColor;display:inline-block"></span>'
                     f'{lbl}</span>')
    if sigs.get("caps_ratio", 0) > 0.3:
        parts.append(f'<span style="background:#221205;border:.5px solid #3a1e0a;border-radius:4px;padding:4px 10px;'
                     f'font-size:11px;color:#C84A1A;display:inline-flex;align-items:center;gap:5px">'
                     f'<span style="width:5px;height:5px;border-radius:50%;background:currentColor;display:inline-block"></span>'
                     f'Caps {int(sigs["caps_ratio"]*100)}%</span>')
    return " ".join(parts)

# ── Verdict panel ─────────────────────────────────────────────────────────────
def _verdict(result):
    label   = result["label"]
    sp      = int(result["spam_prob"]*100)
    anatomy = result["anatomy"]
    is_spam = label == "SPAM"
    vc = "#EF9F27" if is_spam else "#1D9E75"
    vb = "#221205" if is_spam else "#0d1e16"
    vr = "#3a1e0a" if is_spam else "#0f3020"
    ic = "⚠" if is_spam else "✓"

    # Header + arc side by side
    col_v, col_g = st.columns([3, 1], gap="small")
    with col_v:
        _md(f'<div style="background:{vb};border:.5px solid {vr};border-radius:6px 6px 0 0;'
            f'padding:10px 14px;display:flex;align-items:center;justify-content:space-between;margin-top:12px">'
            f'<span style="font-size:10px;color:#6b4020;letter-spacing:.08em;text-transform:uppercase">classification</span>'
            f'<span style="font-size:16px;color:{vc};font-weight:500;letter-spacing:.05em">{ic} {label}</span></div>')
        _md(f'<div style="background:#161614;border-left:.5px solid #2e2e2a;border-right:.5px solid #2e2e2a;'
            f'padding:10px 14px;display:flex;align-items:center;gap:10px">'
            f'<span style="font-size:10px;color:#5a5a54;text-transform:uppercase;letter-spacing:.06em;white-space:nowrap">Model</span>'
            f'<span style="font-size:11px;color:#d4d3cc">{MODEL_INFO[result["model_id"]]["label"]}</span>'
            f'<span style="font-size:10px;color:#5a5a54;margin-left:8px">threshold {result["threshold"]:.2f}</span></div>')
    with col_g:
        _arc_gauge(result["spam_prob"], is_spam)

    if anatomy:
        tags = "".join(
            f'<span style="background:{"#2a1a05" if a["weight"]>.05 else "#221205"};'
            f'border:.5px solid {"#EF9F27" if a["weight"]>.05 else "#3a1e0a"};'
            f'border-radius:4px;padding:3px 8px;font-size:11px;'
            f'color:{"#EF9F27" if a["weight"]>.05 else "#C84A1A"};'
            f'display:inline-flex;align-items:center;gap:5px">'
            f'{_e(a["token"])}<span style="font-size:10px;color:#6b4020">+{a["weight"]:.3f}</span></span>'
            for a in anatomy[:8]
        )
        _md(f'<div style="background:#161614;border:.5px solid #2e2e2a;border-top:none;'
            f'border-radius:0 0 6px 6px;padding:10px 14px">'
            f'<p style="font-size:10px;color:#5a5a54;letter-spacing:.08em;text-transform:uppercase;margin:0 0 8px">spam anatomy — top triggers</p>'
            f'<div style="display:flex;flex-wrap:wrap;gap:6px">{tags}</div></div>')
    else:
        _md('<div style="background:#161614;border-left:.5px solid #2e2e2a;border-right:.5px solid #2e2e2a;'
            'border-bottom:.5px solid #2e2e2a;border-radius:0 0 6px 6px;height:4px"></div>')

# ── Compare-all panel ─────────────────────────────────────────────────────────
def _compare_panel(results: dict):
    _md(_sec("all models — comparison"))
    rows = ""
    for mid, r in results.items():
        dc  = "#EF9F27" if r["label"]=="SPAM" else "#1D9E75"
        sp  = int(r["spam_prob"]*100)
        bar = f'<div style="flex:1;height:3px;background:#2a2a26;border-radius:2px;overflow:hidden"><div style="height:100%;width:{sp}%;background:{dc}"></div></div>'
        rows += (f'<div style="display:flex;align-items:center;gap:10px;padding:8px 14px;'
                 f'border-bottom:.5px solid #2a2a26">'
                 f'<span style="font-size:11px;color:#d4d3cc;min-width:110px">{MODEL_INFO[mid]["label"]}</span>'
                 f'{bar}'
                 f'<span style="font-size:11px;color:{dc};min-width:42px;text-align:right">{r["label"]}</span>'
                 f'<span style="font-size:11px;color:#5a5a54;min-width:36px;text-align:right">{sp}%</span>'
                 f'</div>')
    _md(f'<div style="background:#161614;border:.5px solid #2e2e2a;border-radius:6px;overflow:hidden">{rows}</div>')

# ── Examples ──────────────────────────────────────────────────────────────────
EXAMPLES = [
    ("Prize scam",    "Congratulations! You've WON a FREE iPhone 15. Click http://bit.ly/cl4im to claim NOW before midnight!"),
    ("Bank phishing", "ALERT: Your SBI account suspended. Verify KYC at http://sbi-verify.co/update or lose access in 24hrs."),
    ("OTP fraud",     "Your OTP for bank transaction is 847291. NEVER share. If not you, call 1800-XXXX immediately."),
    ("Ham – friend",  "Hey! Free for lunch today? New place on MG Road I've been wanting to try."),
    ("Ham – delivery","Your Amazon package #405-1234 has been delivered. Contact support if you didn't receive it."),
]

# ── Analyse tab ───────────────────────────────────────────────────────────────
def _analyse_tab():
    _md('<p style="font-size:19px;font-weight:500;color:#d4d3cc;margin:0 0 2px">🛡 SpamDetect</p>'
        '<p style="font-size:12px;color:#5a5a54;margin:0 0 14px">Select a model, paste an SMS, classify — or compare all 4 at once.</p>')

    _md(_sec("choose model"))
    _model_selector()
    _md('<hr style="border:none;border-top:.5px solid #2a2a26;margin:14px 0 12px">')

    col_main, col_right = st.columns([5, 2], gap="large")

    with col_right:
        _md(_sec("try an example"))
        for lbl, ex in EXAMPLES:
            spam_eg = any(k in lbl.lower() for k in ["scam","phishing","fraud","otp"])
            dot = "🟠" if spam_eg else "🟢"
            if st.button(f"{dot} {lbl}", key=f"eg_{lbl}"):
                st.session_state.message_buffer  = ex
                st.session_state.last_result     = None
                st.session_state.compare_results = None
                st.rerun()
        _md('<div style="margin-top:18px;padding-top:14px;border-top:.5px solid #2a2a26">'
            + _sec("signal legend") +
            '<div style="font-size:11px;color:#5a5a54;line-height:2.2">'
            '<span style="color:#C84A1A">■</span>&nbsp; URL detected<br>'
            '<span style="color:#BA7517">■</span>&nbsp; Phone / currency<br>'
            '<span style="color:#EF9F27">■</span>&nbsp; Spam trigger word<br>'
            '<span style="color:#1D9E75">■</span>&nbsp; Ham (safe)</div></div>')

    with col_main:
        message = st.text_area("msg", label_visibility="collapsed",
                               placeholder="Paste or type an SMS message here…",
                               value=st.session_state.message_buffer, height=110)
        st.session_state.message_buffer = message

        if message.strip():
            sigs = extract_signals(message)
            _md(f'<div style="display:flex;align-items:center;flex-wrap:wrap;gap:6px;margin:6px 0 10px">'
                f'{_chips(sigs)}'
                f'<span style="font-size:11px;color:#4a4a44;margin-left:2px">'
                f'{sigs["char_count"]} chars · {sigs["word_count"]} words</span></div>')
        else:
            _md("<div style='margin-bottom:12px'></div>")

        btn_col1, btn_col2 = st.columns([3, 2], gap="small")
        with btn_col1:
            classify_btn = st.button("▶  analyse message →", key="classify_btn")
        with btn_col2:
            compare_btn  = st.button("⊞  compare all models", key="compare_btn")

        if (classify_btn or compare_btn) and not message.strip():
            _md('<div style="background:#1e1a0e;border:.5px solid #3a3010;border-radius:6px;'
                'padding:10px 14px;font-size:12px;color:#9a8a40;margin-top:8px">⚠ Please enter a message first.</div>')
            return

        if classify_btn and message.strip():
            if not (models_loaded := _check_models()): return
            with st.spinner(""):
                result = predict_sms(message, st.session_state.active_model)
                result["message"] = message
                st.session_state.history.append(result)
                st.session_state.last_result     = result
                st.session_state.compare_results = None

        if compare_btn and message.strip():
            if not (models_loaded := _check_models()): return
            with st.spinner("Running all 4 models…"):
                all_r = predict_all(message)
                for mid, r in all_r.items(): r["message"] = message
                best  = max(all_r.values(), key=lambda r: r["spam_prob"])
                best["message"] = message
                st.session_state.history.append(best)
                st.session_state.compare_results = all_r
                st.session_state.last_result     = None

        # Results
        if st.session_state.compare_results:
            _md('<hr style="border:none;border-top:.5px solid #2a2a26;margin:12px 0 10px">')
            _compare_panel(st.session_state.compare_results)
        elif st.session_state.last_result:
            r = st.session_state.last_result
            _md('<hr style="border:none;border-top:.5px solid #2a2a26;margin:12px 0 8px">')
            _md(_sec("message preview — highlighted triggers"))
            _md(f'<div style="background:#161614;border:.5px solid #2a2a26;border-radius:6px;'
                f'padding:12px 14px;line-height:1.8">{_highlight(r["message"], r["anatomy"])}</div>')
            _verdict(r)

def _check_models():
    from src.predict import models_loaded
    if not models_loaded():
        st.error("No models found. Run: python -m src.train_all")
        return False
    return True

# ── Batch tab ─────────────────────────────────────────────────────────────────
def _batch_tab():
    _md('<p style="font-size:13px;color:#5a5a54;margin:0 0 10px">One message per line · max 200 · uses active model</p>')
    raw = st.text_area("batch", label_visibility="collapsed",
                       placeholder="Message 1\nMessage 2\n…", height=180, key="batch_input")
    if st.button("▶  classify all →", key="batch_btn"):
        if not raw.strip(): st.warning("Enter at least one message."); return
        lines = [l.strip() for l in raw.splitlines() if l.strip()][:200]
        results = []
        prog = st.progress(0, text="Classifying…")
        for i, ln in enumerate(lines):
            r = predict_sms(ln, st.session_state.active_model); r["message"] = ln
            results.append(r); prog.progress((i+1)/len(lines))
        prog.empty()
        ns = sum(1 for r in results if r["label"]=="SPAM")
        _md(f'<div style="display:flex;gap:12px;margin-bottom:14px">'
            f'<div style="background:#161614;border-radius:6px;padding:10px 16px;flex:1">'
            f'<p style="font-size:18px;color:#d4d3cc;font-weight:500;margin:0">{len(results)}</p>'
            f'<p style="font-size:10px;color:#5a5a54;text-transform:uppercase;margin:3px 0 0">total</p></div>'
            f'<div style="background:#161614;border-radius:6px;padding:10px 16px;flex:1">'
            f'<p style="font-size:18px;color:#EF9F27;font-weight:500;margin:0">{ns}</p>'
            f'<p style="font-size:10px;color:#5a5a54;text-transform:uppercase;margin:3px 0 0">spam</p></div>'
            f'<div style="background:#161614;border-radius:6px;padding:10px 16px;flex:1">'
            f'<p style="font-size:18px;color:#1D9E75;font-weight:500;margin:0">{len(results)-ns}</p>'
            f'<p style="font-size:10px;color:#5a5a54;text-transform:uppercase;margin:3px 0 0">ham</p></div></div>')
        for r in results:
            dc = "#EF9F27" if r["label"]=="SPAM" else "#1D9E75"
            ex = r["message"][:90]+"…" if len(r["message"])>90 else r["message"]
            _md(f'<div style="display:flex;align-items:center;gap:10px;padding:8px 12px;background:#161614;'
                f'border:.5px solid #2a2a26;border-radius:5px;margin-bottom:5px">'
                f'<span style="width:8px;height:8px;border-radius:50%;background:{dc};flex-shrink:0;display:inline-block"></span>'
                f'<span style="font-size:11px;color:#6b6a64;flex:1">{_e(ex)}</span>'
                f'<span style="font-size:11px;color:{dc};min-width:45px;text-align:right">{r["label"]}</span>'
                f'<span style="font-size:10px;color:#4a4a44;min-width:32px;text-align:right">{int(r["confidence"]*100)}%</span></div>')
        import io, csv
        buf = io.StringIO(); w = csv.writer(buf)
        w.writerow(["message","label","spam_prob","confidence","model"])
        for r in results: w.writerow([r["message"],r["label"],r["spam_prob"],r["confidence"],r["model_id"]])
        st.download_button("⬇  Download CSV", buf.getvalue().encode(),
                           "spam_results.csv", "text/csv", key="dl_csv")

# ── Model stats tab ───────────────────────────────────────────────────────────
def _stats_tab():
    metrics = get_all_metrics()
    if not metrics:
        st.info("Train models first: python -m src.train_all"); return

    model_names  = [MODEL_INFO[m]["label"] for m in MODEL_INFO]
    model_ids    = list(MODEL_INFO.keys())
    acc_vals  = [round(metrics.get(m,{}).get("accuracy",0)*100, 1) for m in model_ids]
    f1_vals   = [round(metrics.get(m,{}).get("f1",0)*100, 1)       for m in model_ids]
    prec_vals = [round(metrics.get(m,{}).get("precision",0)*100, 1) for m in model_ids]
    rec_vals  = [round(metrics.get(m,{}).get("recall",0)*100, 1)    for m in model_ids]

    st.markdown("""
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
""", unsafe_allow_html=True)

    chart_html = f"""
<div style="position:relative;width:100%;height:280px;background:#161614;border:.5px solid #2a2a26;border-radius:6px;padding:14px;box-sizing:border-box">
<canvas id="mchart" role="img" aria-label="Model comparison bar chart showing accuracy, F1, precision, recall for 4 models">Model comparison data.</canvas>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
(function(){{
const ctx = document.getElementById('mchart');
new Chart(ctx, {{
  type: 'bar',
  data: {{
    labels: {model_names},
    datasets: [
      {{label:'Accuracy', data:{acc_vals}, backgroundColor:'#EF9F27', borderRadius:3}},
      {{label:'F1 Spam',  data:{f1_vals},  backgroundColor:'#C84A1A', borderRadius:3}},
      {{label:'Precision',data:{prec_vals},backgroundColor:'#378ADD', borderRadius:3}},
      {{label:'Recall',   data:{rec_vals}, backgroundColor:'#1D9E75', borderRadius:3}},
    ]
  }},
  options: {{
    responsive:true, maintainAspectRatio:false,
    plugins:{{legend:{{display:false}}, tooltip:{{callbacks:{{label:c=>c.dataset.label+': '+c.raw+'%'}}}}}},
    scales:{{
      x:{{ticks:{{color:'#6b6a64',font:{{size:11}}}}, grid:{{color:'#2a2a26'}}}},
      y:{{min:60,max:100,ticks:{{color:'#6b6a64',font:{{size:11}},callback:v=>v+'%'}},grid:{{color:'#2a2a26'}}}}
    }}
  }}
}});
}})();
</script>
<div style="display:flex;flex-wrap:wrap;gap:14px;margin-top:10px;font-size:11px;color:#6b6a64">
  <span><span style="display:inline-block;width:10px;height:10px;background:#EF9F27;border-radius:2px;margin-right:4px"></span>Accuracy</span>
  <span><span style="display:inline-block;width:10px;height:10px;background:#C84A1A;border-radius:2px;margin-right:4px"></span>F1 spam</span>
  <span><span style="display:inline-block;width:10px;height:10px;background:#378ADD;border-radius:2px;margin-right:4px"></span>Precision</span>
  <span><span style="display:inline-block;width:10px;height:10px;background:#1D9E75;border-radius:2px;margin-right:4px"></span>Recall</span>
</div>
"""
    st.markdown(chart_html, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:20px'>", unsafe_allow_html=True)
    _md(_sec("per-model details"))
    for mid in model_ids:
        m    = metrics.get(mid, {})
        info = MODEL_INFO[mid]
        active = st.session_state.active_model == mid
        ac = "#EF9F27" if active else "#2e2e2a"
        _md(f'<div style="background:#161614;border:.5px solid {ac};border-radius:6px;padding:12px 14px;margin-bottom:8px">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">'
            f'<p style="font-size:13px;font-weight:500;color:{"#EF9F27" if active else "#d4d3cc"};margin:0">{info["label"]}</p>'
            f'<p style="font-size:11px;color:#5a5a54;margin:0">{info["tagline"]}</p></div>'
            f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px">'
            + "".join(
                f'<div style="background:#1c1c1a;border-radius:4px;padding:6px 10px">'
                f'<p style="font-size:16px;color:#d4d3cc;font-weight:500;margin:0">{int(m.get(k,0)*100)}%</p>'
                f'<p style="font-size:10px;color:#5a5a54;margin:2px 0 0">{k}</p></div>'
                for k in ("accuracy","f1","precision","recall")
            ) + '</div></div>')
    st.markdown("</div>", unsafe_allow_html=True)

# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    _init()
    _sidebar()
    tabs = st.tabs(["Analyse", "Batch classifier", "Model stats"])
    with tabs[0]: _analyse_tab()
    with tabs[1]: _batch_tab()
    with tabs[2]: _stats_tab()

if __name__ == "__main__":
    main()