import os
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from legal_core import (
    load_and_chunk,
    build_index,
    build_bm25_index,
    _get_reranker,
    classify_contract,
    extract_contract_info,
    detect_risks,
    expand_queries,
    hybrid_retrieve,
    rerank,
    generate_answer,
)

# ===========================================================================
# PAGE CONFIG
# ===========================================================================
st.set_page_config(
    page_title="Legal Contract Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================================================================
# CUSTOM CSS
# ===========================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Cairo:wght@300;400;600&display=swap');

:root {
    --bg: #0a0a0f;
    --surface: #111118;
    --border: #1e1e2e;
    --accent: #7c6af7;
    --accent2: #c084fc;
    --text: #e2e2f0;
    --muted: #6b6b8a;
    --success: #4ade80;
    --warning: #fbbf24;
    --danger: #f87171;
    --mono: 'IBM Plex Mono', monospace;
    --sans: 'Cairo', sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--sans) !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
.stApp { background: var(--bg); }

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

.main-header {
    text-align: center;
    padding: 1.5rem 0 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.main-title {
    font-family: var(--mono);
    font-size: 2rem;
    font-weight: 600;
    background: linear-gradient(135deg, #7c6af7, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.main-sub {
    color: var(--muted);
    font-size: 0.82rem;
    font-family: var(--mono);
    margin-top: 0.3rem;
}

/* Info cards */
.info-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
}
.info-label {
    color: var(--muted);
    font-size: 0.75rem;
    font-family: var(--mono);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.info-value {
    color: var(--text);
    font-size: 1rem;
    font-weight: 600;
    margin-top: 0.2rem;
}

/* Risk badges */
.risk-high {
    background: rgba(248,113,113,0.12);
    border: 1px solid rgba(248,113,113,0.35);
    border-left: 4px solid #f87171;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    direction: rtl;
}
.risk-medium {
    background: rgba(251,191,36,0.10);
    border: 1px solid rgba(251,191,36,0.3);
    border-left: 4px solid #fbbf24;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    direction: rtl;
}
.risk-low {
    background: rgba(74,222,128,0.08);
    border: 1px solid rgba(74,222,128,0.25);
    border-left: 4px solid #4ade80;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    direction: rtl;
}

/* Chat */
.user-msg {
    background: rgba(124,106,247,0.08);
    border: 1px solid rgba(124,106,247,0.2);
    border-radius: 12px 12px 4px 12px;
    padding: 0.9rem 1.1rem;
    margin: 0.8rem 0 0.4rem;
    text-align: right;
    direction: rtl;
    font-size: 0.95rem;
}
.bot-msg {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px 12px 12px 4px;
    padding: 0.9rem 1.1rem;
    margin: 0.4rem 0 0.8rem;
    direction: rtl;
    font-size: 0.95rem;
    line-height: 1.7;
}
.source-box {
    background: rgba(124,106,247,0.05);
    border: 1px solid rgba(124,106,247,0.15);
    border-radius: 6px;
    padding: 0.6rem 0.9rem;
    margin-top: 0.6rem;
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--muted);
    direction: ltr;
}

/* Buttons */
.stButton button {
    background: linear-gradient(135deg, #7c6af7, #c084fc) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: var(--mono) !important;
    font-weight: 600 !important;
}
.stButton button:hover { opacity: 0.85 !important; }

/* Inputs */
.stTextInput input, .stTextArea textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #7c6af7 !important;
    box-shadow: 0 0 0 2px rgba(124,106,247,0.2) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 8px !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.8rem 1rem;
}
[data-testid="stMetricValue"] {
    font-family: var(--mono) !important;
    color: #7c6af7 !important;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.78rem !important; }

hr { border-color: var(--border) !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# SESSION STATE
# ===========================================================================
def init_state():
    defaults = {
        "indexed": False,
        "chunks": [],
        "contract_info": {},
        "risks": [],
        "chat_history": [],
        "chunk_count": 0,
        "page_count": 0,
        "contract_type": "",
        "show_viz": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ===========================================================================
# SIDEBAR
# ===========================================================================
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 0.5rem; text-align:center;'>
        <div style='font-family:monospace; font-size:1.3rem; font-weight:600;
             background:linear-gradient(135deg,#7c6af7,#c084fc);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            ⚖️ Legal Analyzer
        </div>
        <div style='color:#6b6b8a; font-size:0.72rem; margin-top:0.3rem;'>
            Contract Analysis Pipeline
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    _env_key = os.environ.get("rag_system") or os.environ.get("GROQ_API_KEY", "")
    if _env_key:
        os.environ["GROQ_API_KEY"] = _env_key
    api_key = _env_key

    st.divider()

    st.markdown("**📄 ارفع العقد (PDF)**")
    uploaded_file = st.file_uploader(
        "contract_upload", type=["pdf"], label_visibility="collapsed"
    )

    if uploaded_file:
        if st.button("⚡ تحليل العقد", use_container_width=True):
            with st.spinner("جاري تحليل العقد..."):
                try:
                    import tempfile, os as _os

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name


                    chunks = load_and_chunk(tmp_path)
                    full_text = " ".join([c.page_content for c in chunks])


                    build_index(chunks)
                    build_bm25_index(chunks)
                    _get_reranker()


                    contract_type  = classify_contract(full_text)
                    contract_info  = extract_contract_info(full_text)
                    contract_info["contract_type"] = contract_type
                    risks          = detect_risks(chunks)


                    st.session_state.indexed       = True
                    st.session_state.chunks        = chunks
                    st.session_state.contract_info = contract_info
                    st.session_state.risks         = risks
                    st.session_state.chat_history  = []
                    st.session_state.chunk_count   = len(chunks)
                    st.session_state.contract_type = contract_type

                    _os.unlink(tmp_path)
                    st.session_state.show_viz = True
                    st.success("✅ اكتمل التحليل!")

                except Exception as e:
                    st.error(f"❌ Error: {e}")

    st.divider()

    if st.session_state.indexed:
        st.markdown(f"**📋 {st.session_state.contract_type}**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Chunks", st.session_state.chunk_count)
        with col2:
            high_risks = len([r for r in st.session_state.risks if r["severity"] == "high"])
            st.metric("⚠️ Risks", high_risks)

        st.divider()
        if st.button("🗑️ مسح المحادثة", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


# ===========================================================================
# HEADER
# ===========================================================================
st.markdown("""
<div class='main-header'>
    <div class='main-title'>⚖️ Legal Contract Analyzer</div>
    <div class='main-sub'>Classify · Extract · Detect Risks · Q&A</div>
</div>
""", unsafe_allow_html=True)

if not st.session_state.indexed:
    st.info("📄 ارفع العقد من الـ Sidebar واضغط تحليل العقد")
else:
    st.success(f"✅ جاهز — {st.session_state.contract_type}")

st.markdown("<br>", unsafe_allow_html=True)


# ===========================================================================
# TABS
# ===========================================================================
if st.session_state.indexed:


    risks     = st.session_state.risks
    info      = st.session_state.contract_info
    high_r    = len([r for r in risks if r["severity"] == "high"])
    medium_r  = len([r for r in risks if r["severity"] == "medium"])
    low_r     = len([r for r in risks if r["severity"] == "low"])

    col_v1, col_v2, col_v3 = st.columns(3)

    with col_v1:

        if risks:
            fig = go.Figure(go.Pie(
                labels=["🔴 عالية", "🟡 متوسطة", "🟢 منخفضة"],
                values=[high_r, medium_r, low_r],
                hole=0.6,
                marker=dict(colors=["#f87171", "#fbbf24", "#4ade80"]),
                textinfo="label+value",
                hoverinfo="label+percent",
            ))
            fig.update_layout(
                title=dict(text="توزيع المخاطر", font=dict(color="#e2e2f0", size=14)),
                paper_bgcolor="#111118",
                plot_bgcolor="#111118",
                font=dict(color="#e2e2f0"),
                showlegend=False,
                margin=dict(t=40, b=10, l=10, r=10),
                height=250,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("<div style='text-align:center;padding:3rem;color:#4ade80;font-size:2rem;'>✅ لا مخاطر</div>", unsafe_allow_html=True)

    with col_v2:

        total_risks = high_r * 3 + medium_r * 2 + low_r
        max_score   = max(total_risks, 1)
        risk_pct    = min(int((total_risks / (len(risks) * 3 + 1)) * 100), 100) if risks else 0
        color       = "#f87171" if risk_pct > 60 else "#fbbf24" if risk_pct > 30 else "#4ade80"

        fig2 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_pct,
            title={"text": "مستوى الخطورة", "font": {"color": "#e2e2f0", "size": 14}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#6b6b8a"},
                "bar": {"color": color},
                "bgcolor": "#1e1e2e",
                "steps": [
                    {"range": [0, 30], "color": "rgba(74,222,128,0.15)"},
                    {"range": [30, 60], "color": "rgba(251,191,36,0.15)"},
                    {"range": [60, 100], "color": "rgba(248,113,113,0.15)"},
                ],
                "threshold": {"line": {"color": color, "width": 3}, "value": risk_pct},
            },
            number={"suffix": "%", "font": {"color": "#e2e2f0"}},
        ))
        fig2.update_layout(
            paper_bgcolor="#111118",
            font=dict(color="#e2e2f0"),
            margin=dict(t=40, b=10, l=20, r=20),
            height=250,
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col_v3:

        if risks:
            risk_types = {}
            for r in risks:
                rt = r.get("risk_type", "غير محدد")
                risk_types[rt] = risk_types.get(rt, 0) + 1

            fig3 = go.Figure(go.Bar(
                x=list(risk_types.values()),
                y=list(risk_types.keys()),
                orientation="h",
                marker=dict(
                    color=["#f87171", "#fbbf24", "#7c6af7", "#4ade80", "#60a5fa"][:len(risk_types)],
                    line=dict(width=0)
                ),
                text=list(risk_types.values()),
                textposition="auto",
            ))
            fig3.update_layout(
                title=dict(text="المخاطر حسب النوع", font=dict(color="#e2e2f0", size=14)),
                paper_bgcolor="#111118",
                plot_bgcolor="#111118",
                font=dict(color="#e2e2f0"),
                xaxis=dict(gridcolor="#1e1e2e"),
                yaxis=dict(gridcolor="#1e1e2e"),
                margin=dict(t=40, b=10, l=10, r=10),
                height=250,
            )
            st.plotly_chart(fig3, use_container_width=True)

    st.divider()
    tab1, tab2, tab3 = st.tabs(["📊 تقرير العقد", "⚠️ المخاطر", "💬 اسأل عن العقد"])

    with tab1:
        info = st.session_state.contract_info

        st.markdown("### 📋 معلومات العقد الأساسية")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class='info-card'>
                <div class='info-label'>نوع العقد</div>
                <div class='info-value'>{info.get('contract_type', 'غير محدد')}</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class='info-card'>
                <div class='info-label'>تاريخ البداية</div>
                <div class='info-value'>{info.get('start_date') or 'غير محدد'}</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class='info-card'>
                <div class='info-label'>تاريخ الانتهاء</div>
                <div class='info-value'>{info.get('end_date') or 'غير محدد'}</div>
            </div>""", unsafe_allow_html=True)

        col4, col5 = st.columns(2)
        with col4:
            st.markdown(f"""
            <div class='info-card'>
                <div class='info-label'>قيمة العقد</div>
                <div class='info-value'>{info.get('contract_value') or 'غير محدد'}</div>
            </div>""", unsafe_allow_html=True)
        with col5:
            st.markdown(f"""
            <div class='info-card'>
                <div class='info-label'>القانون المُطبَّق</div>
                <div class='info-value'>{info.get('governing_law') or 'غير محدد'}</div>
            </div>""", unsafe_allow_html=True)

        st.divider()

        st.markdown("### 👥 أطراف العقد")
        parties = info.get("parties", [])
        if parties:
            cols = st.columns(len(parties))
            for i, (col, party) in enumerate(zip(cols, parties)):
                with col:
                    st.markdown(f"""
                    <div class='info-card'>
                        <div class='info-label'>الطرف {i+1}</div>
                        <div class='info-value'>{party}</div>
                    </div>""", unsafe_allow_html=True)
        else:
            st.write("غير محدد")

        st.divider()

        subject = info.get("subject", "")
        if subject:
            st.markdown("### 📝 موضوع العقد")
            st.markdown(f"""
            <div class='info-card'>
                <div class='info-value' style='font-size:0.95rem; line-height:1.6;'>{subject}</div>
            </div>""", unsafe_allow_html=True)

        obligations = info.get("key_obligations", [])
        if obligations:
            st.markdown("### 📌 الالتزامات الرئيسية")
            for ob in obligations:
                st.markdown(f"- {ob}")

        terminations = info.get("termination_conditions", [])
        if terminations:
            st.markdown("### 🚪 شروط الإنهاء")
            for t in terminations:
                st.markdown(f"- {t}")

        payment = info.get("payment_terms")
        if payment:
            st.markdown("### 💰 شروط الدفع")
            st.markdown(f"""
            <div class='info-card'>
                <div class='info-value' style='font-size:0.95rem;'>{payment}</div>
            </div>""", unsafe_allow_html=True)

    with tab2:
        risks = st.session_state.risks

        if not risks:
            st.success("✅ لم يتم اكتشاف بنود خطيرة في هذا العقد")
        else:

            high   = [r for r in risks if r["severity"] == "high"]
            medium = [r for r in risks if r["severity"] == "medium"]
            low    = [r for r in risks if r["severity"] == "low"]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🔴 عالية الخطورة", len(high))
            with col2:
                st.metric("🟡 متوسطة الخطورة", len(medium))
            with col3:
                st.metric("🟢 منخفضة الخطورة", len(low))

            st.divider()

            severity_icons = {"high": "🔴", "medium": "🟡", "low": "🟢"}
            severity_labels = {"high": "عالية", "medium": "متوسطة", "low": "منخفضة"}

            for i, risk in enumerate(risks, 1):
                sev = risk["severity"]
                icon = severity_icons.get(sev, "⚪")
                label = severity_labels.get(sev, sev)
                css_class = f"risk-{sev}"

                st.markdown(f"""
                <div class='{css_class}'>
                    <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;'>
                        <span style='font-size:0.75rem; color:var(--muted); font-family:monospace;'>
                            Chunk {risk.get('chunk_id','?')} | صفحة {risk.get('page','?')}
                        </span>
                        <span style='font-size:0.8rem; font-weight:600;'>
                            {icon} {label} — {risk.get('risk_type','غير محدد')}
                        </span>
                    </div>
                    <div style='font-size:0.88rem; margin-bottom:0.5rem; opacity:0.85;'>
                        {risk['text'][:250]}...
                    </div>
                    <div style='font-size:0.85rem; font-weight:600; margin-bottom:0.3rem;'>
                        ⚠️ {risk.get('explanation','')}
                    </div>
                    <div style='font-size:0.82rem; color:var(--muted);'>
                        💡 {risk.get('recommendation','')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with tab3:

        for turn in st.session_state.chat_history:
            st.markdown(
                f"<div class='user-msg'>🧑 {turn['question']}</div>",
                unsafe_allow_html=True
            )
            sources_html = ""
            if turn.get("sources"):
                items = "".join([
                    f"<div style='padding:0.2rem 0; border-bottom:1px solid #1e1e2e;'>"
                    f"<span style='background:rgba(192,132,252,0.15);color:#c084fc;"
                    f"padding:0.1rem 0.4rem;border-radius:4px;font-size:0.7rem;'>"
                    f"📄 ص {s['page']}</span> "
                    f"<span style='color:#6b6b8a'>{s['text'][:80]}...</span></div>"
                    for s in turn["sources"]
                ])
                sources_html = f"<div class='source-box'><b style='color:#7c6af7'>📚 المصادر:</b>{items}</div>"
            st.markdown(
                f"<div class='bot-msg'>⚖️ {turn['answer']}{sources_html}</div>",
                unsafe_allow_html=True
            )

        with st.form("chat_form", clear_on_submit=True):
            col1, col2 = st.columns([5, 1])
            with col1:
                question = st.text_input(
                    "legal_q", placeholder="اسأل عن أي بند في العقد...",
                    label_visibility="collapsed"
                )
            with col2:
                submitted = st.form_submit_button("إرسال ➤", use_container_width=True)

        if submitted and question.strip():
            with st.spinner("⚖️ جاري تحليل السؤال..."):
                try:

                    expanded = expand_queries(question)

                    all_hits, seen = [], set()
                    for q in expanded:
                        hits = hybrid_retrieve(q, top_k=20)
                        for hit in hits:
                            key = hit["text"][:100]
                            if key not in seen:
                                seen.add(key)
                                all_hits.append(hit)


                    reranked = rerank(question, all_hits, top_k=5)

                    result = generate_answer(
                        question, reranked, st.session_state.chat_history
                    )

                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": result["answer"],
                        "sources": result["sources"],
                    })
                    if len(st.session_state.chat_history) > 5:
                        st.session_state.chat_history.pop(0)

                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Error: {e}")

else:

    st.markdown("""
    <div style='text-align:center; padding:3rem; color:#6b6b8a;'>
        <div style='font-size:3rem;'>⚖️</div>
        <div style='font-size:1rem; margin-top:1rem;'>ارفع العقد من الـ Sidebar للبدء</div>
        <div style='font-size:0.82rem; margin-top:0.5rem; font-family:monospace;'>
            PDF → Classify → Extract → Risks → Q&A
        </div>
    </div>
    """, unsafe_allow_html=True)
