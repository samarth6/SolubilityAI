import streamlit as st

CSS = """
<style>

/* Metric cards */
.metric-card {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-left: 4px solid #339af0;
    border-radius: 8px;
    padding: 14px 16px;
    text-align: center;
    margin-bottom: 8px;
}
.metric-card .val {
    font-size: 1.5rem;
    font-weight: 700;
    color: #1864ab;
    display: block;
    line-height: 1.2;
}
.metric-card .lbl {
    font-size: 0.68rem;
    color: #868e96;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-top: 5px;
    display: block;
}

/* Info box  */
.info-box {
    background: #f1f3f5;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 18px 22px;
    margin: 12px 0;
    font-size: 0.9rem;
    line-height: 2;
    color: #212529;
}
.info-box b, .info-box strong {
    color: #1864ab;
}

/* About cards */
.about-card {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 12px;
    color: #343a40;
    font-size: 0.88rem;
    line-height: 1.7;
}
.about-card h4 {
    color: #1864ab;
    margin: 0 0 8px 0;
    font-size: 0.95rem;
}

/* Lipinski rule rows*/
.rule-pass {
    background: #d3f9d8;
    border-left: 4px solid #2f9e44;
    border-radius: 6px;
    padding: 8px 14px;
    margin-bottom: 6px;
    font-size: 0.88rem;
    color: #1a1a1a;
}
.rule-fail {
    background: #ffe3e3;
    border-left: 4px solid #c92a2a;
    border-radius: 6px;
    padding: 8px 14px;
    margin-bottom: 6px;
    font-size: 0.88rem;
    color: #1a1a1a;
}

/*  History table  */
.history-row {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 6px;
    padding: 8px 14px;
    margin-bottom: 5px;
    font-size: 0.85rem;
    color: #212529;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

/* Fuzzy rule cards */
.rule-card {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-left: 3px solid #339af0;
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.83rem;
    color: #212529;
}
.rule-card .cond { color: #495057; margin-bottom: 3px; }
.rule-card .res  { font-weight: 600; }

</style>
"""

def inject_styles():
    st.markdown(CSS, unsafe_allow_html=True)
