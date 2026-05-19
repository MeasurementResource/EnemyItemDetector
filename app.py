import streamlit as st
import pandas as pd
import anthropic
import json
import re
import io
import math
import time
import os
from itertools import combinations

# ─── API Key (shared key — loaded from Streamlit secrets or env var) ──────────

def get_api_key() -> str:
    """Load API key from Streamlit secrets (cloud) or environment variable (local)."""
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except (KeyError, FileNotFoundError):
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        return key

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Item Relationship Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .main-header {
        padding: 1.5rem 0 1rem;
        border-bottom: 2px solid #111827;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin: 0;
        color: #111827;
    }
    .main-header p {
        color: #6B7280;
        font-size: 0.9rem;
        margin: 0.4rem 0 0;
    }
    .eyebrow {
        font-size: 0.7rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #9CA3AF;
        margin-bottom: 0.25rem;
    }

    .metric-card {
        background: #F9FAFB;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        border-top: 3px solid #6366F1;
    }
    .metric-card.clone  { border-top-color: #EF4444; }
    .metric-card.variant { border-top-color: #8B5CF6; }
    .metric-card.enemy  { border-top-color: #F59E0B; }
    .metric-card.reviewed { border-top-color: #10B981; }
    .metric-card .val { font-size: 1.8rem; font-weight: 700; margin: 0; }
    .metric-card .lbl { font-size: 0.75rem; color: #6B7280; margin: 0; }

    .result-card {
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        background: #fff;
    }
    .result-card.clone  { border-left: 4px solid #EF4444; background: #FEF2F2; }
    .result-card.variant { border-left: 4px solid #8B5CF6; background: #F5F3FF; }
    .result-card.enemy  { border-left: 4px solid #F59E0B; background: #FFFBEB; }

    .badge {
        display: inline-block;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        padding: 2px 10px;
        border-radius: 20px;
        margin-right: 6px;
    }
    .badge.clone   { background: #FEE2E2; color: #B91C1C; }
    .badge.variant { background: #EDE9FE; color: #7C3AED; }
    .badge.enemy   { background: #FEF3C7; color: #B45309; }

    .sim-bar-wrap { display: flex; align-items: center; gap: 8px; margin: 4px 0; }
    .sim-bar-bg   { flex: 1; height: 6px; background: #F3F4F6; border-radius: 3px; overflow: hidden; }
    .sim-bar-fill { height: 100%; border-radius: 3px; }
    .sim-score    { font-size: 0.75rem; font-weight: 700; min-width: 36px; text-align: right; }

    .item-box {
        background: #fff;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 0.85rem;
        line-height: 1.6;
    }
    .item-box .item-id { font-size: 0.7rem; color: #9CA3AF; font-weight: 700; text-transform: uppercase; margin-bottom: 4px; }
    .item-box .item-stem { color: #111827; margin-bottom: 6px; }
    .item-box .item-opt  { color: #6B7280; }
    .item-box .item-opt.correct { color: #059669; font-weight: 600; }

    .research-note {
        font-size: 0.72rem;
        color: #9CA3AF;
        border-top: 1px solid #F3F4F6;
        padding-top: 1rem;
        margin-top: 1.5rem;
        line-height: 1.7;
    }

    .stProgress > div > div { background: linear-gradient(90deg, #6366F1, #B91C1C) !important; }
    div[data-testid="stExpander"] { border: 1px solid #E5E7EB !important; border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────

RELATIONSHIP_META = {
    "clone": {
        "label": "Clone",
        "desc": "Virtually identical — same assessment point, trivially reworded",
        "color": "#B91C1C",
        "dot": "#EF4444",
        "priority": 1,
    },
    "variant": {
        "label": "Variant",
        "desc": "Same learning objective, meaningfully different wording",
        "color": "#7C3AED",
        "dot": "#8B5CF6",
        "priority": 2,
    },
    "enemy": {
        "label": "Enemy",
        "desc": "One item reveals or clues the answer to the other",
        "color": "#B45309",
        "dot": "#F59E0B",
        "priority": 3,
    },
}

CHUNK_SIZE = 6

# ─── Session State Init ───────────────────────────────────────────────────────

def init_state():
    defaults = {
        "stage": "upload",       # upload | map | preview | results
        "items": [],
        "results": [],
        "mapping": {},
        "raw_df": None,
        "filename": None,
        "review_decisions": {},  # key: "i1-i2", value: decision string
        "filter_type": "all",
        "sort_by": "score",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─── Helper Functions ─────────────────────────────────────────────────────────

def item_text(item: dict, include_key: bool = True) -> str:
    parts = [f"Stem: {item.get('stem', '(no stem)')}"]
    opts = item.get("options", [])
    if opts:
        parts.append("Options:")
        for i, o in enumerate(opts):
            letter = chr(65 + i)
            correct = item.get("correct_answer", "")
            marker = " ✓" if correct.upper() == letter else ""
            parts.append(f"  {letter}. {o}{marker}")
    if include_key and item.get("correct_answer"):
        parts.append(f"Correct answer: {item['correct_answer']}")
    return "\n".join(parts)


def sim_color(score: float) -> str:
    if score >= 0.8: return "#EF4444"
    if score >= 0.6: return "#F59E0B"
    if score >= 0.4: return "#8B5CF6"
    return "#6B7280"


def sim_bar_html(score: float) -> str:
    pct = int(score * 100)
    color = sim_color(score)
    return f"""
    <div class="sim-bar-wrap">
        <div class="sim-bar-bg">
            <div class="sim-bar-fill" style="width:{pct}%; background:{color};"></div>
        </div>
        <span class="sim-score" style="color:{color};">{pct}%</span>
    </div>"""


def analyze_chunk(items: list, pair_indices: list, client: anthropic.Anthropic) -> list:
    pairs_text = "\n\n".join([
        f"=== Pair [{i+1}, {j+1}] ===\nITEM {i+1}:\n{item_text(items[i])}\n\nITEM {j+1}:\n{item_text(items[j])}"
        for i, j in pair_indices
    ])

    prompt = f"""You are a psychometrician specializing in enemy item detection for high-stakes competency-based assessments.

Analyze each pair of test items and classify their relationship.

CLASSIFICATIONS:
- "clone": Items virtually identical — same stem, options, trivially reworded. Never place on same form.
- "variant": Same learning objective, meaningfully different wording or distractors. Likely should not appear together.
- "enemy": One item gives away, implies, or clues the answer to the other. Should not appear on same form.
- "none": No meaningful psychometric relationship. Safe to administer together.

ENEMY SUBTYPE (for clone/variant/enemy):
- "too_similar": Substantial text or option overlap (lexical)
- "giveaway": One item's content reveals correct answer to other
- "same_topic": Same learning objective, different surface text (semantic)

SIMILARITY SCORE: 0.0–1.0 (semantic + content similarity estimate):
- 0.9–1.0: Nearly identical
- 0.7–0.89: Highly similar
- 0.5–0.69: Moderate overlap
- 0.3–0.49: Some relationship
- 0.0–0.29: Minimal

Respond ONLY with a valid JSON array. No markdown, no explanation outside JSON.

[
  {{
    "item1": 1,
    "item2": 2,
    "relationship": "clone",
    "subtype": "too_similar",
    "similarity_score": 0.92,
    "confidence": "high",
    "stem_overlap": "high",
    "answer_overlap": "high",
    "reason": "One-sentence psychometric justification."
  }}
]

Pairs to analyze:
{pairs_text}"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()
    raw = re.sub(r"```json|```", "", raw).strip()
    return json.loads(raw)


def run_full_analysis(items: list, client: anthropic.Anthropic):
    pairs = list(combinations(range(len(items)), 2))
    total = len(pairs)
    chunks = [pairs[i:i+CHUNK_SIZE] for i in range(0, total, CHUNK_SIZE)]

    results = []
    progress_bar = st.progress(0, text="Starting analysis…")
    status = st.empty()
    done = 0

    for chunk in chunks:
        try:
            chunk_results = analyze_chunk(items, chunk, client)
            flagged = [r for r in chunk_results if r.get("relationship", "none") != "none"]
            results.extend(flagged)
        except Exception as e:
            st.error(f"Analysis error on chunk: {e}")
            break
        done += len(chunk)
        pct = done / total
        progress_bar.progress(pct, text=f"Evaluated {done:,} / {total:,} pairs…")
        status.caption(f"🔬 {len(results)} relationships flagged so far")

    progress_bar.empty()
    status.empty()

    # Sort by similarity score descending (highest priority first)
    results.sort(key=lambda r: r.get("similarity_score", 0), reverse=True)
    return results


def parse_uploaded_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file, dtype=str).fillna("")
    else:
        return pd.read_excel(uploaded_file, dtype=str).fillna("")


def apply_mapping(df: pd.DataFrame, mapping: dict) -> list:
    items = []
    for _, row in df.iterrows():
        stem_col = mapping.get("stem")
        if not stem_col or not row.get(stem_col, "").strip():
            continue
        opts = []
        for k in ["optA", "optB", "optC", "optD"]:
            col = mapping.get(k)
            if col and row.get(col, "").strip():
                opts.append(row[col].strip())
        item = {
            "id": row.get(mapping.get("id", ""), str(len(items) + 1)).strip() or str(len(items) + 1),
            "stem": row[stem_col].strip(),
            "options": opts,
            "correct_answer": row.get(mapping.get("key", ""), "").strip() or None,
            "content_area": row.get(mapping.get("content", ""), "").strip() or None,
        }
        items.append(item)
    return items


def results_to_df(results: list, items: list, review_decisions: dict) -> pd.DataFrame:
    rows = []
    for r in results:
        i1 = items[r["item1"] - 1] if r["item1"] - 1 < len(items) else {}
        i2 = items[r["item2"] - 1] if r["item2"] - 1 < len(items) else {}
        key = f"{r['item1']}-{r['item2']}"
        rows.append({
            "Item 1 Index": r["item1"],
            "Item 1 ID": i1.get("id", ""),
            "Item 1 Stem": i1.get("stem", ""),
            "Item 2 Index": r["item2"],
            "Item 2 ID": i2.get("id", ""),
            "Item 2 Stem": i2.get("stem", ""),
            "Relationship": r.get("relationship", ""),
            "Subtype": r.get("subtype", ""),
            "Similarity Score": r.get("similarity_score", ""),
            "Confidence": r.get("confidence", ""),
            "Stem Overlap": r.get("stem_overlap", ""),
            "Answer Overlap": r.get("answer_overlap", ""),
            "Reason": r.get("reason", ""),
            "Reviewer Decision": review_decisions.get(key, ""),
        })
    return pd.DataFrame(rows)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="eyebrow">Configuration</div>', unsafe_allow_html=True)

    # Show API key status (no input — key is shared via secrets)
    api_key = get_api_key()
    if api_key:
        st.success("✓ API key configured", icon="🔑")
    else:
        st.error("API key not found. Add ANTHROPIC_API_KEY to your Streamlit secrets or environment.")

    st.divider()
    st.markdown("### About")
    st.caption(
        "Detects clone, variant, and enemy item pairs in competency-based assessment banks using semantic NLP analysis.\n\n"
        "**Methodology based on:**\n"
        "- Micir et al. (2022) — TF-IDF + ML pipeline\n"
        "- Stevenor et al. (2025) — Semantic similarity (SBERT approach)\n\n"
        "AI flags require human SME review before updating item bank."
    )

    if st.session_state.stage not in ("upload",):
        st.divider()
        if st.button("🔄 Start Over", use_container_width=True):
            for k in ["stage", "items", "results", "mapping", "raw_df", "filename", "review_decisions"]:
                st.session_state[k] = init_state.__wrapped__() if hasattr(init_state, "__wrapped__") else None
            st.session_state.stage = "upload"
            st.session_state.items = []
            st.session_state.results = []
            st.session_state.mapping = {}
            st.session_state.raw_df = None
            st.session_state.filename = None
            st.session_state.review_decisions = {}
            st.rerun()

# ─── Header ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <div class="eyebrow">Psychometric QA Tool</div>
    <h1>Item Relationship Detector</h1>
    <p>Upload an item bank to surface clones, variants, and enemy pairs — prioritized by semantic similarity for human review.</p>
</div>
""", unsafe_allow_html=True)

# ─── Stage: UPLOAD ────────────────────────────────────────────────────────────

if st.session_state.stage == "upload":
    st.markdown("### Step 1 — Upload your item bank")

    uploaded = st.file_uploader(
        "Drop your spreadsheet here",
        type=["xlsx", "xls", "csv"],
        help="Excel or CSV file with your item bank. Must have at least a column for item stems.",
        label_visibility="collapsed",
    )

    if uploaded:
        with st.spinner("Reading file…"):
            try:
                df = parse_uploaded_file(uploaded)
                st.session_state.raw_df = df
                st.session_state.filename = uploaded.name

                # Auto-detect column mapping
                cols = {c.lower().strip(): c for c in df.columns}
                mapping = {}

                stem_candidates = ["stem", "question", "item", "item text", "question text", "item stem", "text"]
                mapping["stem"] = next((cols[c] for c in stem_candidates if c in cols), None)

                id_candidates = ["id", "item id", "item_id", "#", "no", "number", "item number"]
                mapping["id"] = next((cols[c] for c in id_candidates if c in cols), None)

                key_candidates = ["key", "answer", "correct answer", "correct", "answer key", "correct option"]
                mapping["key"] = next((cols[c] for c in key_candidates if c in cols), None)

                content_candidates = ["content", "content area", "topic", "domain", "category", "content code", "blueprint"]
                mapping["content"] = next((cols[c] for c in content_candidates if c in cols), None)

                for letter, candidates in [
                    ("optA", ["a", "option a", "option 1", "choice a"]),
                    ("optB", ["b", "option b", "option 2", "choice b"]),
                    ("optC", ["c", "option c", "option 3", "choice c"]),
                    ("optD", ["d", "option d", "option 4", "choice d"]),
                ]:
                    mapping[letter] = next((cols[c] for c in candidates if c in cols), None)

                st.session_state.mapping = mapping
                st.session_state.stage = "map"
                st.rerun()
            except Exception as e:
                st.error(f"Could not read file: {e}")

    st.caption("Supported formats: .xlsx · .xls · .csv")

# ─── Stage: MAP ───────────────────────────────────────────────────────────────

elif st.session_state.stage == "map":
    df = st.session_state.raw_df
    mapping = st.session_state.mapping.copy()

    st.markdown(f"### Step 2 — Map columns")
    st.caption(f"**{st.session_state.filename}** · {len(df):,} rows detected")

    col_options = ["(not mapped)"] + list(df.columns)

    def col_idx(val):
        return col_options.index(val) if val in col_options else 0

    with st.form("mapping_form"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Required**")
            mapping["stem"] = st.selectbox("Item stem / question text *", col_options, index=col_idx(mapping.get("stem")))
            st.markdown("**Answer options**")
            mapping["optA"] = st.selectbox("Option A", col_options, index=col_idx(mapping.get("optA")))
            mapping["optB"] = st.selectbox("Option B", col_options, index=col_idx(mapping.get("optB")))
            mapping["optC"] = st.selectbox("Option C", col_options, index=col_idx(mapping.get("optC")))
            mapping["optD"] = st.selectbox("Option D", col_options, index=col_idx(mapping.get("optD")))
        with c2:
            st.markdown("**Optional (improve detection)**")
            mapping["id"] = st.selectbox("Item ID", col_options, index=col_idx(mapping.get("id")))
            mapping["key"] = st.selectbox("Correct answer key ⭐", col_options, index=col_idx(mapping.get("key")),
                                           help="Including the answer key improves detection accuracy (Stevenor et al., 2025)")
            mapping["content"] = st.selectbox("Content area / topic", col_options, index=col_idx(mapping.get("content")),
                                               help="Content codes help detect same-topic enemy pairs (Micir et al., 2022)")

        submitted = st.form_submit_button("Continue →", type="primary", use_container_width=True)

    if submitted:
        stem_col = mapping.get("stem")
        if not stem_col or stem_col == "(not mapped)":
            st.error("Please map the item stem column.")
        else:
            # Clean up "(not mapped)" → None
            clean = {k: (v if v != "(not mapped)" else None) for k, v in mapping.items()}
            st.session_state.mapping = clean
            items = apply_mapping(df, clean)
            if len(items) < 2:
                st.error(f"Only {len(items)} valid items found. Need at least 2.")
            else:
                st.session_state.items = items
                st.session_state.stage = "preview"
                st.rerun()

    if st.button("← Back"):
        st.session_state.stage = "upload"
        st.rerun()

# ─── Stage: PREVIEW ───────────────────────────────────────────────────────────

elif st.session_state.stage == "preview":
    items = st.session_state.items
    n = len(items)
    total_pairs = n * (n - 1) // 2

    st.markdown("### Step 3 — Preview & Run")

    # Stats
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Items", f"{n:,}")
    with c2:
        st.metric("Pairs to evaluate", f"{total_pairs:,}")
    with c3:
        has_key = sum(1 for i in items if i.get("correct_answer"))
        st.metric("Items with answer key", f"{has_key:,}")
    with c4:
        has_content = sum(1 for i in items if i.get("content_area"))
        st.metric("Items with content code", f"{has_content:,}")

    # Preview table
    with st.expander("Preview items", expanded=False):
        preview_data = []
        for item in items[:30]:
            preview_data.append({
                "ID": item["id"],
                "Stem": item["stem"][:120] + ("…" if len(item["stem"]) > 120 else ""),
                "Options": len(item["options"]),
                "Has Key": "✓" if item.get("correct_answer") else "—",
                "Content Area": item.get("content_area") or "—",
            })
        st.dataframe(pd.DataFrame(preview_data), use_container_width=True, hide_index=True)
        if n > 30:
            st.caption(f"Showing 30 of {n} items.")

    st.divider()

    api_key = get_api_key()
    if not api_key:
        st.error("⚠️ API key not configured. Contact your administrator.")
    else:
        if st.button("🔬 Run Detection", type="primary", use_container_width=True):
            try:
                client = anthropic.Anthropic(api_key=api_key)
                with st.spinner(""):
                    results = run_full_analysis(items, client)
                st.session_state.results = results
                st.session_state.review_decisions = {}
                st.session_state.stage = "results"
                st.rerun()
            except anthropic.AuthenticationError:
                st.error("Invalid API key. Contact your administrator.")
            except Exception as e:
                st.error(f"Analysis failed: {e}")

    if st.button("← Back to column mapping"):
        st.session_state.stage = "map"
        st.rerun()

# ─── Stage: RESULTS ───────────────────────────────────────────────────────────

elif st.session_state.stage == "results":
    items = st.session_state.items
    results = st.session_state.results
    review_decisions = st.session_state.review_decisions

    counts = {k: sum(1 for r in results if r.get("relationship") == k) for k in RELATIONSHIP_META}
    reviewed_count = len(review_decisions)

    st.markdown("### Results")

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card clone"><p class="val">{counts["clone"]}</p><p class="lbl">Clones</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card variant"><p class="val">{counts["variant"]}</p><p class="lbl">Variants</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card enemy"><p class="val">{counts["enemy"]}</p><p class="lbl">Enemies</p></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card reviewed"><p class="val">{reviewed_count}</p><p class="lbl">Reviewed</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Controls row
    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 2])
    with ctrl1:
        filter_type = st.selectbox(
            "Filter by type",
            ["All", "Clone", "Variant", "Enemy"],
            index=["All", "Clone", "Variant", "Enemy"].index(
                st.session_state.filter_type.capitalize() if st.session_state.filter_type != "all" else "All"
            ),
            label_visibility="collapsed",
        )
        st.session_state.filter_type = filter_type.lower()
    with ctrl2:
        sort_by = st.selectbox(
            "Sort by",
            ["Similarity score ↓", "Type (clone first)"],
            label_visibility="collapsed",
        )
        st.session_state.sort_by = "score" if "score" in sort_by else "type"
    with ctrl3:
        # Export button
        export_df = results_to_df(results, items, review_decisions)
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            export_df.to_excel(writer, index=False, sheet_name="Flagged Pairs")
        buf.seek(0)
        st.download_button(
            "📥 Export to Excel",
            data=buf,
            file_name="enemy_item_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    st.divider()

    # Filter & sort
    filtered = [r for r in results if st.session_state.filter_type == "all" or r.get("relationship") == st.session_state.filter_type]
    if st.session_state.sort_by == "score":
        filtered.sort(key=lambda r: r.get("similarity_score", 0), reverse=True)
    else:
        filtered.sort(key=lambda r: RELATIONSHIP_META.get(r.get("relationship", ""), {}).get("priority", 9))

    if not filtered:
        st.info("No results match the current filter.")
    else:
        st.caption(f"Showing {len(filtered)} flagged pairs · sorted by {'similarity score' if st.session_state.sort_by == 'score' else 'type'}")

        for r in filtered:
            rel = r.get("relationship", "")
            meta = RELATIONSHIP_META.get(rel, {})
            i1_idx = r.get("item1", 1) - 1
            i2_idx = r.get("item2", 1) - 1
            i1 = items[i1_idx] if i1_idx < len(items) else {}
            i2 = items[i2_idx] if i2_idx < len(items) else {}
            pair_key = f"{r.get('item1')}-{r.get('item2')}"
            score = r.get("similarity_score", 0)
            subtype_labels = {"too_similar": "Text overlap", "giveaway": "Answer giveaway", "same_topic": "Same objective"}

            with st.expander(
                f"**{meta.get('label','?')}** · Item {r.get('item1')} (ID: {i1.get('id','?')}) ↔ Item {r.get('item2')} (ID: {i2.get('id','?')})  ·  Similarity: {int(score*100)}%",
                expanded=False,
            ):
                # Similarity bar
                st.markdown(sim_bar_html(score), unsafe_allow_html=True)

                # Overlap signals
                sig_cols = st.columns(4)
                with sig_cols[0]:
                    st.caption(f"**Subtype:** {subtype_labels.get(r.get('subtype',''), r.get('subtype','—'))}")
                with sig_cols[1]:
                    st.caption(f"**Stem overlap:** {r.get('stem_overlap','—')}")
                with sig_cols[2]:
                    st.caption(f"**Answer overlap:** {r.get('answer_overlap','—')}")
                with sig_cols[3]:
                    st.caption(f"**Confidence:** {r.get('confidence','—')}")

                # Reason
                st.markdown(f"*{r.get('reason', '')}*")

                # Item side-by-side
                box1, box2 = st.columns(2)
                for col, item, idx_label in [(box1, i1, r.get('item1')), (box2, i2, r.get('item2'))]:
                    with col:
                        content_tag = f" &nbsp;[{item.get('content_area')}]" if item.get("content_area") else ""
                        opts_html = ""
                        for j, opt in enumerate(item.get("options", [])):
                            letter = chr(65 + j)
                            correct = (item.get("correct_answer", "").upper() == letter)
                            cls = "item-opt correct" if correct else "item-opt"
                            opts_html += f'<div class="{cls}">{letter}. {opt}</div>'
                        st.markdown(f"""
                        <div class="item-box">
                            <div class="item-id">Item {idx_label} · ID {item.get('id','?')}{content_tag}</div>
                            <div class="item-stem">{item.get('stem','')}</div>
                            {opts_html}
                        </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Human review
                current = review_decisions.get(pair_key, "")
                if current:
                    st.success(f"✓ Reviewed: {current}")
                    if st.button("Clear decision", key=f"clear_{pair_key}"):
                        del st.session_state.review_decisions[pair_key]
                        st.rerun()
                else:
                    decision_cols = st.columns(4)
                    decision_options = ["Confirm enemy", "Confirm variant", "Mark as clone", "Not an issue"]
                    for dcol, dlabel in zip(decision_cols, decision_options):
                        with dcol:
                            if st.button(dlabel, key=f"dec_{pair_key}_{dlabel}", use_container_width=True):
                                st.session_state.review_decisions[pair_key] = dlabel
                                st.rerun()

    st.markdown("""
    <div class="research-note">
    Results sorted by similarity score (highest priority first) per Micir et al. (2022).
    Semantic similarity computed on stem + answer key per Stevenor et al. (2025).
    All AI-generated flags require human SME confirmation before updating item bank coding.
    </div>
    """, unsafe_allow_html=True)
