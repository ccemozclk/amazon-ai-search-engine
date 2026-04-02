import os
import time
import requests
import streamlit as st


BASE_URL = os.getenv("BACKEND_URL", "http://localhost:8001")
FULL_API_URL = f"{BASE_URL.rstrip('/')}/search"


st.set_page_config(page_title="Amazon AI Search", page_icon="🛒", layout="wide")


if "results" not in st.session_state:
    st.session_state.results = None
if "latency" not in st.session_state:
    st.session_state.latency = None
if "used_model" not in st.session_state:
    st.session_state.used_model = None
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "rec_trigger" not in st.session_state:
    st.session_state.rec_trigger = None


def run_search(query: str, model_type: str, top_k: int, model_label: str):
    """Execute the search API call and update session state."""
    with st.spinner("AI is scanning the shelves..."):
        try:
            payload = {"query": query, "top_k": top_k, "model_type": model_type}
            start_time = time.time()
            response = requests.post(FULL_API_URL, json=payload, timeout=10)
            response.raise_for_status()
            end_time = time.time()

            data = response.json()
            st.session_state.results    = data.get("results", [])
            st.session_state.latency    = end_time - start_time
            st.session_state.used_model = model_label
            st.session_state.last_query = query

        except requests.exceptions.Timeout:
            st.warning("⏳ The AI scan took too long, please try again.")
            st.session_state.results = None
        except requests.exceptions.ConnectionError:
            st.error("🔌 The API server is unreachable. Make sure your Docker containers are running.")
            st.session_state.results = None
        except Exception as e:
            st.error(f"⚠️ An unexpected error occurred: {e}")
            st.session_state.results = None


st.title("🛒 Amazon AI Search Engine")
st.markdown("Type what you want to search for in natural language. *(Example: Red summer dress or comfortable sneakers)*")


with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.radio(
        "🧠 Search Engine Intelligence",
        options=["Modern AI (SentenceTransformer)", "Traditional AI (Word2Vec)"],
        help="Compare the results of two different AI architectures!"
    )
    top_k = st.slider("How many items should be brought?", min_value=1, max_value=12, value=3, step=1)
    active_model_name = "all-MiniLM-L6-v2" if "Modern" in model_choice else "Skip-Gram (Word2Vec)"
    model_type_param  = "sentence_transformer" if "Modern" in model_choice else "word2vec"

    st.markdown("---")
    st.markdown(f"""
    **System Architecture:**
    * 🧠 **Model:** {active_model_name}
    * 🗄️ **Database:** Qdrant Vector DB
    * ⚡ **Cache:** Redis
    """)


st.markdown("### 🪄 AI Auto-Complete")
col_auto1, col_auto2 = st.columns([4, 1])
with col_auto1:
    auto_text = st.text_input(
        "Auto-complete input",
        placeholder="Example: black leather...",
        label_visibility="collapsed"
    )
with col_auto2:
    auto_btn = st.button("Predict next word", use_container_width=True)

if auto_btn and auto_text:
    try:
        res = requests.post(f"{BASE_URL}/autocomplete", json={"text": auto_text, "top_k": 3}, timeout=5)
        if res.status_code == 200:
            suggs = res.json().get("suggestions", [])
            if suggs:
                st.success(f"Suggestions: **{', '.join(suggs)}**")
            else:
                st.warning("I haven't learned this word combination yet.")
    except Exception:
        st.error("Connection Error!")

st.markdown("---")

# ── Search form ──────────────────────────────────────────────────────────────
with st.form(key="search_form"):
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        query = st.text_input(
            "Search input",
            placeholder="Black leather jacket...",
            label_visibility="collapsed"
        )
    with col_btn:
        submit_btn = st.form_submit_button("🔎 Search", use_container_width=True)

# Determine what triggered a search: the form OR a recommendation click
active_query = None
triggered_by_rec = False

if submit_btn and query:
    active_query = query
elif st.session_state.rec_trigger:
    active_query         = st.session_state.rec_trigger
    triggered_by_rec     = True
    st.session_state.rec_trigger = None

if active_query:
    if triggered_by_rec:
        st.info(f"🔗 Searching based on recommendation: **{active_query}**")
    run_search(active_query, model_type_param, top_k, active_model_name)

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.results is not None:
    results    = st.session_state.results
    latency    = st.session_state.latency
    used_model = st.session_state.used_model
    last_query = st.session_state.last_query

    if not results:
        st.warning("Unfortunately, I couldn't find a product that matches your search query.")
    else:
        st.markdown("---")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric(label="⏱️ Response Time", value=f"{latency:.3f} sec")
        with m2:
            st.metric(label="🛍️ Found Products", value=len(results))
        with m3:
            st.metric(label="🧠 Model", value=used_model)
        with m4:
            st.metric(label="🔍 Query", value=f'"{last_query}"')

        st.markdown("<br>", unsafe_allow_html=True)

        cols_per_row = 3
        for i in range(0, len(results), cols_per_row):
            cols      = st.columns(cols_per_row)
            row_items = results[i : i + cols_per_row]

            for col_idx, (col, item) in enumerate(zip(cols, row_items)):
                with col:
                    with st.container(border=True):
                        # ── Product info ─────────────────────────────────
                        product_name = item.get("product_name", "Unnamed Product")
                        st.subheader(product_name)
                        st.caption(f"ASIN: {item.get('asin', '-')}")

                        price = item.get("price", 0.0)
                        if price and float(price) > 0:
                            st.write(f"**Price:** ${price:.2f}")
                        else:
                            st.write("**Price:** Not Available")

                        rating  = item.get("rating", 0.0)
                        reviews = item.get("review_count", 0)
                        st.write(f"⭐ **{rating}** ({reviews} reviews)")
                        st.metric("Similarity Score", f"{item.get('score', 0) * 100:.1f}%")

                        # ── "Customers who bought this also bought" ───────
                        recommendations = item.get("recommendations", [])
                        if recommendations:
                            st.markdown("---")
                            st.markdown("🛒 **Customers who bought this also bought:**")
                            for r_i, rec_name in enumerate(recommendations):
                                # Truncate long names for the button label
                                label     = rec_name if len(rec_name) <= 40 else rec_name[:37] + "..."
                                btn_key   = f"rec_{i}_{col_idx}_{r_i}"
                                if st.button(
                                    f"🔍 {label}",
                                    key=btn_key,
                                    use_container_width=True,
                                    help=rec_name,
                                ):
                                    st.session_state.rec_trigger = rec_name
                                    st.rerun()
