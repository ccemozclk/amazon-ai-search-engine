import os
import time
import requests
import streamlit as st


BASE_URL = os.getenv("BACKEND_URL", "http://localhost:8001")
ENDPOINT = "/search" 
FULL_API_URL = f"{BASE_URL.rstrip('/')}{ENDPOINT}"


st.set_page_config(page_title="Amazon AI Search", page_icon="🛒", layout="wide")


if "results" not in st.session_state:
    st.session_state.results = None
if "latency" not in st.session_state:
    st.session_state.latency = None
if "used_model" not in st.session_state:
    st.session_state.used_model = None


st.title("🛒 Amazon AI Search Engine")
st.markdown("Type the word you want to search for in natural language. *(Example: Red summer dress or comfortable sneakers)*")


with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.radio(
        "🧠 Search Engine Intelligence",
        options=["Modern AI (SentenceTransformer)", "Traditional AI (Word2Vec)"],
        help="Compare the results of two different AI architectures!"
        )


    top_k = st.slider("How many items should be brought?", min_value=1, max_value=12, value=3, step=1)

    active_model_name = "all-MiniLM-L6-v2" if "Modern" in model_choice else "Skip-Gram (Word2Vec)"
    
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
    auto_text = st.text_input("Type the first few words of your search phrase.:", placeholder="Example: black leather...", label_visibility="collapsed")
with col_auto2:
    auto_btn = st.button("Predict the next word.", use_container_width=True)

if auto_btn and auto_text:
    try:
        res = requests.post(f"{BASE_URL}/autocomplete", json={"text": auto_text, "top_k": 3}, timeout=5)
        if res.status_code == 200:
            suggs = res.json().get("suggestions", [])
            if suggs:
                st.success(f"Suggestions: **{', '.join(suggs)}**")
            else:
                st.warning("I haven't learned this word combination yet.")
    except Exception as e:
        st.error("Connection Error !! ")

st.markdown("---")

with st.form(key="search_form"):
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        query = st.text_input("What were you looking for?", placeholder="Black leather jacket...", label_visibility="collapsed")
    with col_btn:
        submit_btn = st.form_submit_button("🔎 Search", use_container_width=True)

if submit_btn and query:
    with st.spinner("AI is scanning the shelves..."):
        try:
            model_type_param = "word2vec" if "Word2Vec" in model_choice else "sentence_transformer"
            
            payload = {"query": query, "top_k": top_k, "model_type": model_type_param}

            start_time = time.time()
            response = requests.post(FULL_API_URL, json=payload, timeout=10)
            response.raise_for_status()
            end_time = time.time()

            data = response.json()
            st.session_state.results = data.get("results", [])
            st.session_state.latency = end_time - start_time
            st.session_state.used_model = active_model_name


        except requests.exceptions.Timeout:
            st.warning("⏳ The database is too large! The AI ​​scan took too long, please try again.")
            st.session_state.results = None
        except requests.exceptions.ConnectionError:
            st.error("🔌 The API server is unreachable. Make sure your Docker containers are running.")
            st.session_state.results = None
        except Exception as e:
            st.error(f"⚠️ An unexpected error occurred: {e}")
            st.session_state.results = None


if st.session_state.results is not None:
    results = st.session_state.results
    latency = st.session_state.latency
    used_model = st.session_state.used_model

    if not results:
        st.warning("Unfortunately, I couldn't find a product that matches your recipe.")
    else:
        st.markdown("---")

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(label="⏱️ Response Time", value=f"{latency:.3f} sec")
        with m2:
            st.metric(label="🛍️ Found Product", value=len(results))
        with m3:
            st.metric(label="🧠 Model", value=used_model)

        st.markdown("<br>", unsafe_allow_html=True)

        
        cols_per_row = 3
        for i in range(0, len(results), cols_per_row):
            cols = st.columns(cols_per_row)
            row_items = results[i : i + cols_per_row]

            for col, item in zip(cols, row_items):
                with col:
                    with st.container(border=True):
                        st.subheader(item.get("product_name", "Unnamed Product"))
                        st.caption(f"ASIN: {item.get('asin', '-')}")
                        
                        st.write(f"**Price:** {item.get('price', 'No Price Data')}")

                        rating = item.get('rating', 0.0)
                        reviews = item.get('review_count', 0)
                        st.write(f"⭐ **{rating}** ({reviews} Review)")
                        st.metric("Similarity Score", f"{item.get('score', 0)*100:.1f}%")

                        recommendations = item.get("recommendations", [])
                        if recommendations:
                            st.markdown("---")
                            st.caption("🛒 **Frequently Bought Together:**")
                            st.caption(f"`{', '.join(recommendations)}`")