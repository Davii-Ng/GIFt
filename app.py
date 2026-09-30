"""Step 4: a small web page for searching.

Run:  streamlit run app.py
Streamlit re-runs this whole file from top to bottom every time the user types or moves a slider.
"""
import streamlit as st

from search import search_gif

st.set_page_config(page_title="GIF Search", layout="wide")
st.title("GIF Search")
st.caption("Describe the GIF in your own words.")

# Input widgets. Each returns the current value.
query = st.text_input("Describe a GIF", placeholder="a dog being confused")
k = st.slider("Results", 3, 20, 9)

if query:
    try:
        results = search_gif(query, k)
    except RuntimeError as e:  # raised by search.py when the index has not been built
        st.error(str(e))
        st.stop()

    # Show the GIFs in a grid of 3 columns.
    columns = st.columns(3)
    for i, r in enumerate(results):
        with columns[i % 3]:  # i % 3 cycles through columns 0, 1, 2, 0, 1, 2 ...
            st.markdown(f'<img src="{r["url"]}" style="width:100%">', unsafe_allow_html=True)
            st.caption(f"{r['description']} ({r['score']:.2f})")
