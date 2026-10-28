
"""Step 4: Streamlit GIF Search Application.

Run: streamlit run app.py
"""

import streamlit as st

from search import search_gif

# Page configuration
st.set_page_config(
    page_title="GIF Search",
    layout="wide"
)

st.title("GIF Search")
st.caption("Describe the GIF in your own words.")

# User input
query = st.text_input(
    "Describe a GIF",
    placeholder="a dog being confused"
)

k = st.slider(
    "Results",
    min_value=3,
    max_value=20,
    value=9
)

if query:
    try:
        # Search GIFs and measure execution time
        results, timings = search_gif(
            query,
            k,
            return_timing=True
        )

    except RuntimeError as e:
        st.error(str(e))
        st.stop()

    # Performance metrics
    st.subheader("Search Performance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Model Load",
        f"{timings['load_ms']:.2f} ms"
    )

    col2.metric(
        "Embedding Time",
        f"{timings['embedding_ms']:.2f} ms"
    )

    col3.metric(
        "Vector Search",
        f"{timings['vector_search_ms']:.2f} ms"
    )

    col4.metric(
        "Total Search Time",
        f"{timings['total_ms']:.2f} ms"
    )

    st.divider()

    # Display GIF results
    st.subheader(f"Top {len(results)} GIF Results")

    columns = st.columns(3)

    for i, r in enumerate(results):
        with columns[i % 3]:
            st.markdown(
                f'<img src="{r["url"]}" '
                f'style="width:100%">',
                unsafe_allow_html=True
            )

            st.caption(
                f"{r['description']} "
                f"({r['score']:.2f})"
            )
