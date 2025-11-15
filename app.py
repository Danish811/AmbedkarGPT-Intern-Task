import streamlit as st
from main import load_text, split_text, build_or_load_vectorstore, build_qa_chain

st.title("AmbedkarGPT — Minimal RAG UI")


uploaded = st.file_uploader("Upload speech.txt (optional)", type="txt")
if uploaded:
    text = uploaded.read().decode("utf-8")
else:
    try:
        text = load_text("speech.txt")
        st.text("Using local speech.txt")
    except Exception:
        st.info("Upload a speech.txt to proceed.")
        st.stop()
   
if st.button("Build vectorstore & QA"):
    chunks = split_text(text)
    vectordb = build_or_load_vectorstore(chunks, rebuild=False)
    qa = build_qa_chain(vectordb)
    st.success("Ready — ask a question below")
    st.session_state["qa"] = qa

q = st.text_input("Question", value="What does the author say is the 'real remedy' for caste?")
if st.button("Ask") and "qa" in st.session_state:
    qa = st.session_state["qa"]
    res = qa.invoke({"input": q})
    ans = res.get("answer")
    st.subheader("Answer")
    st.write(ans)