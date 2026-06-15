import streamlit as st
import ollama

prompt = st.text_input("Ask something")

if st.button("Ask LLM"):

    response = ollama.chat(
    model="gemma2:2b",
    messages=[{"role":"user","content":prompt}],
    keep_alive="30m"
)

    st.write(response['message']['content'])