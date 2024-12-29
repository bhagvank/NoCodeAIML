import streamlit as st
from openai import OpenAI
TOGETHER_API_KEY= "xxxx"
with st.sidebar:
    openai_api_key = st.text_input("Together AI API Key",TOGETHER_API_KEY, key="chatbot_api_key", type="default")
    
    
st.title("📝 GENAI Summarizer")
uploaded_file = st.file_uploader("Upload an article", type=("txt", "md"))
question = st.text_input(
    "Ask something about the article","Can you give me a short summary?",
    placeholder="Can you give me a short summary?",
    disabled=not uploaded_file,
)


if uploaded_file and question and not openai_api_key:
    st.info("Please add your open API key to continue.")

if uploaded_file and question and openai_api_key:
    article = uploaded_file.read().decode()
    prompt = f"""Here's an article:\n\n<article>
    {article}\n\n</article>\n\n
"""

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    client = OpenAI(
      api_key=openai_api_key,
      base_url="https://api.together.xyz/v1",
    )
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = client.chat.completions.create(model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", messages=st.session_state.messages)
    msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
