from openai import OpenAI
import streamlit as st
TOGETHER_API_KEY= "xxxx"
with st.sidebar:
    openai_api_key = st.text_input("Together AI API Key",TOGETHER_API_KEY, key="chatbot_api_key", type="default")
    
st.title("💬 No Code Gen AI Platform")
st.caption("🚀 A Streamlit chatbot powered by Together.AI")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    #client = OpenAI(api_key=openai_api_key)
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
