import streamlit as st
from openai import OpenAI

with st.sidebar:
    openai_api_key = st.text_input("Together AI API Key","48666fef627332bcc4c315d79903619276e1469f1d628a6193df3c0d128d50ae", key="chatbot_api_key", type="default")
    
    
st.title("📝 Together.ai Summarizer")
uploaded_file = st.file_uploader("Upload an article", type=("txt", "md"))
question = st.text_input(
    "Ask something about the article","Can you give me a short summary?",
    placeholder="Can you give me a short summary?",
    disabled=not uploaded_file,
)

topics = """"You are responsible for feeding the question to an agent that given context will try to answer the question. I want you to produce an output that would contain the key topics discussed by the customer in the review and a brief takeaway for each topic. 
Topics should be within - Living Situation, Recent Activity, Family Members, Interests and Hobbies,
Daily Routine,Decision Making Process,Additional Insights,Key Takeaways,Possible areas for deeper analysis, and Work Schedule.
The context may or may not be relevant. Rewrite the question to highlight the fact that
only some pieces of context (or none) maybe be relevant."""

if uploaded_file and question and not openai_api_key:
    st.info("Please add your open API key to continue.")

if uploaded_file and question and openai_api_key:
    article = uploaded_file.read().decode()
    prompt = f"""Here's an article:\n\n<article>
    {article}\n\n</article>\n\n {topics}
"""

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
