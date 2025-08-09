import os
from typing import List, Dict

import streamlit as st
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()

# -----------------------------
# Configuration and Prompts
# -----------------------------

SYSTEM_PROMPT = (
    "You are a helpful senior Python engineer. "
    "Your primary goal is to answer Python programming questions clearly and accurately. "
    "Provide concise explanations and working, minimal code examples when helpful. "
    "Prefer Python's standard library when possible and explain trade-offs. "
    "If a question is not about Python, briefly decline and steer the user back to Python topics."
)

DEFAULT_MODEL = "gpt-4"
ALLOWED_MODELS = ["gpt-4", "gpt-3.5-turbo"]


# -----------------------------
# Session State Management
# -----------------------------

def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of {"role": "user"|"assistant", "content": str}
    if "model" not in st.session_state:
        st.session_state.model = DEFAULT_MODEL
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.2


def clear_chat() -> None:
    st.session_state.messages = []
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()


# -----------------------------
# OpenAI Interaction
# -----------------------------

def build_openai_messages() -> List[Dict[str, str]]:
    history = st.session_state.messages[-40:]  # limit context size
    return [{"role": "system", "content": SYSTEM_PROMPT}] + history


def get_assistant_response(model: str, temperature: float) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=build_openai_messages(),
        temperature=temperature,
    )
    return response.choices[0].message.content


# -----------------------------
# UI Rendering
# -----------------------------

def render_sidebar() -> None:
    st.sidebar.header("Settings")
    st.session_state.model = st.sidebar.selectbox(
        "Model",
        options=ALLOWED_MODELS,
        index=ALLOWED_MODELS.index(DEFAULT_MODEL),
        help="Choose the language model."
    )
    st.session_state.temperature = st.sidebar.slider(
        "Creativity (temperature)",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.temperature,
        step=0.05,
        help="Higher values produce more varied answers; lower values are more focused."
    )

    st.sidebar.button("Clear chat", on_click=clear_chat)

    api_key_present = bool(os.getenv("OPENAI_API_KEY", "") or st.secrets.get("OPENAI_API_KEY", ""))
    if not api_key_present:
        st.sidebar.warning("No OpenAI API key found. Set OPENAI_API_KEY in your environment or Streamlit secrets.")


def render_chat_history() -> None:
    for msg in st.session_state.messages:
        if msg["role"] in ("user", "assistant"):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])


def main() -> None:
    st.set_page_config(page_title="Python Programming Chatbot", page_icon="🐍")
    st.title("🐍 Python Programming Chatbot")
    st.caption("Ask Python questions about syntax, libraries, best practices, and more.")

    init_session_state()
    render_sidebar()
    render_chat_history()

    user_input = st.chat_input("Ask a Python programming question")
    if user_input:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    reply = get_assistant_response(
                        model=st.session_state.model,
                        temperature=st.session_state.temperature,
                    )
                except Exception as e:
                    st.error(f"An error occurred while contacting the model: {e}")
                    reply = "Sorry, I ran into an error. Please try again."

            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()