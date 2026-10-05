"""
Council Assist - Streamlit chat interface
-----------------------------------------
A simple browser-based chat app for residents to ask questions about
City of Melbourne services. It shows the answer, the council sources used,
and whether the answer came from Ollama or the extractive fallback.

Run with:  streamlit run app.py
     (on Windows, if "streamlit" isn't found: python -m streamlit run app.py)
"""

import streamlit as st
from rag import CouncilAssist, ollama_available, OLLAMA_MODEL, THRESHOLD

st.set_page_config(page_title="Council Assist", page_icon="🏛️", layout="wide")


# Load the assistant once and keep it cached (so the KB isn't reloaded every click).
@st.cache_resource
def load_assistant():
    return CouncilAssist()


assistant = load_assistant()


# ---------------------------------------------------------------------------
# Sidebar - project info and example questions
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏛️ Council Assist")
    st.caption("City of Melbourne resident FAQ assistant")

    st.markdown("**Pipeline**")
    st.markdown(
        f"- Retrieval: BM25 (built from scratch), top-3\n"
        f"- Generator: Ollama `{OLLAMA_MODEL}`\n"
        f"- Refusal threshold: BM25 ≥ {THRESHOLD}"
    )

    st.markdown("**Status**")
    if ollama_available():
        st.markdown("🟢 Ollama connected")
    else:
        st.markdown("🟡 Ollama not running – extractive mode")

    st.markdown(f"**Knowledge base:** {len(assistant.kb)} passages")

    st.markdown("**Try asking**")
    example_questions = [
        "How many free hard waste collections do I get each year?",
        "Should I bag my soft plastics and put them in the recycling?",
        "Can I get a rebate for installing solar panels?",
    ]
    for example in example_questions:
        if st.button(example, use_container_width=True):
            st.session_state.pending_question = example


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
st.title("Council Assist")
st.caption(
    "Ask everyday questions about City of Melbourne services. Answers come only "
    "from official council information — if it's not in the knowledge base, I'll say so."
)

# Keep the conversation history in session state.
if "messages" not in st.session_state:
    st.session_state.messages = []


def render_message(message):
    """Draw a single chat message (user or assistant)."""
    avatar = "🧑" if message["role"] == "user" else "🏛️"
    with st.chat_message(message["role"], avatar=avatar):

        # User messages are just plain text.
        if message["role"] == "user":
            st.markdown(message["content"])
            return

        # Assistant messages include the answer plus source details.
        response = message["response"]

        if response["answered"]:
            st.markdown(response["answer"])
            st.success(
                f"Answered from {len(response['sources'])} council sources · {response['mode']}",
                icon="✅",
            )
        else:
            st.warning(response["answer"], icon="🛑")
            st.caption(f"Declined: {response['mode']}")

        # Show which passages were retrieved, with scores and source links.
        with st.expander("Sources used", expanded=response["answered"]):
            for doc, score in response["sources"]:
                st.markdown(
                    f"**[{doc['id']}] {doc['title']}** — relevance score {score:.2f}  \n"
                    f"<span style='color:#5b6b6b;font-size:0.9em'>{doc['text']}</span>  \n"
                    f"[{doc['source']}]({doc['source']})",
                    unsafe_allow_html=True,
                )


# Show the existing conversation.
for message in st.session_state.messages:
    render_message(message)


# Get the next question - either typed, or from an example button.
question = st.chat_input("e.g. When does my pet registration expire?")
if st.session_state.get("pending_question"):
    question = st.session_state.pop("pending_question")

if question:
    # Add and show the user's message.
    user_message = {"role": "user", "content": question}
    st.session_state.messages.append(user_message)
    render_message(user_message)

    # Get the assistant's answer and show it.
    assistant_message = {"role": "assistant", "response": assistant.ask(question)}
    st.session_state.messages.append(assistant_message)
    render_message(assistant_message)