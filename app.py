import streamlit as st
import os
from BMW_Chatbot import BMWChatbot



# Page configuration
st.set_page_config(
    page_title="BMW Chatbot",
    page_icon="🚗",
    layout="centered"
)



# Sidebar for branding and info
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/44/BMW.svg/1024px-BMW.svg.png", width=120)
    st.header("BMW Chatbot")
    st.info(
        """
        **Features:**
        - Answers questions about BMW cars from bmw.csv using TF-IDF + cosine similarity.
        - Handles greetings, thanks, help, and exit intents.
        - Uses simple NLP to detect year and fuel type and filters the dataset accordingly.
        - Can answer aggregate questions:
            - "average price of 3 series 2017"
            - "cheapest 5 series diesel"
            - "most expensive 1 series"
        - Logs all interactions to chat_log.txt
        - Has a special mode to analyze sentiment + readability of a text:
            - "analyze: <your review text>"
        """
    )

# Header
st.title("🚗 BMW Chatbot")
st.caption("Ask about BMW models, prices, specs, or get recommendations based on your budget!")


# Initialize the chatbot (cached to prevent reloading on every interaction)
@st.cache_resource
def get_chatbot():
    csv_path = "bmw.csv"
    if not os.path.exists(csv_path):
        st.error(f"Error: '{csv_path}' not found. Please make sure the dataset is in the same directory.")
        return None
    try:
        bot = BMWChatbot(csv_path)
        return bot
    except Exception as e:
        st.error(f"Failed to initialize chatbot: {e}")
        return None

bot = get_chatbot()



# Welcome message (only on first load)
if "welcomed" not in st.session_state:
    st.success("Welcome to the BMW Chatbot! 🚗")
    st.session_state.welcomed = True

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm BMWBot. I can help you find the perfect BMW, check prices, or analyze car reviews. How can I help you today?"}
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if message["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(f"**BMWBot:** {message['content']}")
    else:
        with st.chat_message("user"):
            st.markdown(f"**You:** {message['content']}")

# Accept user input
prompt = st.chat_input("Ask about a BMW (e.g., 'Best car under 20k', 'Price of X5')...")
if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(f"**You:** {prompt}")

    # Generate assistant response
    if bot:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = bot.generate_reply(prompt)
                st.markdown(f"**BMWBot:** {response}")
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.caption("BMW Chatbot | Developed by Mohammad Hamim | 202280090114")
