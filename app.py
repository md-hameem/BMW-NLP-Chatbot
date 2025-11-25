import streamlit as st
import os
from BMW_Chatbot import BMWChatbot

# Page configuration
st.set_page_config(
    page_title="BMW Chatbot",
    page_icon="🚗",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🚗 BMW Chatbot")
st.markdown("Ask me about BMW models, prices, specs, or get recommendations based on your budget!")

# Initialize the chatbot (cached to prevent reloading on every interaction)
@st.cache_resource
def get_chatbot():
    csv_path = "bmw.csv"
    if not os.path.exists(csv_path):
        st.error(f"Error: '{csv_path}' not found. Please make sure the dataset is in the same directory.")
        return None
    
    # Initialize bot
    try:
        bot = BMWChatbot(csv_path)
        return bot
    except Exception as e:
        st.error(f"Failed to initialize chatbot: {e}")
        return None

bot = get_chatbot()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm BMWBot. I can help you find the perfect BMW, check prices, or analyze car reviews. How can I help you today?"}
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask about a BMW (e.g., 'Best car under 20k', 'Price of X5')..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    if bot:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = bot.generate_reply(prompt)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
