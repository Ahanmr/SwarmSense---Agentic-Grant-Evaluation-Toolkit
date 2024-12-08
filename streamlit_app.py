import streamlit as st
import sys
import time
from typing import Iterator, Optional
sys.path.append('./gaia-swarm')
from swarm import Swarm
from agents import based_agent
from openai import OpenAI
import os

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'mode' not in st.session_state:
    st.session_state.mode = 'chat'
if 'client' not in st.session_state:
    st.session_state.client = Swarm()
if 'auto_running' not in st.session_state:
    st.session_state.auto_running = False

class StreamHandler:
    def __init__(self, placeholder, initial_text=""):
        self.placeholder = placeholder
        self.text = initial_text
        self.tool_calls = []

    def on_content(self, content: str):
        if content:
            self.text += content
            self.placeholder.markdown(self.text + "▌")

    def on_tool_call(self, tool_name: str):
        if tool_name:
            tool_text = f"\n🛠️ *Using tool: {tool_name}*"
            self.tool_calls.append(tool_text)
            self.placeholder.markdown(self.text + "\n".join(self.tool_calls) + "▌")

    def complete(self):
        final_text = self.text
        if self.tool_calls:
            final_text += "\n" + "\n".join(self.tool_calls)
        self.placeholder.markdown(final_text)
        return final_text

def process_stream_response(response: Iterator[dict], stream_handler: StreamHandler) -> str:
    """Process streaming response with real-time updates"""
    for chunk in response:
        if "content" in chunk and chunk["content"] is not None:
            stream_handler.on_content(chunk["content"])
            
        if "tool_calls" in chunk and chunk["tool_calls"] is not None:
            for tool_call in chunk["tool_calls"]:
                f = tool_call["function"]
                name = f["name"]
                if name:
                    stream_handler.on_tool_call(name)

        if "delim" in chunk and chunk["delim"] == "end":
            break

    return stream_handler.complete()

def run_autonomous_action():
    """Run an autonomous action and display results"""
    thought = (
        "Be creative and do something interesting on the Base blockchain. "
        "Choose an action and execute it now. Choose those that highlight your identity and abilities best."
    )
    
    # Add autonomous thought to chat history
    st.session_state.messages.append({"role": "user", "content": thought})
    
    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        stream_handler = StreamHandler(message_placeholder)
        
        response = st.session_state.client.run(
            agent=based_agent,
            messages=[{"role": "user", "content": thought}],
            stream=True
        )
        
        full_response = process_stream_response(response, stream_handler)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    return full_response

def handle_user_input():
    """Handle user input in chat mode"""
    if prompt := st.chat_input("What would you like the Based Agent to do?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Generate and display assistant response
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            stream_handler = StreamHandler(message_placeholder)
            
            response = st.session_state.client.run(
                agent=based_agent,
                messages=st.session_state.messages,
                stream=True
            )
            
            full_response = process_stream_response(response, stream_handler)
            
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

def main():
    st.set_page_config(
        page_title="Based Agent Interface",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Based Agent Interface")
    
    # Mode selection in sidebar
    with st.sidebar:
        st.title("Settings")
        selected_mode = st.radio(
            "Choose Operation Mode",
            options=["Chat", "Autonomous"],
            index=0,
            key="mode_selector"
        )
        
        if selected_mode == "Autonomous":
            interval = st.slider(
                "Action Interval (seconds)",
                min_value=2,
                max_value=10,
                value=3
            )
            
            if st.button("Start Autonomous Mode", type="primary"):
                st.session_state.auto_running = True
            
            if st.button("Stop Autonomous Mode", type="secondary"):
                st.session_state.auto_running = False
    
    # Main chat interface
    if selected_mode == "Chat":
        # Create a container for the chat history
        chat_container = st.container()
        
        # Display chat messages from history
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else None):
                    st.markdown(message["content"])
        
        handle_user_input()
    
    # Autonomous mode
    else:
        # Create a container for the chat history
        chat_container = st.container()
        
        # Display chat messages from history
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else None):
                    st.markdown(message["content"])
        
        if getattr(st.session_state, 'auto_running', False):
            run_autonomous_action()
            time.sleep(interval)
            st.rerun()

if __name__ == "__main__":
    main() 