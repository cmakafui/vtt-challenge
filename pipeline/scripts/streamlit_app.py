import asyncio
import json
import os
from pathlib import Path

import streamlit as st
from fastmcp import Client
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool, Tool
from langchain_openai import AzureChatOpenAI

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

st.title("mozAIc")

config_path = os.getenv("AZURE_CONFIG_PATH")
model = "gpt-4.1-mini"
config = json.loads(Path(config_path).read_text())[model]
llm = None


async def list_tools():
    # Connect via stdio to a local script
    async with Client("pipeline/mcp/innovation_entity_server.py") as client:
        tools = await client.list_tools()
        # print(f"Available tools: {tools}")
        # st.session_state.messages.append({"role": "assistant", "content": f"Available tools: {tools}"})

        lc_tools = [Tool(
            args_schema=tool.inputSchema,
            name=tool.name,
            description=tool.description,
            func=lambda *_: ...,
        ) for tool in tools]
    return lc_tools

async def run_tool(tool_name, tool_args):
    # Connect via stdio to a local script
    async with Client("pipeline/mcp/innovation_entity_server.py") as client:
        result = await client.call_tool(tool_name, tool_args)
        return result




# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Set a default model
if "openai_model" not in st.session_state:
    # call list_tools synchronously
    lc_tools = loop.run_until_complete(list_tools())

    llm = AzureChatOpenAI(
        model=model,
        api_key=config['api_key'],
        api_version=config['api_version'],
        azure_endpoint=config['api_base']
    )

    llm = llm.bind_tools(
        tools=lc_tools,
        tool_choice="auto",  # Automatically choose the best tool
    )

    st.session_state["openai_model"]  = llm



# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message: AIMessage = st.session_state["openai_model"].invoke(
            input=st.session_state.messages,
        )

        if message.tool_calls:
            run_tool_res = loop.run_until_complete(
                run_tool(
                    tool_name=message.tool_calls[0]['name'],
                    tool_args=message.tool_calls[0]['args'],
                )
            )
            st.markdown(f"Tool calls: {message.tool_calls}\n\nTool result: {run_tool_res}")
            st.session_state.messages.append({"role": "assistant", "content": f"Tool calls: {message.tool_calls}\n\nTool result: {run_tool_res}"})
        else:
            st.session_state.messages.append({"role": "assistant", "content": message.text()})
            st.markdown(message.text())
