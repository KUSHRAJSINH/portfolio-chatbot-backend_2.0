import asyncio
import streamlit as st
from dotenv import load_dotenv

# Load env variables
load_dotenv()

import os
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient


# -----------------------------
# Async Agent Function (UNCHANGED LOGIC)
# -----------------------------
async def run_agent(query):

    client = MultiServerMCPClient(
        {
            "person-search": {
                "command": "python",
                "args": ["mcp_server/server.py"],
                "transport": "stdio",
            }
        }
    )

    tools = await client.get_tools()

    llm = ChatOpenAI(
        model="arcee-ai/trinity-large-preview:free",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPEN_ROUTER_API_KEY"),
        temperature=0
    )

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt="""
You are a person search agent.

Workflow:
1. Use search tool to find links
2. Use scrape tool to get content
3. Use extract tool to structure data

Rules:
- Always pass proper arguments
- Never call tools without input
- Do not guess URLs

Return:
- Name
- Role
- Company
- Links
"""
    )

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": query}]}
    )

    return result["messages"][-1].content


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="Person Search Agent",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 AI Person Search Agent")
st.write("Search any person and extract their profile using MCP tools.")

# Input box
query = st.text_input("Enter person's name:", placeholder="e.g. Sundar Pichai")

# Button
if st.button("Search"):

    if not query.strip():
        st.warning("Please enter a name.")
    else:
        with st.spinner("🔎 Searching and extracting data..."):

            try:
                # Run async inside Streamlit
                result = asyncio.run(run_agent(query))

                st.success("✅ Result Found")

                st.markdown("### 📄 Profile Details")
                st.write(result)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")


# Footer
st.markdown("---")
st.caption("Built with MCP + LangChain + Groq 🚀")