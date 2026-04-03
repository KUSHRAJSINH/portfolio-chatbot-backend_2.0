import streamlit as st
import asyncio
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

# Page Config
st.set_page_config(page_title="Person Search Agent", page_icon="🕵️", layout="wide")
load_dotenv()

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Core Logic ---
async def run_agent(query):
    # Initialize MCP Client
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

    system_prompt = """
    You are a professional person search agent.
    
    Your Workflow:
    1. SEARCH: Find relevant links for the person.
    2. SCRAPE: Get the raw content from those links.
    3. EXTRACT: Structure that content into the requested format.

    Rules:
    - Only use tools when necessary.
    - If a tool requires an argument, provide it clearly.
    - Provide a structured final answer.
    """

    # Using 'prompt' as the keyword for the system instructions
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt
    )

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": query}]}
    )
    return result["messages"][-1].content

# --- Frontend Interface ---
st.title("🕵️ Person Intelligence Agent")
st.caption("Advanced search and data extraction via MCP & Llama 3.1")

# Layout: Two columns
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.subheader("Search Controls")
    query = st.text_input("Who should I find?", placeholder="e.g. Raghav Patel")
    search_button = st.button("Run Research")
    
    with st.expander("Tool Configuration", expanded=False):
        st.write("**MCP Server:** person-search")
        st.write("**Engine:** Groq Llama 3.1")
        st.write("**Workflow:** Search → Scrape → Extract")

with col2:
    st.subheader("Research Results")
    if search_button:
        if not query:
            st.warning("Please enter a name.")
        else:
            try:
                with st.status("🛠️ Agent is executing workflow...", expanded=True) as status:
                    st.write("Connecting to MCP Server...")
                    # Run the async agent
                    final_answer = asyncio.run(run_agent(query))
                    status.update(label="✅ Analysis Complete!", state="complete", expanded=False)

                st.markdown("### Final Report")
                st.info(final_answer)
                
            except Exception as e:
                st.error(f"Execution Error: {str(e)}")
    else:
        st.info("Results will appear here once the agent finishes its search.")

# Footer
st.divider()
st.markdown("<center>Powered by <b>Model Context Protocol</b> & <b>LangGraph</b></center>", unsafe_allow_html=True)