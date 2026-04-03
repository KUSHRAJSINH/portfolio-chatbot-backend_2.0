import asyncio
from dotenv import load_dotenv
load_dotenv()

import os
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_groq import ChatGroq

async def main():

    # MCP
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
    print("Loaded MCP Tools:", [t.name for t in tools])

    """   
    # LLM
    llm = ChatOpenAI(
        model="arcee-ai/trinity-large-preview:free",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPEN_ROUTER_API_KEY"),
        temperature=0
    )
    """
    
    # LLM
    
    llm=ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        timeout=30
    )

    # AGENT (ReAct)
    system_prompt = """
    You are a professional person intelligence agent. 
    
    Your mission: Search, Scrape, and Extract detailed profile info for the requested person.
    
    CRITICAL RULES:
    1. If you cannot find a GitHub or LinkedIn link, DO NOT hallucinate. Instead, ask the user to provide the relevant link so you can proceed with accurate data extraction.
    2. Once you have the extracted JSON profile, you are done. Provide it as the final answer and STOP.
    3. Be concise and professional.
    """


    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt
    )

    query = input("Ask about person: ")

    try:
        result = await agent.ainvoke({"messages": [("user", query)]})

        print("\nFinal Result:\n")
        print(result["messages"][-1].content)

    except Exception as e:
        print("\n[ERROR]\n", str(e))


if __name__ == "__main__":
    asyncio.run(main())