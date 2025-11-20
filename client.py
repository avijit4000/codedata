# from langchain_mcp_adapters.client import MultiServerMCPClient
# from langgraph.prebuilt import create_react_agent
# from langchain_groq import ChatGroq
#
# from dotenv import load_dotenv
#
# load_dotenv()
#
# import asyncio
#
#
# async def main():
#     client = MultiServerMCPClient(
#         {
#             "math": {
#                 "command": "python",
#                 "args": ["mathserver.py"],  ## Ensure correct absolute path
#                 "transport": "stdio",
#
#             },
#             "weather": {
#                 "url": "http://localhost:8000/mcp",  # Ensure server is running here
#                 "transport": "streamable_http",
#             }
#
#         }
#     )
#
#     import os
#     os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
#
#     tools = await client.get_tools()
#     model = ChatGroq(model="qwen-qwq-32b")
#     agent = create_react_agent(
#         model, tools
#     )
#
#     math_response = await agent.ainvoke(
#         {"messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]}
#     )
#
#     print("Math response:", math_response['messages'][-1].content)
#
#     weather_response = await agent.ainvoke(
#         {"messages": [{"role": "user", "content": "what is the weather in California?"}]}
#     )
#     print("Weather response:", weather_response['messages'][-1].content)
#
#
# asyncio.run(main())



# from langchain_mcp_adapters.client import MultiServerMCPClient
# from langchain.agents import create_agent
# from langchain_openai import AzureChatOpenAI
#
# from dotenv import load_dotenv
# import asyncio
# import os
#
# # Load environment variables
# load_dotenv(dotenv_path="D:/Pyn/Test_work/LLMs/.env")
#
# async def main():
#     client = MultiServerMCPClient(
#         {
#             "math": {
#                 "command": "python",
#                 "args": ["mathserver.py"],
#                 "transport": "stdio",
#             },
#             "weather": {
#                 "url": "http://localhost:8000/mcp",
#                 "transport": "streamable_http",
#             },
#         }
#     )
#
#     azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
#     api_key = os.getenv("AZURE_OPENAI_API_KEY")
#     api_version = os.getenv("AZURE_OPENAI_API_VERSION")
#     azure_model = os.getenv("AZURE_OAI_MODEL", "gpt-4o-mini")
#
#     model = AzureChatOpenAI(
#         azure_endpoint=azure_endpoint,
#         api_key=api_key,
#         api_version=api_version,
#         azure_deployment=azure_model,  # deployment name on Azure
#         temperature=0.3,
#         streaming=False,
#     )
#
#     # Fetch MCP tools
#     tools = await client.get_tools()
#
#     # Create agent with Azure model
#     agent = create_agent(model, tools)
#
#     # Math tool test
#     math_response = await agent.ainvoke(
#         {"messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]}
#     )
#     print("Math response:", math_response["messages"][-1].content)
#
#     # Weather tool test
#     weather_response = await agent.ainvoke(
#         {"messages": [{"role": "user", "content": "what is the weather in California?"}]}
#     )
#     print("Weather response:", weather_response["messages"][-1].content)
#
#
# if __name__ == "__main__":
#     asyncio.run(main())


#
# from langchain_mcp_adapters.client import MultiServerMCPClient
# from langchain.agents import create_agent
# from langchain_openai import AzureChatOpenAI
# from dotenv import load_dotenv
# import asyncio
# import os
#
# # Load environment variables
# load_dotenv(dotenv_path="D:/Pyn/Test_work/LLMs/.env")
#
# async def main():
#     # Initialize MultiServer MCP client
#     client = MultiServerMCPClient(
#         {
#             "math": {
#                 "command": "python",
#                 "args": ["mathserver.py"],
#                 "transport": "stdio",
#             },
#             "weather": {
#                 "url": "http://localhost:8000/mcp",
#                 "transport": "streamable_http",
#             },
#         }
#     )
#
#     # Load Azure credentials
#     azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
#     api_key = os.getenv("AZURE_OPENAI_API_KEY")
#     api_version = os.getenv("AZURE_OPENAI_API_VERSION")
#     azure_model = os.getenv("AZURE_OAI_MODEL", "gpt-4o-mini")
#
#     # Create Azure LLM model
#     model = AzureChatOpenAI(
#         azure_endpoint=azure_endpoint,
#         api_key=api_key,
#         api_version=api_version,
#         azure_deployment=azure_model,
#         temperature=0.3,
#         streaming=False,
#     )
#
#     # Fetch MCP tools
#     tools = await client.get_tools()
#
#     # Create agent
#     agent = create_agent(model, tools)
#
#     print("✅ Connected to MCP Agent. Type your question or 'quit' to exit.\n")
#
#     # Continuous interactive loop
#     while True:
#         user_input = input("You: ").strip()
#         if user_input.lower() in ["quit", "exit"]:
#             print("👋 Exiting MCP Agent. Goodbye!")
#             break
#
#         try:
#             response = await agent.ainvoke(
#                 {"messages": [{"role": "user", "content": user_input}]}
#             )
#             print("Agent:", response["messages"][-1].content)
#         except Exception as e:
#             print("⚠️ Error:", str(e))
#
# if __name__ == "__main__":
#     asyncio.run(main())



# from langchain_mcp_adapters.client import MultiServerMCPClient
# from langchain.agents import create_agent
# from langchain_openai import AzureChatOpenAI
# from dotenv import load_dotenv
# import requests
# import asyncio
# import os
# from bs4 import BeautifulSoup  # for parsing HTML
#
# # Load environment variables
# load_dotenv(dotenv_path="D:/Pyn/Test_work/LLMs/.env")
#
# # Custom tool to fetch info from dashboard
# async def fetch_dashboard_data(query: str) -> str:
#     """
#     Fetch data from Optum dashboard and extract key information.
#     (You can modify based on how the dashboard exposes info.)
#     """
#     url = "http://mr-portals-monitoring.optum.com/acquisition/overview/ampKeyMetricsView"
#     try:
#         response = requests.get(url, timeout=10)
#         soup = BeautifulSoup(response.text, "html.parser")
#
#         # Example: extract metric cards, tables, or text content
#         metrics = soup.get_text(separator=" ", strip=True)[:2000]  # limit content
#         return f"Dashboard data summary (truncated): {metrics}"
#     except Exception as e:
#         return f"Failed to fetch dashboard: {e}"
#
# async def main():
#     client = MultiServerMCPClient({
#         "math": {"command": "python", "args": ["mathserver.py"], "transport": "stdio"},
#         "weather": {"url": "http://localhost:8000/mcp", "transport": "streamable_http"},
#     })
#
#     azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
#     api_key = os.getenv("AZURE_OPENAI_API_KEY")
#     api_version = os.getenv("AZURE_OPENAI_API_VERSION")
#     azure_model = os.getenv("AZURE_OAI_MODEL", "gpt-4o-mini")
#
#     model = AzureChatOpenAI(
#         azure_endpoint=azure_endpoint,
#         api_key=api_key,
#         api_version=api_version,
#         azure_deployment=azure_model,
#         temperature=0.3,
#     )
#
#     tools = await client.get_tools()
#
#     # Add dashboard info tool
#     async def dashboard_tool(input_text: str):
#         return await fetch_dashboard_data(input_text)
#
#     # Add tool manually
#     tools.append({
#         "name": "fetch_dashboard_info",
#         "description": "Fetches and interprets metrics from Optum monitoring dashboard.",
#         "func": dashboard_tool,
#     })
#
#     agent = create_agent(model, tools)
#
#     print("🤖 Ask_Isaac is ready! Type your question or 'quit' to exit.\n")
#
#     while True:
#         user_input = input("You: ").strip()
#         if user_input.lower() in ["quit", "exit"]:
#             print("👋 Goodbye from Ask_Isaac!")
#             break
#
#         try:
#             # If question contains keyword related to dashboard, trigger fetch
#             if "metric" in user_input.lower() or "dashboard" in user_input.lower():
#                 dashboard_info = await fetch_dashboard_data(user_input)
#                 user_input += f"\n\nReference data:\n{dashboard_info}"
#
#             response = await agent.ainvoke(
#                 {"messages": [{"role": "user", "content": user_input}]}
#             )
#             print("Ask_Isaac:", response["messages"][-1].content)
#
#         except Exception as e:
#             print("⚠️ Error:", e)
#
#
# if __name__ == "__main__":
#     asyncio.run(main())

# --------------------------------------------------------------------------------------------------

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import asyncio
import os
import sys

# Load environment variables
load_dotenv()
# dotenv_path="D:/Pyn/Test_work/LLMs/.env"

async def main():
    # Connect to math (stdio) and weather (HTTP) MCP servers
    import sys

    client = MultiServerMCPClient(
        {
            "math": {
                "command": sys.executable,
                "args": ["mathserver.py"],
                "transport": "stdio",
            },
            "matrix_server_stdio": {
                "command": sys.executable,
                "args": ["matrix_server_2.py"],
                "transport": "stdio",
            },
            "matrix_server": {
                "url": "http://localhost:8000/mcp",
                "transport": "streamable_http",
            },
            "weather": {
                "command": sys.executable,
                "args": ["weather.py"],
                "transport": "stdio",
            },
        }
    )

    # Initialize Azure OpenAI model
    model = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_deployment=os.getenv("AZURE_OAI_MODEL", "gpt-4o-mini"),
        temperature=0.3,
    )

    # Load MCP tools
    tools = await client.get_tools()

    # Create ReAct agent
    agent = create_react_agent(model, tools)

    # Test math tool
    # math_response = await agent.ainvoke(
    #     {"messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]}
    # )
    # print("Math response:", math_response["messages"][-1].content)

    # # Test weather tool
    # weather_response = await agent.ainvoke(
    #     {"messages": [{"role": "user", "content": "What is the current weather in Los Angeles, California?"}]}
    # )
    # print("Weather response:", weather_response["messages"][-1].content)

    db_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What is the mean 90th percentile latency across all days in October 2025?"}]}
    )
    print("Database response:", db_response["messages"][-1].content)

    db_response_2 = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What is the mean 90th percentile latency across all days in October 2025?"}]}
    )
    print("Database response:", db_response_2["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())


# ------------------------------------------------------------------------------------------------



