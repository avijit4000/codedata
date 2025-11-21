import os
import sys
import asyncio
import logging
from dotenv import load_dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import AzureChatOpenAI


# -------------------------------------------------------
# Setup Logging
# -------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("a2a_mcp.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("A2A-MCP")


# -------------------------------------------------------
# Main A2A + MCP Orchestrator Class
# -------------------------------------------------------
class A2AMCPSystem:
    def __init__(self):
        load_dotenv()

        logger.info("Initializing MCP Client...")
        self.client = MultiServerMCPClient(
            {
                "math": {
                    "command": sys.executable,
                    "args": ["mathserver.py"],
                    "transport": "stdio",
                },
                "database": {
                    "url": "http://localhost:8000/mcp",
                    "transport": "streamable_http",
                },
                "weather": {
                    "url": "http://localhost:8000/mcp",
                    "transport": "streamable_http",
                },
            }
        )

        logger.info("Initializing Azure OpenAI model...")
        self.model = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_deployment=os.getenv("AZURE_OAI_MODEL", "gpt-4o-mini"),
            temperature=0.3,
        )

        self.agent = None

    async def setup(self):
        """Load tools and create the agent."""
        logger.info("Fetching MCP tools...")
        tools = await self.client.get_tools()

        logger.info("Creating ReAct agent...")
        self.agent = create_react_agent(self.model, tools)

    async def ask(self, prompt: str):
        """Send a prompt to the agent and return the response."""
        logger.info(f"User Prompt: {prompt}")
        response = await self.agent.ainvoke(
            {"messages": [{"role": "user", "content": prompt}]}
        )
        final_answer = response["messages"][-1].content
        logger.info(f"Response: {final_answer}")
        return final_answer

    async def run_tests(self):
        """Run all sample tests."""
        logger.info("Running test queries...")

        math_q = "what's (3 + 5) x 12?"
        weather_q = "What is the current weather in Los Angeles, California?"
        db_q = "Find the time of day when most people order snacks, and identify the location that receives the highest number of snack orders"

        print("\nMath Response:")
        print(await self.ask(math_q))

        print("\nWeather Response:")
        print(await self.ask(weather_q))

        print("\nDatabase Response:")
        print(await self.ask(db_q))


# -------------------------------------------------------
# Main Entrypoint
# -------------------------------------------------------
async def main():
    system = A2AMCPSystem()
    await system.setup()
    await system.run_tests()


if __name__ == "__main__":
    asyncio.run(main())
