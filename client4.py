import asyncio
import os
import sys
import logging
from dotenv import load_dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import AzureChatOpenAI


# ============================================================
# Load environment
# ============================================================
load_dotenv(dotenv_path="D:/Pyn/Test_work/LLMs/.env")


# ============================================================
# Setup Logging
# ============================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)

logger = logging.getLogger("MCP_App")


# ============================================================
# Class: MCP Server Manager
# Handles multiple transports: stdio, HTTP, SSE
# ============================================================
class MCPServerManager:
    def __init__(self, server_configs: dict):
        self.server_configs = server_configs
        self.client = None
        self.logger = logging.getLogger("MCPServerManager")
        self.logger.info("Initialized MCPServerManager")

    async def connect(self):
        """Initialize MultiServer MCP Client."""
        self.logger.info("Connecting to MCP servers...")
        self.logger.debug(f"Server configs: {self.server_configs}")

        try:
            self.client = MultiServerMCPClient(self.server_configs)
            self.logger.info("MCP client connected successfully")
        except Exception as e:
            self.logger.error(f"Failed to connect MCP client: {e}")
            raise

        return self.client

    async def get_tools(self):
        if not self.client:
            raise RuntimeError("Connect MCP client first using connect()")

        self.logger.info("Fetching MCP tools...")
        try:
            tools = await self.client.get_tools()
            self.logger.info(f"Loaded {len(tools)} MCP tools")
            return tools
        except Exception as e:
            self.logger.error(f"Error fetching tools: {e}")
            raise


# ============================================================
# Class: LLM / Agent Manager
# ============================================================
class ReActAgentManager:
    def __init__(self):
        self.agent = None
        self.logger = logging.getLogger("ReActAgentManager")
        self.logger.info("Initialized ReActAgentManager")

    def init_azure_model(self):
        """Initialize Azure OpenAI Model."""
        self.logger.info("Initializing Azure OpenAI model...")

        try:
            model = AzureChatOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                azure_deployment=os.getenv("AZURE_OAI_MODEL", "gpt-4o-mini"),
                temperature=0.3,
            )
            self.logger.info("Azure OpenAI model created successfully")
            return model
        except Exception as e:
            self.logger.error(f"Model initialization failed: {e}")
            raise

    def build_agent(self, model, tools):
        """Create ReAct agent."""
        self.logger.info("Building ReAct agent...")
        try:
            self.agent = create_react_agent(model, tools)
            self.logger.info("ReAct agent initialized successfully")
            return self.agent
        except Exception as e:
            self.logger.error(f"Failed to build ReAct agent: {e}")
            raise

    async def ask(self, message: str):
        """Send a message to the agent."""
        if not self.agent:
            raise RuntimeError("Agent not initialized.")

        self.logger.debug(f"Agent received prompt: {message}")

        try:
            response = await self.agent.ainvoke(
                {"messages": [{"role": "user", "content": message}]}
            )
            output = response["messages"][-1].content
            self.logger.debug(f"Agent response: {output}")
            return output
        except Exception as e:
            self.logger.error(f"Agent failed to process message: {e}")
            raise


# ============================================================
# MAIN APP CONTROLLER
# ============================================================
class AppController:
    def __init__(self):
        self.mcp_manager = None
        self.agent_manager = None
        self.logger = logging.getLogger("AppController")
        self.logger.info("AppController initialized")

    async def setup(self):
        """Setup MCP servers + LLM agent"""

        self.logger.info("Starting application setup...")

        # ------------------------------------------------------------
        # Define all your MCP servers
        # ------------------------------------------------------------
        server_configs = {
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
                "command": sys.executable,
                "args": ["weather.py"],
                "transport": "stdio",
            },
            "alerts": {
                "url": "http://localhost:9000/sse",
                "transport": "sse",
            },
            "stock_updates": {
                "url": "http://localhost:9100/sse",
                "transport": "sse",
            }
        }

        # Initialize server manager
        self.mcp_manager = MCPServerManager(server_configs)
        await self.mcp_manager.connect()

        # Load tools
        tools = await self.mcp_manager.get_tools()

        # Initialize Azure model
        self.agent_manager = ReActAgentManager()
        model = self.agent_manager.init_azure_model()

        # Build agent
        self.agent_manager.build_agent(model, tools)

        self.logger.info("Application setup complete")

    async def run_tests(self):
        """Run sample queries to test all integrated MCP servers."""
        self.logger.info("Running MCP test queries...")

        print("\n🔢 Math tool:")
        print(await self.agent_manager.ask("what's (3 + 5) x 12?"))

        print("\n⛅ Weather tool:")
        print(await self.agent_manager.ask(
            "What is the current weather in Los Angeles, California?"
        ))

        print("\n🗄️ Database tool:")
        print(await self.agent_manager.ask(
            "Find the time of day when most people order snacks, and identify the location that receives the highest number of snack orders."
        ))

        print("\n📢 Alerts (SSE) tool:")
        print(await self.agent_manager.ask(
            "Subscribe to critical system alerts and show me the latest event."
        ))

        self.logger.info("All test queries executed successfully")


# ============================================================
# MAIN ENTRYPOINT
# ============================================================
async def main():
    logger.info("Starting MCP App...")

    app = AppController()
    await app.setup()
    await app.run_tests()

    logger.info("MCP App finished execution")


if __name__ == "__main__":
    asyncio.run(main())
