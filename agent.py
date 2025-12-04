import asyncio
from langgraph.prebuilt import create_react_agent
from src.tools.utils.logging import logger
from src.LLM.llm_model.llm_model import llmtoolmodel
from src.LLM.mcp_server.multi_server import multi_server


class MCPToolCallSystem:

    async def setup(self):
        logger.info("Loading model and tools...")

        # Inject MCP client into llm model
        llmtoolmodel.client = multi_server.client

        # Get model + tools
        self.model = llmtoolmodel.model
        self.tools = await llmtoolmodel.setup()

        logger.info("Creating ReAct agent...")
        self.agent = create_react_agent(self.model, self.tools)

    async def ask(self, prompt: str):
        logger.info(f"User Prompt: {prompt}")

        response = await self.agent.ainvoke(
            {"messages": [{"role": "user", "content": prompt}]}
        )

        final_answer = response["messages"][-1].content
        logger.info(f"Response: {final_answer}")
        return final_answer

    # async def run_tests(self):
    #     logger.info("Running test queries...")
    #
    #     math_q = "what's (3 + 5) x 12?"
    #     weather_q = "What is the weather in Los Angeles?"
    #     db_q = "What is the average 90th percentile latency across all dates?"
    #     # db_q = "which date the highest 90th percentile latency last week?"
    #     # db_q = "what is the avaraget js error rate last week?"
    #
    #     print("\nMath Response:")
    #     # print(await self.ask(math_q))
    #
    #     print("\nWeather Response:")
    #     # print(await self.ask(weather_q))
    #
    #     print("\nDatabase Response:")
    #     print(await self.ask(db_q))

    # async def run_tests(self):
    #     logger.info("MCP Chatbot Ready. Ask your question...")
    #
    #     # Take ONE question from user
    #     user_question = input("\nEnter your question: ").strip()
    #
    #     if not user_question:
    #         print("No question entered. Exiting...")
    #         return
    #
    #     # Get response
    #     response = await self.ask(user_question)
    #
    #     print("\nFinal Response:")
    #     print(response)
    #
    #     logger.info("Session completed. Closing program...")

    async def run_tests(self):
        logger.info("MCP Chatbot Ready. Type your question.")
        logger.info("Type 'quit' to exit.\n")

        while True:
            user_question = input("You: ").strip()

            if not user_question:
                print("⚠️ Please enter a question.\n")
                continue

            if user_question.lower() in {"quit", "exit", "bye"}:
                print("\n✅ Session ended. Goodbye!")
                logger.info("User exited the chatbot.")
                break

            try:
                response = await self.ask(user_question)
                print("\nAgent:", response, "\n")
            except Exception as e:
                logger.error(f"Error while processing question: {e}")
                print("\n❌ Error processing your question.\n")




async def main():
    system = MCPToolCallSystem()
    await system.setup()
    await system.run_tests()


if __name__ == "__main__":
    asyncio.run(main())
