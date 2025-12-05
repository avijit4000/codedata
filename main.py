# import asyncio
# from src.LLM.agent.agent import main as agent_main
#
# if __name__ == "__main__":
#     asyncio.run(agent_main())


import asyncio
from fastapi import FastAPI
from pydantic import BaseModel

# Import your agent system
from src.LLM.agent.agent import MCPToolCallSystem

app = FastAPI()

# Request body model
class QuestionRequest(BaseModel):
    question: str


# Create global agent instance
mcp_system = MCPToolCallSystem()


@app.on_event("startup")
async def startup_event():
    await mcp_system.setup()


@app.post("/ask")
async def ask_question(payload: QuestionRequest):
    try:
        response = await mcp_system.ask(payload.question)
        return {
            "question": payload.question,
            "answer": response
        }
    except Exception as e:
        return {
            "error": str(e)
        }


# For running directly: python src/main.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)


# pip install fastapi uvicorn pydantic
# python -m src.main
# uvicorn src.main:app --reload
# http://localhost:8000/ask
# Content-Type: application/json
# {
#   "question": "What is MCP?"
# }

# {
#   "question": "What is MCP?",
#   "answer": "MCP stands for Model Context Protocol..."
# }