# import asyncio
# from src.LLM.agent.agent import main as agent_main
#
# if __name__ == "__main__":
#     asyncio.run(agent_main())

#
# import asyncio
# from fastapi import FastAPI
# from pydantic import BaseModel
#
# # Import your agent system
# from src.LLM.agent.agent import MCPToolCallSystem
#
# app = FastAPI()
#
# # Request body model
# class QuestionRequest(BaseModel):
#     question: str
#
#
# # Create global agent instance
# mcp_system = MCPToolCallSystem()
#
#
# @app.on_event("startup")
# async def startup_event():
#     await mcp_system.setup()
#
#
# @app.post("/ask")
# async def ask_question(payload: QuestionRequest):
#     try:
#         response = await mcp_system.ask(payload.question)
#         return {
#             "question": payload.question,
#             "answer": response
#         }
#     except Exception as e:
#         return {
#             "error": str(e)
#         }
#
#
# # For running directly: python src/main.py
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)


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


import asyncio
from flask import Flask, request, jsonify

# Import your agent system
from src.LLM.agent.agent import MCPToolCallSystem

app = Flask(__name__)

# Create global agent instance
mcp_system = MCPToolCallSystem()

# Setup event loop and initialize agent ONCE
loop = asyncio.get_event_loop()
loop.run_until_complete(mcp_system.setup())


@app.route("/ask", methods=["POST"])
def ask_question():
    try:
        data = request.get_json()
        question = data.get("question")

        if not question:
            return jsonify({"error": "Question is required"}), 400

        # Run async function inside Flask
        response = loop.run_until_complete(mcp_system.ask(question))

        return jsonify({
            "question": question,
            "answer": response
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# Run with: python -m src.main
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
