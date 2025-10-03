from langchain_openai import AzureChatOpenAI
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from datetime import datetime
import os
from dotenv import load_dotenv

# Load env
load_dotenv(dotenv_path="D:/Pyn/Test_work/LLMs/.env")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
azure_model = os.getenv("AZURE_OAI_MODEL")

# ✅ Azure OpenAI model
llm = AzureChatOpenAI(
    openai_api_key=api_key,
    azure_endpoint=azure_endpoint,
    openai_api_version=api_version,
    deployment_name=azure_model,
    temperature=0,
)

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# ✅ Your custom template
template = """
You are a helpful assistant. A question was asked: "{asking_question}".

Based on the user's input below, provide **only the exact answer** relevant to the question.

- If the question is about "date of birth" or any date field, detect the date in any format (e.g., "April 19, 1999", "19/04/1999") and return it in DD-MM-YYYY format.
- For other fields, return the relevant answer as-is, without any extra text, punctuation, or explanation.

User input: "{user_ans}"

Output:
"""

# ✅ Prompt with memory placeholder and template
prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm
with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# -------- DEMO --------
config = {"configurable": {"session_id": "user1"}}

# Step 1: Provide info
with_memory.invoke(
    {"input": "My name is Avijit Biswas.", "asking_question": "What is my first name?", "user_ans": "Avijit Biswas"},
    config=config
)

response = with_memory.invoke(
    {"input": "What is my first name?", "asking_question": "What is my first name?", "user_ans": "Avijit Biswas"},
    config=config
)
print(response.content)
