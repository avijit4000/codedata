# from langchain_openai import AzureChatOpenAI
# from langchain.chains import ConversationChain
# from langchain.memory import ConversationBufferMemory
# from datetime import datetime
# import os
# from dotenv import load_dotenv
#
# # ✅ Configure your Azure OpenAI deployment
#
# load_dotenv(dotenv_path="D:/Pyn/Test_work/LLMs/.env")
# azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
# api_key = os.getenv("AZURE_OPENAI_API_KEY")
# api_version = os.getenv("AZURE_OPENAI_API_VERSION")
# azure_model = os.getenv("AZURE_OAI_MODEL")
#
# llm = AzureChatOpenAI(
#     openai_api_key=api_key,
#     azure_endpoint=azure_endpoint,
#     openai_api_version=api_version,
#     deployment_name=azure_model,   # Replace with your Azure deployment model name
#     temperature=0
# )
#
# # ✅ Create memory to store conversation locally
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#
# # ✅ Build a conversational chain
# conversation = ConversationChain(
#     llm=llm,
#     memory=memory,
#     verbose=True
# )
#
# # -------- DEMO --------
#
# # Step 1: Provide info (this gets stored in memory)
# conversation.predict(input="My name is Avijit Biswas. My date of birth is 19-04-1993.")
#
# # Step 2: Ask questions one by one
# print(conversation.predict(input="What is my first name?"))
# print(conversation.predict(input="What is my last name?"))
#
# # Step 3: Age calculation (let’s guide the model to compute correctly)
# today = datetime.today()
# dob = datetime.strptime("19-04-1993", "%d-%m-%Y")
# age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
#
# print(f"My age is {age} years.")


#
# from langchain_openai import AzureChatOpenAI
# from langchain.memory import ChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from datetime import datetime
# import os
# from dotenv import load_dotenv
#
# # Load env
# load_dotenv(dotenv_path="D:/Pyn/Test_work/LLMs/.env")
# azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
# api_key = os.getenv("AZURE_OPENAI_API_KEY")
# api_version = os.getenv("AZURE_OPENAI_API_VERSION")
# azure_model = os.getenv("AZURE_OAI_MODEL")
#
# # ✅ Azure OpenAI model
# llm = AzureChatOpenAI(
#     openai_api_key=api_key,
#     azure_endpoint=azure_endpoint,
#     openai_api_version=api_version,
#     deployment_name=azure_model,
#     temperature=0,
# )
#
# # ✅ Memory store (in-RAM)
# store = {}
#
# def get_session_history(session_id: str):
#     if session_id not in store:
#         store[session_id] = ChatMessageHistory()
#     return store[session_id]
#
# # ✅ Prompt with memory placeholder
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a helpful assistant that remembers user details."),
#     MessagesPlaceholder(variable_name="history"),
#     ("human", "{input}")
# ])
#
# # ✅ Chain with memory
# chain = prompt | llm
# with_memory = RunnableWithMessageHistory(
#     chain,
#     get_session_history,
#     input_messages_key="input",
#     history_messages_key="history",
# )
#
# # -------- DEMO --------
# config = {"configurable": {"session_id": "user1"}}
#
# # Step 1: Provide info
# with_memory.invoke({"input": "My name is Avijit Biswas. My date of birth is 19-04-1996."}, config=config)
#
# # Step 2: Ask questions
# print(with_memory.invoke({"input": "What is my first name?"}, config=config))
# print(with_memory.invoke({"input": "What is my last name?"}, config=config))
# print(with_memory.invoke({"input": "What is my date of birth year?"}, config=config))
# # Step 3: Age calculation (Python side)
# today = datetime.today()
# dob = datetime.strptime("19-04-1993", "%d-%m-%Y")
# age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
# print(f"My age is {age} years.")



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

# ✅ Prompt with memory placeholder
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant that remembers user details. "
     "When answering questions, always reply in a friendly style like:\n"
     "- 'Hi, your first name is Avijit.'\n"
     "- 'Hi, your last name is Biswas.'\n"
     "- 'Hi, your date of birth year is 1996.'\n"),
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
with_memory.invoke({"input": "My name is Avijit Biswas. "}, config=config)

# Step 2: Ask questions
print(with_memory.invoke({"input": "What is my first name?"}, config=config))
print(with_memory.invoke({"input": "What is my last name?"}, config=config))
print(with_memory.invoke({"input": "Where are you leaving?"}, config=config))

response = with_memory.invoke({"input": "What is my first name?"}, config=config)
print(response.content)