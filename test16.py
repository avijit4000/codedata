# from langchain_openai import AzureChatOpenAI
# from langchain.chains import ConversationChain
# from langchain.memory import ConversationBufferMemory
# import os
# from dotenv import load_dotenv
#
# load_dotenv(dotenv_path="D:/Pyn/Test_work/LLMs/.env")
#
# azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
# api_key = os.getenv("AZURE_OPENAI_API_KEY")
# api_version = os.getenv("AZURE_OPENAI_API_VERSION")
# azure_model = os.getenv("AZURE_OAI_MODEL")
# print(azure_endpoint, api_key, api_version, azure_model)
#
# # Initialize Azure Chat model
# chat_model = AzureChatOpenAI(
#     openai_api_key=api_key,
#     azure_endpoint=azure_endpoint,
#     azure_deployment=azure_model,
#     api_version=api_version,
#     temperature=0.7
# )
#
# # Update memory key to 'history' (required in new LangChain)
# memory = ConversationBufferMemory(memory_key="history", return_messages=True)
#
# # Conversation chain
# conversation = ConversationChain(
#     llm=chat_model,
#     memory=memory,
#     input_key="input"   # ensures input variable matches prompt
# )
#
#
# print("Azure OpenAI Chatbot (type 'exit' to quit)")
# while True:
#     user_input = input("You: ")
#     if user_input.lower() in ["exit", "quit"]:
#         print("Chatbot: Goodbye!")
#         break
#     response = conversation.run(user_input)
#     print(f"Chatbot: {response}")

#
# from langchain_openai import AzureChatOpenAI
# from langchain.memory import ConversationBufferMemory
# import os
# from dotenv import load_dotenv
#
# # Load environment variables
# load_dotenv(dotenv_path="D:/Pyn/Test_work/LLMs/.env")
#
# azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
# api_key = os.getenv("AZURE_OPENAI_API_KEY")
# api_version = os.getenv("AZURE_OPENAI_API_VERSION")
# azure_model = os.getenv("AZURE_OAI_MODEL")
#
# # Initialize Azure Chat model
# chat_model = AzureChatOpenAI(
#     openai_api_key=api_key,
#     azure_endpoint=azure_endpoint,
#     azure_deployment=azure_model,
#     api_version=api_version,
#     temperature=0.7
# )
#
# # Memory for conversation context (optional, keeps chat history)
# memory = ConversationBufferMemory(memory_key="history", return_messages=True)
#
# # Form questions
# questions = {
#     "first_name": "What is your first name?",
#     "last_name": "What is your last name?",
#     "address": "Please provide your address.",
#     "gender": "What is your gender? (Male/Female/Other)",
#     "terms_and_conditions": "Do you agree to the terms and conditions? (Yes/No)"
# }
#
# # Store user responses
# responses = {}
#
# print("Form Chatbot (type 'exit' anytime to quit)")
#
# # Loop through questions
# for key, question in questions.items():
#     while True:
#         user_input = input(f"{question}\nYou: ")
#         if user_input.lower() in ["exit", "quit"]:
#             print("Chatbot: Exiting the form. Goodbye!")
#             exit()
#
#         # Optional: Let LLM validate/echo response
#         chat_response = chat_model.generate([{"role": "user", "content": f"{question} Answer: {user_input}"}])
#         print(f"Chatbot: {chat_response.generations[0][0].text.strip()}")
#
#         # Save answer and move to next question
#         responses[key] = user_input
#         break
#
# # Show all collected responses at the end
# print("\nAll responses submitted:")
# for key, answer in responses.items():
#     print(f"{key}: {answer}")
#


#
# from langchain_openai import AzureChatOpenAI
# from langchain.schema import HumanMessage
# from langchain.memory import ConversationBufferMemory
# import os
# from dotenv import load_dotenv
# load_dotenv(dotenv_path="D:/Pyn/Test_work/LLMs/.env")
#
# azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
# api_key = os.getenv("AZURE_OPENAI_API_KEY")
# api_version = os.getenv("AZURE_OPENAI_API_VERSION")
# azure_model = os.getenv("AZURE_OAI_MODEL")
#
# chat_model = AzureChatOpenAI(
#     openai_api_key=api_key,
#     azure_endpoint=azure_endpoint,
#     azure_deployment=azure_model,
#     api_version=api_version,
#     temperature=0.7
# )
#
# memory = ConversationBufferMemory(memory_key="history", return_messages=True)
#
# questions = {
#     "first_name": "What is your first name?",
#     "last_name": "What is your last name?",
#     "address": "Please provide your address.",
#     "gender": "What is your gender? (Male/Female/Other)",
#     "terms_and_conditions": "Do you agree to the terms and conditions? (Yes/No)"
# }
#
# responses = {}
#
# print("Form Chatbot (type 'exit' anytime to quit)")
#
# # Loop through questions
# for key, question in questions.items():
#     while True:
#         user_input = input(f"{question}\nYou: ")
#         if user_input.lower() in ["exit", "quit"]:
#             print("Chatbot: Exiting the form. Goodbye!")
#             exit()
#
#         # Send user input to LLM using HumanMessage
#         chat_response = chat_model([HumanMessage(content=f"{question} Answer: {user_input}")])
#         print(f"Chatbot: {chat_response.content.strip()}")
#
#         # Save answer and move to next question
#         responses[key] = user_input
#         break
#
# print("\nAll responses submitted:")
# for key, answer in responses.items():
#     print(f"{key}: {answer}")








# from langchain_openai import AzureChatOpenAI
# from langchain.schema import HumanMessage
# from langchain.memory import ConversationBufferMemory
# import os
# from dotenv import load_dotenv
# from langchain.chains import LLMChain
# from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
#
# load_dotenv(dotenv_path="D:/Pyn/Test_work/LLMs/.env")
#
# azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
# api_key = os.getenv("AZURE_OPENAI_API_KEY")
# api_version = os.getenv("AZURE_OPENAI_API_VERSION")
# azure_model = os.getenv("AZURE_OAI_MODEL")
#
# chat_model = AzureChatOpenAI(
#     openai_api_key=api_key,
#     azure_endpoint=azure_endpoint,
#     azure_deployment=azure_model,
#     api_version=api_version,
#     temperature=0.7
# )
#
# memory = ConversationBufferMemory(memory_key="history", return_messages=True)
#
# questions = {
#     "first_name": "What is your first name?",
#     "last_name": "What is your last name?",
#     "address": "Please provide your address.",
#     "gender": "What is your gender? (Male/Female/Other)",
#     "terms_and_conditions": "Do you agree to the terms and conditions? (Yes/No)"
# }
# responses = {}
# asked_keys = list(questions.keys())
# i = 0
# while i < len(asked_keys):
#     key = asked_keys[i]
#     question = questions[key]
#     user_input = input(f"{question}\nYou: ")
#
#     if user_input.lower() in ["exit", "quit"]:
#         print("Chatbot: Exiting the form. Goodbye!")
#         exit()
#
#
#     class prompTemplate:
#         def question_ask(self,asking_question: str, user_input: str) -> str:
#             return f"""
#                 You are a validation AI. A form chatbot asked the user a question: "{asking_question}".
#
#                 Now check if the user’s input is an actual answer to the question OR if the user is trying to correct a previous field.
#
#                 User Input: "{user_input}"
#
#                 If the input is a **valid answer**, respond strictly with: "ANSWER"
#                 If the input is **not an answer** and seems like a correction (e.g., user says "wrong", "change", "I need to update", etc.), respond strictly with: "CORRECTION" and mention which field they want to correct if identifiable.
#                 """
#
#     template_builder = prompTemplate()
#     question_template = template_builder.question_ask(user_input)
#     prompt = PromptTemplate(input_variables=["user_input"], template=question_template)
#     correction_match = LLMChain(llm=chat_model, prompt=prompt, verbose=False)
#
#     if correction_match:
#         correction_field = correction_match.group(1).lower()
#         if correction_field in questions:
#             print(f"Chatbot: Okay, let's correct your {correction_field}.")
#             i = asked_keys.index(correction_field)
#             continue
#
#     chat_response = chat_model([HumanMessage(content=f"{question} Answer: {user_input}")])
#     print(f"Chatbot: {chat_response.content.strip()}")
#
#     responses[key] = user_input
#     i += 1
#
# print("\nAll responses submitted:")
# for key, answer in responses.items():
#     print(f"{key}: {answer}")


#
# from langchain_openai import AzureChatOpenAI
# from langchain.schema import HumanMessage
# import os
# from dotenv import load_dotenv
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
#
# # Load environment
# load_dotenv(dotenv_path="D:/Pyn/Test_work/LLMs/.env")
#
# azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
# api_key = os.getenv("AZURE_OPENAI_API_KEY")
# api_version = os.getenv("AZURE_OPENAI_API_VERSION")
# azure_model = os.getenv("AZURE_OAI_MODEL")
#
# chat_model = AzureChatOpenAI(
#     openai_api_key=api_key,
#     azure_endpoint=azure_endpoint,
#     azure_deployment=azure_model,
#     api_version=api_version,
#     temperature=0.3
# )
#
# # Form questions
# questions = {
#     "first_name": "What is your first name?",
#     "last_name": "What is your last name?",
#     "address": "Please provide your address.",
#     "gender": "What is your gender? (Male/Female/Other)",
#     "terms_and_conditions": "Do you agree to the terms and conditions? (Yes/No)"
# }
#
# responses = {}
# asked_keys = list(questions.keys())
# i = 0
#
# # Prompt to check if input is answer or correction
# validation_prompt = PromptTemplate(
#     input_variables=["asking_question", "user_input", "questions_list"],
#     template="""
# You are a validation AI. A form chatbot asked the user a question: "{asking_question}".
#
# User replied: "{user_input}"
#
# List of possible fields: {questions_list}
#
# Your task:
# - If this is a valid answer to the question, return: ANSWER
# - If this is a correction like "I need to change first name" or "wrong last name", return: CORRECT <field_name>
#
# Only reply exactly "ANSWER" or "CORRECT <field_name>".
# """
# )
#
# validator = LLMChain(llm=chat_model, prompt=validation_prompt, verbose=False)
#
# print("Form Chatbot Started (type 'exit' anytime to quit)")
# while i < len(asked_keys):
#     key = asked_keys[i]
#     question = questions[key]
#     user_input = input(f"{question}\nYou: ")
#
#     if user_input.lower() in ["exit", "quit"]:
#         print("Chatbot: Exiting. Goodbye!")
#         exit()
#
#     # Validate input meaning
#     validation_result = validator.run({
#         "asking_question": question,
#         "user_input": user_input,
#         "questions_list": ", ".join(questions.keys())
#     }).strip()
#
#     if validation_result.startswith("CORRECT"):
#         correction_field = validation_result.replace("CORRECT", "").strip()
#         if correction_field in questions:
#             print(f"Chatbot: Okay, updating {correction_field}.")
#             i = asked_keys.index(correction_field)
#             continue
#
#     # Store answer normally
#     responses[key] = user_input
#     print(f"Chatbot: ✅ Recorded.")
#     i += 1
#
# print("\n🎉 Form Completed. Final Responses:")
# for key, answer in responses.items():
#     print(f"{key}: {answer}")


#
# from langchain_openai import AzureChatOpenAI
# from langchain.schema import HumanMessage
# import os
# from dotenv import load_dotenv
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
#
# # Load environment
# load_dotenv(dotenv_path="D:/Pyn/Test_work/LLMs/.env")
#
# azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
# api_key = os.getenv("AZURE_OPENAI_API_KEY")
# api_version = os.getenv("AZURE_OPENAI_API_VERSION")
# azure_model = os.getenv("AZURE_OAI_MODEL")
#
# chat_model = AzureChatOpenAI(
#     openai_api_key=api_key,
#     azure_endpoint=azure_endpoint,
#     azure_deployment=azure_model,
#     api_version=api_version,
#     temperature=0.3
# )
#
# # Form questions
# questions = {
#     "first_name": "What is your first name?",
#     "last_name": "What is your last name?",
#     "address": "Please provide your address.",
#     "gender": "What is your gender? (Male/Female/Other)",
#     "terms_and_conditions": "Do you agree to the terms and conditions? (Yes/No)"
# }
#
# responses = {}
# asked_keys = list(questions.keys())
# i = 0  # current index pointer
#
# # Prompt to detect correction
# validation_prompt = PromptTemplate(
#     input_variables=["asking_question", "user_input", "questions_list"],
#     template="""
# You are a validation AI. A form chatbot asked the user a question: "{asking_question}".
#
# User replied: "{user_input}"
#
# List of possible fields: {questions_list}
#
# Your task:
# - If this is a valid answer to the question, return: ANSWER
# - If this is a correction like "I need to change first name" or "wrong last name", return: CORRECT <field_name>
#
# Only reply exactly "ANSWER" or "CORRECT <field_name>".
# """
# )
#
# validator = LLMChain(llm=chat_model, prompt=validation_prompt, verbose=False)
#
# print("Form Chatbot Started (type 'exit' anytime to quit)")
#
# while i < len(asked_keys):
#     key = asked_keys[i]
#     question = questions[key]
#     user_input = input(f"{question}\nYou: ")
#
#     if user_input.lower() in ["exit", "quit"]:
#         print("Chatbot: Exiting. Goodbye!")
#         exit()
#
#     # Validate input meaning
#     validation_result = validator.run({
#         "asking_question": question,
#         "user_input": user_input,
#         "questions_list": ", ".join(questions.keys())
#     }).strip()
#
#     # If correction triggered
#     if validation_result.startswith("CORRECT"):
#         correction_field = validation_result.replace("CORRECT", "").strip()
#         if correction_field in questions:
#             print(f"Chatbot: Okay, updating {correction_field}.")
#             i = asked_keys.index(correction_field)  # move pointer to that question
#             continue
#
#     # Save answer normally and move forward
#     responses[key] = user_input
#     print(f"Chatbot: ✅ Recorded.")
#     i += 1
#
# print("\n🎉 Form Completed. Final Responses:")
# for key, answer in responses.items():
#     print(f"{key}: {answer}")


#
# from langchain_openai import AzureChatOpenAI
# from langchain.schema import HumanMessage
# import os
# from dotenv import load_dotenv
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
#
# # Load environment
# load_dotenv(dotenv_path="D:/Pyn/Test_work/LLMs/.env")
#
# azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
# api_key = os.getenv("AZURE_OPENAI_API_KEY")
# api_version = os.getenv("AZURE_OPENAI_API_VERSION")
# azure_model = os.getenv("AZURE_OAI_MODEL")
#
# chat_model = AzureChatOpenAI(
#     openai_api_key=api_key,
#     azure_endpoint=azure_endpoint,
#     azure_deployment=azure_model,
#     api_version=api_version,
#     temperature=0.3
# )
#
# # Form questions
# questions = {
#     "first_name": "What is your first name?",
#     "last_name": "What is your last name?",
#     "address": "Please provide your address.",
#     "gender": "What is your gender? (Male/Female/Other)",
#     "terms_and_conditions": "Do you agree to the terms and conditions? (Yes/No)"
# }
#
# responses = {}
# asked_keys = list(questions.keys())
# i = 0  # current index pointer
#
# # Prompt to detect correction
# validation_prompt = PromptTemplate(
#     input_variables=["asking_question", "user_input", "questions_list"],
#     template="""
# You are a validation AI. A form chatbot asked the user a question: "{asking_question}".
#
# User replied: "{user_input}"
#
# List of possible fields: {questions_list}
#
# Your task:
# - If this is a valid answer to the question, return: ANSWER
# - If this is a correction like "I need to change first name" or "wrong last name", return: CORRECT <field_name>
#
# Only reply exactly "ANSWER" or "CORRECT <field_name>".
# """
# )
#
# validator = LLMChain(llm=chat_model, prompt=validation_prompt, verbose=False)
#
# print("Form Chatbot Started (type 'exit' anytime to quit)")
#
# while i < len(asked_keys):
#     key = asked_keys[i]
#     question = questions[key]
#     user_input = input(f"{question}\nYou: ")
#
#     if user_input.lower() in ["exit", "quit"]:
#         print("Chatbot: Exiting. Goodbye!")
#         exit()
#
#     # Validate input meaning
#     validation_result = validator.run({
#         "asking_question": question,
#         "user_input": user_input,
#         "questions_list": ", ".join(questions.keys())
#     }).strip()
#
#     # If correction triggered
#     if validation_result.startswith("CORRECT"):
#         correction_field = validation_result.replace("CORRECT", "").strip()
#         if correction_field in questions:
#             print(f"Chatbot: Okay, updating {correction_field}.")
#             i = asked_keys.index(correction_field)  # move pointer to that question
#             continue
#
#     # Save answer normally and move forward
#     responses[key] = user_input
#     print(f"Chatbot: ✅ Recorded.")
#
#     i += 1  # move to next field
#
# # Final Output
# print("\n🎉 Form Completed. Final Responses:")
# for key, answer in responses.items():
#     print(f"{key}: {answer}")



from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

load_dotenv(dotenv_path="D:/Pyn/Test_work/LLMs/.env")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
azure_model = os.getenv("AZURE_OAI_MODEL")

chat_model = AzureChatOpenAI(
    openai_api_key=api_key,
    azure_endpoint=azure_endpoint,
    azure_deployment=azure_model,
    api_version=api_version,
    temperature=0.3
)

questions = {
    "first_name": "What is your first name?",
    "last_name": "What is your last name?",
    "address": "Please provide your address.",
    "gender": "What is your gender? (Male/Female/Other)",
    "terms_and_conditions": "Do you agree to the terms and conditions? (Yes/No)"
}

responses = {}
asked_keys = list(questions.keys())

validation_prompt = PromptTemplate(
    input_variables=["asking_question", "user_input", "questions_list"],
    template="""
        You are a validation AI. A form chatbot asked the user a question: "{asking_question}".
        
        User replied: "{user_input}"
        
        List of possible fields: {questions_list}
        
        Your task:
        - If this is a valid answer to the question, return: ANSWER
        - If this is a correction like "I need to change first name" or "wrong last name", return: CORRECT <field_name>
        
        Only reply exactly "ANSWER" or "CORRECT <field_name>".
        """
        )

validator = LLMChain(llm=chat_model, prompt=validation_prompt, verbose=False)

print("Form Chatbot Started (type 'exit' anytime to quit)")

i = 0
return_index = None
while i < len(asked_keys):
    key = asked_keys[i]
    question = questions[key]

    user_input = input(f"{question}\nYou: ").strip()

    if user_input.lower() in ["exit", "quit"]:
        print("Chatbot: Exiting. Goodbye!")
        exit()

    validation_result = (
        validator.run({
            "asking_question": question,
            "user_input": user_input,
            "questions_list": ", ".join(questions.keys())
        })
        .strip()
    )

    if validation_result.startswith("CORRECT"):
        correction_field = validation_result.replace("CORRECT", "").strip()
        if correction_field in questions:
            print(f"Chatbot: Okay, updating {correction_field}.")
            if return_index is None:
                return_index = i
            i = asked_keys.index(correction_field)
            continue

    responses[key] = user_input
    if return_index is not None and key in questions and key != asked_keys[return_index]:
        pass
    if return_index is not None and i == asked_keys.index(key):
        i = return_index
        return_index = None
    else:
        i += 1

print("\n🎉 Form Completed. Final Responses:")
for k, answer in responses.items():
    print(f"{k}: {answer}")
