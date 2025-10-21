import os
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from LLM.prompts.llm_template import prompTemplate
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from ..globals import *



# store = {}

# class FormAssistant:
#     def __init__(self):
#         load_dotenv(dotenv_path="D:/Pyn/Test_work/LLMs/.env")
#         self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
#         self.api_key        = os.getenv("AZURE_OPENAI_API_KEY")
#         self.api_version    = os.getenv("AZURE_OPENAI_API_VERSION")
#         self.azure_model    = os.getenv("AZURE_OAI_MODEL")
#         self.llm = AzureChatOpenAI(
#             openai_api_key=self.api_key,
#             azure_endpoint=self.azure_endpoint,
#             azure_deployment=self.azure_model,
#             api_version=self.api_version,
#             temperature=0.0
#         )
#
#     def generate_question(self, user_input):
#
#         question_template = promptemplate.question_ask(user_input)
#
#         prompt = PromptTemplate(
#             input_variables=["user_input"],
#             template=question_template
#         )
#         asking_chain = LLMChain(
#             llm=self.llm,
#             prompt=prompt,
#             verbose=False
#         )
#         asking_result = asking_chain.run(user_input=user_input)
#         return asking_result.strip()
#
#     @staticmethod
#     def get_session_history(session_id: str):
#         if session_id not in store:
#             store[session_id] = ChatMessageHistory()
#         return store[session_id]
#
#     def extract_answer(self, asking_question, user_ans):
#         ans_template = promptemplate.question_answer(asking_question, user_ans)
#
#         prompt = ChatPromptTemplate.from_messages([
#             ("system", ans_template),
#             MessagesPlaceholder(variable_name="history"),
#             ("human", "{input}")
#         ])
#
#         chain = prompt | self.llm
#
#         with_memory = RunnableWithMessageHistory(
#             chain,
#             FormAssistant.get_session_history,  # Correct static method reference
#             input_messages_key="input",
#             history_messages_key="history",
#         )
#
#         config = {"configurable": {"session_id": "user1"}}
#
#         with_memory.invoke({"input": user_ans}, config=config)
#
#         response = with_memory.invoke({"input": asking_question}, config=config)
#         ans_result=response.content
#         return ans_result.strip()


load_dotenv(dotenv_path="D:/Pyn/Test_work/LLMs/.env")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
azure_model = os.getenv("AZURE_OAI_MODEL")

promptemplate=prompTemplate()

# class FormChatBot:
#     def __init__(self):
#         self.questions = {
#             "first_name": "What is your first name?",
#             "last_name": "What is your last name?",
#             "address": "Please provide your address.",
#             "gender": "What is your gender? (Male/Female/Other)",
#             "terms_and_conditions": "Do you agree to the terms and conditions? (Yes/No)"
#         }
#         self.store = {}
#         self.responses = {}
#         self.asked_keys = list(self.questions.keys())
#         self.index = 0
#         self.return_index = None
#
#         self.chat_model = AzureChatOpenAI(
#             openai_api_key=api_key,
#             azure_endpoint=azure_endpoint,
#             azure_deployment=azure_model,
#             api_version=api_version,
#             temperature=0.7
#         )
#
#         question_template = promptemplate.question_ask(asking_question, user_input, questions_list)
#
#         validation_prompt = PromptTemplate( input_variables=["asking_question", "user_input", "questions_list"],template=question_template )
#
#         self.validator = LLMChain(llm=self.chat_model, prompt=validation_prompt, verbose=False)
#
#         self.ans_template = promptemplate.question_answer(asking_question, user_ans)
#
#         extraction_prompt = PromptTemplate( input_variables=["asking_question", "user_input"], template=self.ans_template )
#
#         prompt = ChatPromptTemplate.from_messages([
#             ("system", extraction_prompt.template),  # ✅ Correct - use actual string
#             MessagesPlaceholder(variable_name="history"),
#             ("human", "{input}")
#         ])
#
#         self.chain = prompt | self.chat_model
#
#     def get_session_history(self,session_id: str):
#         if session_id not in self.store:
#             self.store[session_id] = ChatMessageHistory()
#         return self.store[session_id]
#
#     def get_next_question(self):
#         if self.index < len(self.asked_keys):
#             return self.questions[self.asked_keys[self.index]]
#         return None
#
#
#
#     def process_input(self, user_input, user1):
#
#         key = self.asked_keys[self.index]
#         question = self.questions[key]
#
#         validation_result = self.validator.run({
#             "asking_question": question,
#             "user_input": user_input,
#             "questions_list": ", ".join(self.questions.keys())
#         }).strip()
#
#         if validation_result.startswith("CORRECT"):
#             correction_field = validation_result.replace("CORRECT", "").strip()
#
#             if correction_field in self.questions:
#                 if self.return_index is None:
#                     self.return_index = self.index
#                 self.index = self.asked_keys.index(correction_field)
#                 return f"Okay, let's update {correction_field}. What is your {correction_field.replace('_', ' ')}?"
#
#         print(user_input)
#
#         with_memory = RunnableWithMessageHistory(
#             self.chain,
#             self.get_session_history,
#             input_messages_key="input",
#             history_messages_key="history",
#         )
#
#         config = {"configurable": {"session_id": f"{user1}"}}
#
#         response = with_memory.invoke({
#             "input": question,
#             "asking_question": question,
#             "user_input": user_input
#         }, config=config)
#
#         ans_result = response.content
#         extractor_result = ans_result.strip()
#
#         print(extractor_result)
#
#         self.responses[key] = extractor_result
#
#         if self.return_index is not None:
#             if self.index == self.return_index:
#                 self.return_index = None
#                 self.index += 1
#             else:
#                 self.index = self.return_index
#         else:
#             self.index += 1
#
#         return extractor_result


class FormChatBot:
    def __init__(self):
        self.questions = questions
        self.store = {}
        self.responses = {}
        self.asked_keys = list(self.questions.keys())
        self.index = 0
        self.return_index = None

        self.chat_model = AzureChatOpenAI(
            openai_api_key=api_key,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_model,
            api_version=api_version,
            temperature=0.7
        )

    def get_session_history(self, session_id: str):
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def get_next_question(self):
        if self.index < len(self.asked_keys):
            return self.questions[self.asked_keys[self.index]]
        return None

    def process_input(self, user_input, user1):
        key = self.asked_keys[self.index]
        question = self.questions[key]

        # ✅ Build prompt dynamically here (now variables exist)
        question_template = promptemplate.question_ask(question, user_input, ", ".join(self.questions.keys()))
        validation_prompt = PromptTemplate(
            input_variables=["asking_question", "user_input", "questions_list"],
            template=question_template
        )
        self.validator = LLMChain(llm=self.chat_model, prompt=validation_prompt, verbose=False)

        validation_result = self.validator.run({
            "asking_question": question,
            "user_input": user_input,
            "questions_list": ", ".join(self.questions.keys())
        }).strip()

        if validation_result.startswith("CORRECT"):
            correction_field = validation_result.replace("CORRECT", "").strip()
            if correction_field in self.questions:
                if self.return_index is None:
                    self.return_index = self.index
                self.index = self.asked_keys.index(correction_field)
                return f"Okay, let's update {correction_field}. What is your {correction_field.replace('_', ' ')}?"

        # ✅ Now create answer extraction prompt
        ans_template = promptemplate.question_answer(question, user_input)
        extraction_prompt = PromptTemplate(
            input_variables=["asking_question", "user_input"],
            template=ans_template
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", extraction_prompt.template),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        self.chain = prompt | self.chat_model

        with_memory = RunnableWithMessageHistory(
            self.chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        config = {"configurable": {"session_id": f"{user1}"}}

        response = with_memory.invoke({
            "input": question,
            "asking_question": question,
            "user_input": user_input
        }, config=config)

        extractor_result = response.content.strip()
        self.responses[key] = extractor_result

        if self.return_index is not None:
            if self.index == self.return_index:
                self.return_index = None
                self.index += 1
            else:
                self.index = self.return_index
        else:
            self.index += 1

        return extractor_result
