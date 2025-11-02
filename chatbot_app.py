import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain



# Set environment variable or do this securely in production
os.environ["GOOGLE_API_KEY"] = "Enter Your API key"
import asyncio

try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Embedding model
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Vector store and retriever
retriever = Chroma(
    embedding_function=embedding,
    persist_directory="chroma_storage",  # match the dir used earlier

    
).as_retriever(search_kwargs={"k": 2})

# LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

# Custom chat prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant for Adithya Institute of Technology. Use the following context to answer the question. "
                "Answer with the formal replies if the questions is of type formal greetings or message"
               "If the question is unrelated to Adithya Institute of Technology, respond with:\n"
               "\"I'm sorry, I can only answer questions about Adithya Institute of Technology.\""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    ("system", "Context:\n{context}")
])

# Create document chain
doc_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

# Create RAG chain
rag_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=doc_chain
)
print(">>>>>>>>>>>>>", rag_chain)

# Initialize memory
memory = ConversationBufferMemory(return_messages=True)

# Streamlit app
st.title("📘 Adithya College Chatbot")
st.markdown("Ask me anything about the college documents.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# Chat input
if question := st.chat_input("Ask a question about Adithya College"):
    if not question.strip():
        st.warning("Please enter a valid question.")
        st.stop()

    # Display user message
    st.chat_message("user").markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Convert chat history to LangChain-compatible message objects
    chat_history = [
        HumanMessage(msg["content"]) if msg["role"] == "user" else AIMessage(msg["content"])
        for msg in st.session_state.messages[:-1]  # exclude current user message
    ]

    # question = "What is Adithya Institute of Technology?"
    retrieved_docs = retriever.invoke(question)

    print("🔍 Retrieved Docs:", retrieved_docs)
    for i, doc in enumerate(retrieved_docs):
        print(f"\n--- Document {i+1} ---\n{doc.page_content}")

    # Invoke RAG chain with correct inputs
    response = rag_chain.invoke({
        "input": question,
        "chat_history": chat_history
    })

    # Show assistant reply
    st.chat_message("assistant").markdown(response["answer"])
    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

# # Load the vector database
#EMBEDDING_DIR = "adithya_chroma_db"
# embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


# retriever = Chroma(
#     collection_name="adithya_chroma_db",
#     embedding_function=embedding,
#     persist_directory="./chroma_db"
# ).as_retriever()

# # Retriever

# # Language model
# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)



# # Prompt Template
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are an assistant for Adithya Institute of Technology. Use the following context to answer the question. \
# If the question is unrelated to Adithya Institute of Technology, respond with: \
# I'm sorry, I can only answer questions about Adithya Institute of Technology."
# ),
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("human", "{question}"),
#     ("system", "Context:\n{context}")
# ])




# # Memory for chat history
# memory = ConversationBufferMemory(return_messages=True)

# # # Define RAG chain manually (without deprecated ConversationalRetrievalChain)
# # rag_chain = RetrievalQA.from_chain_type(
# #     llm=llm,
# #     retriever=retriever,
# #     return_source_documents=True,
# # )
# question_answer_chain = create_stuff_documents_chain(llm, prompt)
# chain = create_retrieval_chain(retriever, question_answer_chain)



# # Streamlit UI
# st.title("📘 Adithya College Chatbot")
# st.markdown("Ask me anything about the documents!")

# # Chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display messages
# for msg in st.session_state.messages:
#     role = "user" if msg["role"] == "user" else "assistant"
#     st.chat_message(role).markdown(msg["content"])

# # Input
# if question := st.chat_input("Ask a question"):
#     if not question.strip():
#         st.warning("Please enter a valid question.")
#         st.stop()

#     st.chat_message("user").markdown(question)
#     st.session_state.messages.append({"role": "user", "content": question})
#     print(">>>>>>>>>>>>>", question)
#     # Get response
#     # Prepare inputs for the chain
#     chat_history = memory.chat_memory.messages  # Gets messages in correct format

#     # You may need to clean the history if it's not formatted properly
#     # This is optional and depends on your memory setup
#     history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])

#     response = chain.invoke({
#         "question": question,
#         "chat_history": history_str  # or just `chat_history` if chain accepts messages
#     })

#     st.write(response["result"])

#     st.chat_message("assistant").markdown(response)
#     st.session_state.messages.append({"role": "assistant", "content": response})
