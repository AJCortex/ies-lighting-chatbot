import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pinecone import Pinecone

# --- Load API Keys ---
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

# --- Initialize Pinecone ---
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])

# --- Setup ---
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
index_name = "ies-lighting-handbook-qa"

vectorstore = PineconeVectorStore(
    index_name=index_name,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
llm = ChatOpenAI(model="gpt-4", temperature=0)

# --- Query Rewriter ---
rewrite_prompt = ChatPromptTemplate.from_template("""
You are an IES lighting expert. Rewrite the following question using 
technical IES Lighting Handbook terminology to improve search results.
Include relevant terms like: illuminance, lux, footcandles, horizontal, 
vertical, transition spaces, table references.

Original question: {question}
Rewritten question:
""")

rewrite_chain = rewrite_prompt | llm | StrOutputParser()

# --- Answer Prompt ---
prompt = ChatPromptTemplate.from_template("""
You are an expert on the IES Lighting Handbook 10th Edition. 
Use the following context to answer the question. 
If the context references a table by name, use your knowledge of IES 
lighting standards to provide the specific values from that table.
If you cannot find exact values, provide general IES guidance on the topic.

IMPORTANT: Do not reference document IDs, chunk IDs, or any internal 
identifiers in your response. Only cite table numbers and page numbers 
when referencing sources.

Context: {context}

Question: {question}

Provide specific lux or footcandle values where relevant.
""")

# --- Chain ---
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- UI ---
st.title("IES Lighting Handbook Chatbot")
st.write("Ask any question about the IES Lighting Handbook")

query = st.text_input("Your question:")

if query:
    with st.spinner("Searching the IES Lighting Handbook..."):
        technical_query = rewrite_chain.invoke({"question": query})
        response = chain.invoke(technical_query)
        st.write("**Answer:**")
        st.write(response)