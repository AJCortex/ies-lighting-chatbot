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

prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:
{context}

Question: {question}
""")

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
        # Debug: show retrieved documents
        docs = retriever.invoke(query)
        st.write(f"**Retrieved {len(docs)} documents:**")
        for i, doc in enumerate(docs):
            st.write(f"**Chunk {i+1}:**")
            st.write(doc.page_content[:300])
            st.divider()

        # Answer
        response = chain.invoke(query)
        st.write("**Answer:**")
        st.write(response)