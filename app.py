import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA

# --- Setup ---
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
index_name = "ies-lighting-handbook-qa"

vectorstore = PineconeVectorStore(
    index_name=index_name,
    embedding=embeddings
)

retriever = vectorstore.as_retriever()
llm = ChatOpenAI(model="gpt-4", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# --- UI ---
st.title("IES Lighting Handbook Chatbot")
st.write("Ask any question about the IES Lighting Handbook")

query = st.text_input("Your question:")

if query:
    with st.spinner("Searching..."):
        response = qa_chain.invoke(query)
        st.write(response["result"])