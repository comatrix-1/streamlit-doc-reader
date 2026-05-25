import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Chat PDF", page_icon="📄", layout="wide")
st.title("📄 Chat with any PDF using Python & Gemini")


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks


def create_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vector_store = Chroma.from_texts(
        text_chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )


def ask_question(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    new_db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    docs = new_db.similarity_search(user_question)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = PromptTemplate.from_template(
        "Use the following context to answer thh question.\n\nContext:\n{context}\n\nQuestion: {question}"
    )

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    chain = prompt | llm | StrOutputParser()

    response = ""
    with st.spinner("Thinking..."):
        response = chain.invoke({"context": context, "question": user_question})
    st.write("AI Answer: ", response)


with st.sidebar:
    st.title("Menu:")
    pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True)
    if st.button("Submit & Train AI"):
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            create_vector_store(text_chunks)
            st.success("Done! LLM Analyzed your File.")

st.write("Upload your PDFs and press the submit button before asking questions.")
user_question = st.text_input("Ask a question about your documents:")
if user_question:
    ask_question(user_question)
