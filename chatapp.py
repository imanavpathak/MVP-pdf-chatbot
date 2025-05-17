import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


st.set_page_config("Multi PDF Chatbot", page_icon = ":scroll:")
load_dotenv()
# os.getenv("")
openai_api_key = os.getenv("Paste your Api Keys Here")
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")



def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the context, just say "answer is not available in the context".

    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3, openai_api_key=openai_api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    st.write("Reply:", response["output_text"])

    st.write("Reply: ", response["output_text"])




def main():
    
    st.header("MVP-PDF's 📚 - Chatbot 🤖 ")

    user_question = st.text_input("Ask any Question to mvp from the PDF Files uploaded .. ✍️📝")

    if user_question:
        user_input(user_question)

    with st.sidebar:

        st.image("img/manav.png")
        st.write("---")
        
        st.title("📁 PDF File's Section")
        pdf_docs = st.file_uploader("Upload your PDF Files & \n Click on the Submit & Process Button ", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."): # user friendly message.
                raw_text = get_pdf_text(pdf_docs) # get the pdf text
                text_chunks = get_text_chunks(raw_text) # get the text chunks
                get_vector_store(text_chunks) # create vector store
                st.success("Done")
        
        st.write("---")
        st.image("img/manav2.jpg")
        st.write("AI App created by @ Manav Pathak")  # add this line to display the image


st.markdown("""
    <style>
    .footer-bottom {
      background-color: #0b0b0b;
      width: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
      height: 80px;
      position: fixed;
      bottom: 0;
      left: 0;
      z-index: 3;
    }

    .footer-bottom article {
      color: rgb(176, 176, 176);
      font-size: 2.5em;
      font-family: "Times New Roman", Times, serif;
      letter-spacing: 1.5px;
      background: linear-gradient(90deg, #000, #fff, #000);
      background-repeat: no-repeat;
      background-size: 80%;
      animation: animate 5s linear infinite;
      -webkit-background-clip: text;
      -webkit-text-fill-color: rgba(255, 255, 255, 0);
      font-family: "Playwrite NG Modern", cursive;
      font-style: normal;
    }

    @keyframes animate {
      0% {
        background-position: -500%;
      }

      100% {
        background-position: 500%;
      }
    }
    </style>

    <div class="footer-bottom">
        <article>Made by Manav Pathak 🚀</article>
    </div>
""", unsafe_allow_html=True)

    

if __name__ == "__main__":
    main()
