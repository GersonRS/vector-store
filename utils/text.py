from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader


def get_pdf_text(pdf):
    # text = ""
    # pdf_reader = PdfReader(pdf)
    # for page in pdf_reader.pages:
    #     text += page.extract_text()
    loader = PyPDFLoader(pdf)
    documents = loader.load()

    return documents


def get_text_chunks(documents):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    # chunks = text_splitter.split_text(text)
    docs = text_splitter.split_documents(documents=documents)
    return docs
