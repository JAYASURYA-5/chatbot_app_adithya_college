import os
# from langchain.document_loaders import WebBaseLoader
from langchain_community.document_loaders import WikipediaLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import requests
from bs4 import BeautifulSoup
from langchain.docstore.document import Document


# SET YOUR API KEY HERE if not using env
os.environ["GOOGLE_API_KEY"] = "AIzaSyDedYxrt77FBgWPof03oplZ47FKYg6vx0c"

CHROMA_DIR = "chroma_storage"
URLS = ["https://en.wikipedia.org/wiki/Adithya_Institute_of_Technology", "https://en.wikipedia.org/wiki/Adithya_Institute_of_Technology#Location", "https://en.wikipedia.org/wiki/Adithya_Institute_of_Technology#Academics", 
        "https://en.wikipedia.org/wiki/Adithya_Institute_of_Technology#Departments", "https://en.wikipedia.org/wiki/Adithya_Institute_of_Technology#Admission_procedure",
        "https://en.wikipedia.org/wiki/Adithya_Institute_of_Technology#National_level_FIDE_rated_chess_tournament"]

def clean_wikipedia_content(url):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        content_div = soup.find("div", {"id": "bodyContent"})

        if content_div:
            paragraphs = content_div.find_all("p")
            text = "\n".join([p.get_text() for p in paragraphs if p.get_text().strip() != ""])
            print(">>>>>>>>>>>>>>>>>>>", text)
            return Document(page_content=text, metadata={"source": url})
        return None

def scrape_and_embed():
    
    docs = []
    for url in URLS:
        print("urllllllll", url)
        doc = clean_wikipedia_content(url)
        if doc:
            docs.append(doc)


        # # loader = WebBaseLoader(url)
        # docs = WikipediaLoader(query="Adithya Institute of Technology", load_max_docs=5).load()

        # print(">>>>>>>>>>>>", loader)
        # docs = loader.load()
        # print("kkkkkkkkkkk", docs)


    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    vectorstore.persist()
    print("✅ Embeddings saved in Chroma.")

if __name__ == "__main__":
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # retriever = vectorstore.as_retriever()
    retriever = Chroma(
    embedding_function=embedding,
    persist_directory="chroma_storage",  # match the dir used earlier

    
).as_retriever()

    # scrape_and_embed()
    question = "what are the course offered"
    retrieved_docs = retriever.invoke(question)

    for i, doc in enumerate(retrieved_docs):
        print(f"\n--- Doc {i+1} ---\n{doc.page_content[:500]}")