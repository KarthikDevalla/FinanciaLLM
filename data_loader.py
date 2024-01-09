from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader


data_directory=DirectoryLoader('knowledge/', glob='*.pdf', loader_cls=PyPDFLoader)
docs=data_directory.load()
splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text=splitter.split_documents(docs)

embed=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-V2',model_kwargs={'device':'cpu'})
vector_db=FAISS.from_documents(text, embed)
vector_db.save_local('vector_database/db_elements')


