import os 
from tqdm import tqdm

from langchain.vectorstores import Chroma
import pandas as pd
from loguru import logger
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain.storage import InMemoryStore
from huggingface_hub import hf_hub_download
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp, OpenAI
from langchain.retrievers import ParentDocumentRetriever

from langchain.schema.document import Document
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch

EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

class SteamDB:
    def __init__(self, path:str):
        self.path = path
        self.embeddings = self.get_embedding_model()
        self.store = InMemoryStore()
        self.initialize_db("data/chroma")
        logger.debug(f'Columns: {self.df.columns}')
        logger.debug(f'Number of documents in DB {len(self.vectordb.get())}')

    def read_dataset(self):
        self.df = pd.read_csv(self.path)
        # get only 100 rows
        self.df = self.df.head(100)

    def read_files_to_chroma(self, chroma_persist_directory):
        """
        Read files to chroma
        """
        self.read_dataset()
        self.format_dataset()
        docs = self.dataset_to_embedding_db()
        # add docs to the db 
        logger.debug(f"Dir of self.vectordb {type(self.vectordb)}:  {dir(self.vectordb)}")
        # loop over the docs and add them to the db
        for doc in tqdm(docs):
            self.vectordb.add_documents([doc])

    def initialize_db(self, chroma_persist_directory):
        """
        Initialize the db
        """

        self.vectordb = Chroma(
            persist_directory=chroma_persist_directory,
            embedding_function=self.embeddings,
        )
        self.read_files_to_chroma(chroma_persist_directory)
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": 3})


    def format_dataset(self):
        self.df['text'] = self.df['Name'] + ' ' + self.df['About the game']

    def get_list_docs(self):
        """
        Get list of all documents
        """
        filenames = [doc.metadata["filename"] for doc in self.vectordb.get()]
        return filenames

    def get_text_splitter(self, chunk_size:int = 1000, chunk_overlap:int = 10):
        return CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def load_dataset_in_chroma(self, df:pd.DataFrame):
        loader = DataFrameLoader(df)
        logger.debug("Dataset loaded in Chroma")
        return loader.load()

    def get_embedding_model(self, model_name:str = "all-MiniLM-L6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"\n\nUsing device {self.device }\n\n")
        model_kwargs = {"device": self.device}
        return  HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME, model_kwargs=model_kwargs
        )

    def dataset_to_embedding_db(self, model_name:str = "all-MiniLM-L6-v2"):
        self.loader = self.load_dataset_in_chroma(self.df)
        self.text_splitter = self.get_text_splitter()
        return self.text_splitter.split_documents(self.loader)

    def search_similar(self, query:str, n:int = 5):
        logger.debug(f"Searching for query: {query}")
        docs = self.retriever.get_relevant_documents(query)
        logger.debug(f"Found {len(docs)} relevant documents")
        games = []
        for i in range(min(n, len(docs))):
            doc = docs[i]
            logger.debug(f"\n\nDocument {i}: {docs[i]}")
            # logger.debug(f"Page content {i}: {self.df.iloc[docs[i]]['text']}")
            logger.debug(f"Doc metadata keys {doc.metadata.keys()}")
            games.append(
                {
                    "game": doc.metadata["Name"],
                    "about": doc.metadata["About the game"],
                    "score": doc.metadata["User score"],
                    "genres": doc.metadata["Genres"],
                }
            )
        docs = pd.DataFrame(games)
        return docs

