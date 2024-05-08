from langchain.vectorstores import Chroma
import pandas as pd
from loguru import logger
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser



class SteamDB:
    def __init__(self, path:str):
        self.path = path
        self.chroma = Chroma()
        self.text_splitter = self.get_text_splitter()
        self.embedding_model = self.get_embedding_model()
        self.df = self.read_dataset(path)
        self.format_dataset(self.df)
        logger.debug(self.df)
        logger.debug(f'Columns: {self.df.columns}')
        self.dataset_to_embedding_db()
        logger.debug("Steam DB loaded")

    def read_dataset(self, path:str):
        return pd.read_csv(path)

    def format_dataset(self, df:pd.DataFrame):
        self.df['text'] = self.df['Name'] + ' ' + self.df['About the game']

    def get_text_splitter(self, chunk_size:int = 500, chunk_overlap:int = 10):
        return CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def load_dataset_in_chroma(self, df:pd.DataFrame):
        loader = DataFrameLoader(df)
        logger.debug("Dataset loaded in Chroma")
        return loader.load()

    def get_embedding_model(self, model_name:str = "all-MiniLM-L6-v2"):
        if model_name == "all-MiniLM-L6-v2":
            from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
            return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        logger.error(f"Model '{model_name}' not found")
        return None

    def dataset_to_embedding_db(self, model_name:str = "all-MiniLM-L6-v2"):
        self.load_dataset_in_chroma(self.df)
        emb_model = self.get_embedding_model(model_name)
        loader = self.load_dataset_in_chroma(self.df)
        texts = self.text_splitter.split_documents(loader)
        chromadb_index = Chroma.from_documents(texts, emb_model, persist_directory="data/chroma")
        retriever = chromadb_index.as_retriever()
        logger.debug("Dataset embedded in Chroma")

    def search_similar(self, query:str, n:int = 5):
        return self.chroma.search(query, n)

    def text_generation(self):
        model_id = "databricks/dolly-v2-3b" #my favourite textgeneration model for testing
        task="text-generation"

