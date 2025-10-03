import os 
from tqdm import tqdm
import pandas as pd
from loguru import logger
from langchain_chroma import Chroma
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import torch

# Use a smaller, faster embedding model by default
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class SteamDB:
    def __init__(self, path:str):
        self.path = path
        self.embeddings = self.get_embedding_model()
        chroma_dir = os.environ.get("CHROMA_DIR", "data/chroma")
        self.initialize_db(chroma_dir)
        logger.debug(f'Columns: {self.df.columns}')
        logger.debug(f'Number of documents in DB {len(self.vectordb.get())}')

    def read_dataset(self):
        self.df = pd.read_csv(self.path)
        # get only 100 rows
        # self.df = self.df.head(100)

    def read_files_to_chroma(self, chroma_persist_directory):
        """
        Read files to chroma
        """
        self.read_dataset()
        self.format_dataset()
        docs = self.dataset_to_embedding_db()
        # add docs to the db 
        logger.debug(f"Dir of self.vectordb {type(self.vectordb)}:  {dir(self.vectordb)}")
        # loop over the docs and add them to the db with per-chunk unique ids
        for idx, doc in enumerate(tqdm(docs)):
            game_name = str(doc.metadata.get("Name", "unknown"))
            self.vectordb.add_documents([doc], ids=[f"{game_name}:{idx}"])

    def initialize_db(self, chroma_persist_directory):
        """
        Initialize the db
        """
        try:
            os.makedirs(chroma_persist_directory, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create persist directory {chroma_persist_directory}: {e}")
        try:
            self.vectordb = Chroma(
                persist_directory=chroma_persist_directory,
                embedding_function=self.embeddings,
            )
        except Exception as e:
            fallback = os.path.expanduser("~/.cache/steam-recommender/chroma")
            logger.warning(f"Failed to init Chroma at {chroma_persist_directory} ({e}). Falling back to {fallback}.")
            os.makedirs(fallback, exist_ok=True)
            self.vectordb = Chroma(
                persist_directory=fallback,
                embedding_function=self.embeddings,
            )
        self.read_files_to_chroma(chroma_persist_directory)
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": 3})


    def format_dataset(self):
        # delete rows where game name is already in the chromadb
        logger.debug(f"Vector db get {self.vectordb.get().keys()}")
        metadatas = self.vectordb.get().get('metadatas', [])
        existing_names = set()
        for meta in metadatas:
            try:
                if isinstance(meta, dict) and 'Name' in meta:
                    existing_names.add(meta['Name'])
            except Exception:
                continue
        logger.debug(f"Existing Names in vector db {list(existing_names)[:10]}")
        self.df = self.df[~self.df['Name'].isin(existing_names)]
        # drop nan values where name or about the game is nan
        self.df = self.df.dropna(subset=['Name', 'About the game'])
        self.df['text'] = self.df['Name'] + ' ' + self.df['About the game']

    def get_list_docs(self):
        """
        Get list of all documents
        """
        filenames = [doc.metadata["filename"] for doc in self.vectordb.get()]
        return filenames

    def get_text_splitter(self, chunk_size:int = 1000, chunk_overlap:int = 200):
        return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def load_dataset_in_chroma(self, df:pd.DataFrame):
        # DataFrameLoader expects a text column; ensure it exists
        if 'text' not in df.columns:
            if 'Name' in df.columns and 'About the game' in df.columns:
                df = df.copy()
                df['text'] = df['Name'] + ' ' + df['About the game']
            else:
                raise ValueError("Input CSV must contain 'Name' and 'About the game' columns")
        loader = DataFrameLoader(df, page_content_column='text')
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
        # Use the vector store directly to control k per-call
        docs = self.vectordb.similarity_search(query, k=n)
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
                    "about": doc.metadata.get("About the game", ""),
                    "score": doc.metadata.get("User score") or doc.metadata.get("Metacritic score"),
                    "genres": doc.metadata.get("Genres", ""),
                    "image": doc.metadata.get("Header image", ""),
                }
            )
        docs = pd.DataFrame(games)
        return docs

