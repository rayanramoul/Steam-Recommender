from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector



class SteamDB:
    def __init__(self, path:str):
        self.path = path
        self.chroma = Chroma()
        self.selector = SemanticSimilarityExampleSelector(self.chroma)

    def read_dataset(self):
        with open(self.path, 'r') as f:
            data = f.readlines()
        return data






