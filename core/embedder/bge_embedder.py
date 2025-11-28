from FlagEmbedding import BGEM3FlagModel
from embedder.base_embedder import BaseEmbedder
import os

class BGEEmbedder(BaseEmbedder):
    def __init__(self, model_name: str= "BAAI/bge-m3", device: str = "cpu", local_dir: str = "./models/bge-m3",):

        if os.path.isdir(local_dir):
            model_path = local_dir

        self.model = BGEM3FlagModel(
            model_path,
            use_fp16=True,
            device=device
        )
    
    def encode_corpus(self, texts, batch_size, max_length):
        emb = self.model.encode(
            texts,
            batch_size = batch_size,
            max_length = max_length
        )
        return emb['dense_vecs']  # dim (N, 1024) 
    
    def encode_query(self, query, max_length = 256):
        emb = self.model.encode(
            [query],
            batch_size=1,
            max_length=max_length,
        )
        return emb["dense_vecs"][0]  