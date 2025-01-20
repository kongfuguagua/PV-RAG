from sentence_transformers import SentenceTransformer
import torch

class EmbeddingModel:
    def __init__(self, model: str = "/home/ubuntu/llm/models/bge-large-en-v1.5",device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.N_GPU = torch.cuda.device_count()
        self.DEVICE = device
        self.encoder = SentenceTransformer(model, device=self.DEVICE)
        self.embedding_cache = {}
        self.EMBEDDING_DIM = self.encoder.get_sentence_embedding_dimension()
        self.MAX_SEQ_LENGTH_IN_TOKENS = self.encoder.get_max_seq_length()
        self.model_name=model

    def info(self):
        return {
            "N_GPU": self.N_GPU,
            "DEVICE": str(self.DEVICE),
            "model_name": self.model_name,
            "EMBEDDING_DIM": self.EMBEDDING_DIM,
            "MAX_SEQ_LENGTH_IN_TOKENS": self.MAX_SEQ_LENGTH_IN_TOKENS
        }

    def emb_text(self, text: str):
        if text in self.embedding_cache:
            return self.embedding_cache[text],True
        else:
            # if len(text)>=self.MAX_SEQ_LENGTH_IN_TOKENS:
            #     return torch.tensor(0),False
            embedding=self.encoder.encode(text)
            self.embedding_cache[text] = embedding
            return embedding,True
        
    def get_token(self,text):
        tokens=self.encoder.tokenize(text)
        return tokens

    def emb_long_text(self,longtext:str):
        pass