import numpy as np
from core.conversation import Conversation
from core.interaction import Interaction
from engine.scaler import Scaler
from models.base_model import BaseModel
from utils.serialization import Serialization
from utils.logger import log


class TextSimilarity:
    """Compute cosine similarity score between a given query and a list of paragraphs (using Gecko embeddings)"""

    def __init__(self, scaler: Scaler, embedding_model: BaseModel):
        self.scaler = scaler
        self.model = embedding_model

    def embed_paragraphs(self, paragraphs: list[str], trace: int = 0) -> list[list[float]]:
        interactions = []
        for p in paragraphs:
            conversation = Conversation.from_text(p)
            interaction = Interaction(conversation, self.model, trace)
            interaction.prompt = conversation
            interactions.append(interaction)
        self.scaler.run_batch(interactions, trace)
        return [Serialization.b85_to_float_list(i.output.to_text()) for i in interactions]

    def embed_paragraph(self, paragraph: str, trace: int = 0) -> list[float]:
        return self.embed_paragraphs([paragraph], trace)[0]

    # Return max and average similarity scores relative to the given paragraphs
    def score(self, query: str, paragraphs: list[str], max_length_chars: int = 0, trace: int = 0) -> tuple[float, float]:
        if max_length_chars:
            query = query[:max_length_chars]
            paragraphs = [p[:max_length_chars] for p in paragraphs]
        query_embedding = self.embed_paragraph(query, trace)
        paragraph_embeddings = self.embed_paragraphs(paragraphs, trace)
        # Calculate the cosine similarity between the user query embedding and the dataframe embedding
        score_list = []
        for emb in paragraph_embeddings:
            cosine_score = np.dot(emb, query_embedding)
            score_list.append(abs(cosine_score))
        log(f"score_list={score_list}", trace)
        return max(score_list), (sum(score_list) / len(score_list))
