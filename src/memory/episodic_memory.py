from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain.llms.base import BaseLLM
from langchain import LLMChain
from langchain.vectorstores import VectorStore, FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from llm.summarize.prompt import get_template


class Episode(BaseModel):
    thoughts: Dict[str, Any] = Field(..., description="thoughts of the agent")
    action: Dict[str, Any] = Field(..., description="action of the agent")
    result: str = Field(..., description="The plan of the event")
    summary: str = Field("", description="summary of the event")


class EpisodicMemory(BaseModel):
    num_episodes: int = Field(0, description="The number of episodes")
    store: Dict[str, Episode] = Field({}, description="The list of episodes")
    llm: BaseLLM = Field(..., description="llm class for the agent")
    embeddings: HuggingFaceEmbeddings = Field(
        HuggingFaceEmbeddings(), title="Embeddings to use for tool retrieval")
    vector_store: VectorStore = Field(
        None, title="Vector store to use for tool retrieval")

    class Config:
        arbitrary_types_allowed = True

    def memorize_episode(self, episode: Episode) -> None:
        """Memorize an episode."""
        self.num_episodes += 1
        self.store[str(self.num_episodes)] = episode
        self._embed_episode(episode)

    def summarize_and_memorize_episode(self, episode: Episode) -> str:
        """Summarize and memorize an episode."""
        summary = self._summarize(episode.thoughts, episode.action, episode.result)
        episode.summary = summary
        self.memorize_episode(episode)
        return summary

    def _summarize(self, thoughts: Dict[str, Any], action: Dict[str, Any], result: str) -> str:
        """Summarize an episode."""
        prompt = get_template()
        llm_chain = LLMChain(prompt=prompt, llm=self.llm)
        try:
            result = llm_chain.predict(
                thoughts=thoughts,
                action=action,
                result=result
            )
        except Exception as e:
            raise Exception(f"Error: {e}")
        return result

    def remember_all_episode(self) -> List[Episode]:
        """Remember all episodes."""
        return self.store

    def remember_recent_episodes(self, n: int = 5) -> List[Episode]:
        """Remember recent episodes."""
        if not self.store:  # if empty
            return self.store
        n = min(n, len(self.store))
        return list(self.store.values())[-n:]

    def remember_last_episode(self) -> Episode:
        """Remember last episode."""
        if not self.store:
            return None
        return self.store[-1]

    def remember_related_episodes(self, query: str, k: int = 5) -> List[Episode]:
        """Remember related episodes to a query."""
        if self.vector_store is None:
            return []
        relevant_documents = self.vector_store.similarity_search(query, k=k)
        result = []
        for d in relevant_documents:
            episode = Episode(
                thoughts=d.metadata["thoughts"],
                action=d.metadata["action"],
                result=d.metadata["result"],
                summary=d.metadata["summary"]
            )
            result.append(episode)
        return result

    def _embed_episode(self, episode: Episode) -> None:
        """Embed an episode and add it to the vector store."""
        texts = [episode.summary]
        metadatas = [{"index": self.num_episodes,
                      "thoughts": episode.thoughts,
                      "action": episode.action,
                      "result": episode.result,
                      "summary": episode.summary}]
        if self.vector_store is None:
            self.vector_store = FAISS.from_texts(
                texts=texts, embedding=self.embeddings, metadatas=metadatas)
        else:
            self.vector_store.add_texts(texts=texts, metadatas=metadatas)

    def save_local(self, path: str) -> None:
        """Save the vector store locally."""
        self.vector_store.save_local(folder_path=path)

    def load_local(self, path: str) -> None:
        """Load the vector store locally."""
        self.vector_store = FAISS.load_local(
            folder_path=path, embeddings=self.embeddings)
