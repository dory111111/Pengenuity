import json
from typing import Any, Optional
from pydantic import BaseModel, Field
from langchain.llms.base import BaseLLM
from langchain import LLMChain
from langchain.vectorstores import VectorStore, FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from llm.extract_entity.prompt import get_template, get_chat_template
from llm.extract_entity.schema import JsonSchema as ENTITY_EXTRACTION_SCHEMA
from llm.json_output_parser import LLMJsonOutputParser, LLMJsonOutputParserException

CREATE_JSON_SCHEMA_STR = json.dumps(ENTITY_EXTRACTION_SCHEMA.schema)


class SemanticMemory(BaseModel):
    num_episodes: int = Field(0, description="The number of episodes")
    llm: BaseLLM = Field(..., description="llm class for the agent")
    openaichat: Optional[ChatOpenAI] = Field(
        None, description="ChatOpenAI class for the agent")
    embeddings: HuggingFaceEmbeddings = Field(
        HuggingFaceEmbeddings(), title="Embeddings to use for tool retrieval")
    vector_store: VectorStore = Field(
        None, title="Vector store to use for tool retrieval")

    class Config:
        arbitrary_types_allowed = True

    def extract_entity(self, text: str) -> dict:
        """Extract an entity from a text using the LLM"""
        if self.openaichat:
            # If OpenAI Chat is available, it is used for higher accuracy results.
            propmt = get_chat_template().format_prompt(text=text).to_messages()
            result = self.openaichat(propmt).content
        else:
            # Get the result from the LLM
            llm_chain = LLMChain(prompt=get_template(), llm=self.llm)
            try:
                result = llm_chain.predict(text=text)
            except Exception as e:
                raise Exception(f"Error: {e}")

        # Parse and validate the result
        try:
            result_json_obj = LLMJsonOutputParser.parse_and_validate(
                json_str=result,
                json_schema=CREATE_JSON_SCHEMA_STR,
                llm=self.llm
            )
        except LLMJsonOutputParserException as e:
            raise LLMJsonOutputParserException(str(e))
        else:
            self._embed_knowledge(result_json_obj)
            return result_json_obj

    def remember_related_knowledge(self, query: str, k: int = 5) -> dict:
        """Remember relevant knowledge for a query."""
        if self.vector_store is None:
            return {}
        relevant_documents = self.vector_store.similarity_search(query, k=k)
        return {d.metadata["entity"]: d.metadata["description"] for d in relevant_documents}

    def _embed_knowledge(self, entity: dict[str:Any]) -> None:
        """Embed the knowledge into the vector store."""
        description_list = []
        metadata_list = []

        for entity, description in entity.items():
            description_list.append(description)
            metadata_list.append({"entity": entity, "description": description})

        if self.vector_store is None:
            self.vector_store = FAISS.from_texts(
                texts=description_list,
                metadatas=metadata_list,
                embedding=self.embeddings
            )
        else:
            self.vector_store.add_texts(
                texts=description_list,
                metadatas=metadata_list
            )

    def save_local(self, path: str) -> None:
        """Save the vector store to a local folder."""
        self.vector_store.save_local(folder_path=path)

    def load_local(self, path: str) -> None:
        """Load the vector store from a local folder."""
        self.vector_store = FAISS.load_local(
            folder_path=path, embeddings=self.embeddings)
