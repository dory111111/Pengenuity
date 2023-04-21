from pydantic import BaseModel, Field
from langchain.vectorstores import FAISS
from langchain.vectorstores import VectorStore
from langchain.schema import Document
from pydantic import BaseModel, Field
from langchain.vectorstores import VectorStore, FAISS
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
from tools.base import AgentTool


class ProcedualMemoryException(Exception):
    pass


class ToolNotFoundException(ProcedualMemoryException):
    pass


class ProcedualMemory(BaseModel):
    tools: List[AgentTool] = Field([], title="hoge")
    embeddings: HuggingFaceEmbeddings = Field(
        HuggingFaceEmbeddings(), title="Embeddings to use for tool retrieval")
    docs: List[Document] = Field([], title="Documents to use for tool retrieval")
    vector_store: VectorStore = Field(
        None, title="Vector store to use for tool retrieval")

    class Config:
        arbitrary_types_allowed = True

    def memorize_tools(self, tools: List[AgentTool]) -> None:
        """Memorize tools and embed them."""
        for tool in tools:
            self.tools.append(tool)
            self.docs = [Document(page_content=t.description, metadata={
                                  "index": i}) for i, t in enumerate(self.tools)]
        self._embed_docs()

    def remember_tool_by_name(self, tool_name: str) -> AgentTool:
        """Remember a tool by name and return it."""
        tool = [tool for tool in self.tools if tool.name.lower() == tool_name.lower()]

        if tool:
            return tool[0]
        else:
            raise ToolNotFoundException(f"Tool {tool_name} not found")

    def remember_relevant_tools(self, query: str) -> List[AgentTool]:
        """Remember relevant tools for a query."""
        retriever = self.vector_store.as_retriever()
        relevant_documents = retriever.get_relevant_documents(query)
        return [self.tools[d.metadata["index"]] for d in relevant_documents]

    def remember_all_tools(self) -> List[AgentTool]:
        """Remember all tools and return them."""
        return self.tools

    def _embed_docs(self) -> None:
        """Embed tools."""
        self.vector_store: FAISS = FAISS.from_documents(
            self.docs, self.embeddings
        )
