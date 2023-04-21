from pydantic import BaseModel, Extra
from abc import abstractmethod
from typing import ContextManager


class BaseHumanUserInterface(BaseModel):
    """ Base class for human user interface."""
    class Config:
        extra = Extra.forbid

    @abstractmethod
    def get_user_input(self) -> str:
        # waiting for user input
        pass

    @abstractmethod
    def get_binary_user_input(self, message: str) -> bool:
        # get user permission
        pass

    @abstractmethod
    def notify(self, title: str, message: str) -> None:
        # notify user
        pass

    @abstractmethod
    def loading(self) -> ContextManager:
        # waiting for AI to respond
        pass
