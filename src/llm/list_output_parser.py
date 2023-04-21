import re
from typing import List
from pydantic import BaseModel


class LLMListOutputParserException(Exception):
    """Exception for List parsing errors"""
    pass


class ParseListException(LLMListOutputParserException):
    """Exception for List parsing errors"""
    pass


class LLMListOutputParser(BaseModel):
    @classmethod
    def parse(cls, string_list: str, separeted_string=",") -> List[str]:
        """
        Parses the string list and returns a list of strings.
        """
        if not string_list:
            return []

        # Remove square brackets
        string_list = cls._remove_square_brackets(string_list)

        # Split by comma and convert to list
        parsed_list = string_list.split(separeted_string)

        # If the string is not comma-separated, raise ValueError
        if len(parsed_list) == 1 and not parsed_list[0]:
            raise ParseListException(f"The string is not {separeted_string}-separated.")

        return parsed_list

    @staticmethod
    def _remove_square_brackets(string_list: str) -> str:
        """
        Removes square brackets from the string.
        """
        return re.sub(r"\[|\]", "", string_list)
