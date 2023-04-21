import json
import re
from typing import Any, Dict, Union, List
from pydantic import BaseModel
from jsonschema import validate, ValidationError
from langchain.llms.base import BaseLLM
import contextlib
from marvin import ai_fn


class LLMJsonOutputParserException(Exception):
    """Exception for JSON parsing errors"""
    pass


class ParseJsonException(LLMJsonOutputParserException):
    """Exception for JSON parsing errors"""
    pass


class ValidateJsonException(LLMJsonOutputParserException):
    """Exception for JSON validating errors"""
    pass


class FixJsonException(LLMJsonOutputParserException):
    """Exception for JSON fixing errors"""
    pass


@ai_fn()
def auto_fix_json(json_str: str, schema: str) -> str:
    """
    Fixes the provided JSON string to make it parseable and fully complient with the provided schema.
    If an object or field specified in the schema isn't contained within the correct JSON,
    it is ommited.\n This function is brilliant at guessing  when the format is incorrect.

    Parameters:
    description: str
        The description of the function
    function: str
        The function to run

    Returns:
    str
        The fixed JSON string it is valid.
    """


class LLMJsonOutputParser(BaseModel):
    """Parse the output of the LLM."""
    @classmethod
    def parse_and_validate(cls, json_str: str, json_schema: str, llm: BaseLLM) -> Union[str, Dict[Any, Any]]:
        """
        Parses and validates the JSON string.
        """
        # Parse JSON
        try:
            json_str = cls._parse_json(json_str, json_schema, llm)
        except ParseJsonException as e:
            raise ParseJsonException(str(e))

        # Validate JSON
        try:
            return cls._validate_json(json_str, json_schema, llm)
        except ValidationError as e:
            raise ValidateJsonException(str(e))

    @classmethod
    def _remove_square_brackets(cls, json_str: str) -> str:
        """
        Removes square brackets from the JSON string.
        """
        return re.sub(r"\[|\]", "", json_str)

    @classmethod
    def _parse_json(cls, json_str: str,  json_schema: str, llm: BaseLLM) -> Union[str, Dict[Any, Any]]:
        """
        Parses the JSON string.
        """
        with contextlib.suppress(json.JSONDecodeError):
            json_str = json_str.replace("\t", "")
            return json.loads(json_str)

        with contextlib.suppress(json.JSONDecodeError):
            json_str = cls.correct_json(json_str)
            return json.loads(json_str)

        try:
            json_str = cls._remove_square_brackets(json_str)
            brace_index = json_str.index("{")
            maybe_fixed_json = json_str[brace_index:]
            last_brace_index = maybe_fixed_json.rindex("}")
            maybe_fixed_json = maybe_fixed_json[: last_brace_index + 1]
            return json.loads(maybe_fixed_json)
        except (json.JSONDecodeError, ValueError):
            pass
        # Now try to fix this up using the ai_functions
        try:
            ai_fixed_json = cls._fix_json(json_str, json_schema, llm)
            return json.loads(ai_fixed_json)
        except FixJsonException as e:
            raise ParseJsonException("Could not parse JSON:" + str(e))

    @classmethod
    def _validate_json(cls, json_obj: Union[str, Dict[Any, Any]], json_schema: str, llm: BaseLLM) -> Union[str, Dict[Any, Any]]:
        """
        Check if the given JSON string is fully complient with the provided schema.
        """
        schema_obj = json.loads(json_schema)
        try:
            validate(json_obj, schema_obj)
            return json_obj
        except ValidationError:
            # Now try to fix this up using the ai_functions
            try:
                ai_fixed_json = cls._fix_json(json.dumps(json_obj), json_schema, llm)
                return json.loads(ai_fixed_json)
            except FixJsonException as e:
                raise ValidateJsonException("Could not validate JSON:" + str(e))

    @staticmethod
    def _fix_json(json_str: str, schema: str, llm: BaseLLM) -> str:
        """
        Fix the given JSON string to make it parseable and fully complient with the provided schema.
        """
        try:
            fixed_json_str = auto_fix_json(json_str, schema)
        except Exception as e:
            raise FixJsonException(e)
        try:
            json.loads(fixed_json_str)
            return fixed_json_str
        except Exception:
            import traceback
            call_stack = traceback.format_exc()
            raise FixJsonException(f"Failed to fix JSON: '{json_str}' " + call_stack)

    @staticmethod
    def _extract_char_position(error_message: str) -> int:
        """
        Extract the character position from the error message.
        """
        char_pattern = re.compile(r'\(char (\d+)\)')
        if match := char_pattern.search(error_message):
            return int(match[1])
        else:
            raise ValueError("Character position not found in the error message.")

    @staticmethod
    def _add_quotes_to_property_names(json_string: str) -> str:
        """
        Add quotes to the property names in the JSON string.
        """
        def replace_func(match):
            return f'"{match.group(1)}":'

        property_name_pattern = re.compile(r'(\w+):')
        corrected_json_string = property_name_pattern.sub(
            replace_func,
            json_string)

        try:
            json.loads(corrected_json_string)
            return corrected_json_string
        except json.JSONDecodeError as e:
            raise e

    @staticmethod
    def _balance_braces(json_string: str) -> str:
        """
        Add missing braces to the end of the JSON string.
        """
        open_braces_count = json_string.count("{")
        close_braces_count = json_string.count("}")

        while open_braces_count > close_braces_count:
            json_string += "}"
            close_braces_count += 1

        while close_braces_count > open_braces_count:
            json_string = json_string.rstrip("}")
            close_braces_count -= 1

        with contextlib.suppress(json.JSONDecodeError):
            json.loads(json_string)
            return json_string

    @classmethod
    def _fix_invalid_escape(cls, json_str: str, error_message: str) -> str:
        """
        Remove the invalid escape character from the JSON string.
        """
        while error_message.startswith('Invalid \\escape'):
            bad_escape_location = cls._extract_char_position(error_message)
            json_str = json_str[:bad_escape_location] + \
                json_str[bad_escape_location + 1:]
            try:
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError as e:
                error_message = str(e)
        return json_str

    @classmethod
    def correct_json(cls, json_str: str) -> str:
        """
        Correct the given JSON string to make it parseable.
        """
        try:
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError as e:
            error_message = str(e)
            if error_message.startswith('Invalid \\escape'):
                json_str = cls._fix_invalid_escape(json_str, error_message)
            if error_message.startswith('Expecting property name enclosed in double quotes'):
                json_str = cls._add_quotes_to_property_names(json_str)
                try:
                    json.loads(json_str)
                    return json_str
                except json.JSONDecodeError as e:
                    error_message = str(e)
            if balanced_str := cls._balance_braces(json_str):
                return balanced_str
        return json_str
