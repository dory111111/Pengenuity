import json
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from llm.extract_entity.schema import JsonSchema
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
)

# Convert the schema object to a string
JSON_SCHEMA_STR = json.dumps(JsonSchema.schema)

ENTITY_EXTRACTION_TEMPLATE = """
    You are an AI assistant reading a input text and trying to extract entities from it.
    Extract ONLY proper nouns from the input text and return them as a JSON object.
    You should definitely extract all names and places.

    [EXAMPLE]
    INPUT TEXT:
     Apple Computer was founded on April 1, 1976, by Steve Wozniak, Steve Jobs and Ronald Wayne to develop and sell Wozniak's Apple I personal computer. 
     It was incorporated by Jobs and Wozniak as Apple Computer, Inc. in 1977. 
     The company's second computer, the Apple II, became a best seller and one of the first mass-produced microcomputers. 
     Apple Computer went public in 1980 to instant financial success. 
     The company developed computers featuring innovative graphical user interfaces, including the 1984 original Macintosh, announced that year in a critically acclaimed advertisement. 
     By 1985, the high cost of its products, and power struggles between executives, caused problems.
     Wozniak stepped back from Apple Computer amicably and pursued other ventures, while Jobs resigned bitterly and founded NeXT, taking some Apple Computer employees with him.
    RESPONCE:
     {{
        "Apple Computer Company": "a company founded in 1976 by Steve Wozniak, Steve Jobs, and Ronald Wayne to develop and sell personal computers",
        "Steve Wozniak": "an American inventor, electronics engineer, programmer, philanthropist, and technology entrepreneur who co-founded Apple Computer Company with Steve Jobs",
        "Steve Jobs": "an American entrepreneur, business magnate, inventor, and industrial designer who co-founded Apple Computer Company with Steve Wozniak and Ronald Wayne, and later founded NeXT",
        "Ronald Wayne": "an American retired electronics industry worker and co-founder of Apple Computer Company, who left the company after only 12 days"
    }}
    [INPUT TEXT] (for reference only):
    {text}
    """

SCHEMA_TEMPLATE = f"""
    [RULE]
    Your response must be provided exclusively in the JSON format outlined below, without any exceptions. 
    Any additional text, explanations, or apologies outside of the JSON structure will not be accepted. 
    Please ensure the response adheres to the specified format and can be successfully parsed by Python's json.loads function.

    Strictly adhere to this JSON RESPONSE FORMAT for your response.
    Failure to comply with this format will result in an invalid response. 
    Please ensure your output strictly follows RESPONSE FORMAT.

    [JSON RESPONSE FORMAT]
    {JSON_SCHEMA_STR}

    [RESPONSE]""".replace("{", "{{").replace("}", "}}")


def get_template() -> PromptTemplate:
    template = f"{ENTITY_EXTRACTION_TEMPLATE}\n{SCHEMA_TEMPLATE}"
    return PromptTemplate(input_variables=["text"], template=template)


def get_chat_template() -> ChatPromptTemplate:
    messages = []
    messages.append(SystemMessagePromptTemplate.from_template(
        ENTITY_EXTRACTION_TEMPLATE))
    messages.append(SystemMessagePromptTemplate.from_template(SCHEMA_TEMPLATE))
    return ChatPromptTemplate.from_messages(messages)
