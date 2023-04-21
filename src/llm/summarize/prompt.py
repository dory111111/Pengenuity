
from typing import List
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

# Convert the schema object to a string
BASE_TEMPLATE = """
[THOUGHTS]
{thoughts}

[ACTION]
{action}

[RESULT OF ACTION]
{result}

[INSTRUSCTION]
Using above [THOUGHTS], [ACTION], and [RESULT OF ACTION], please summarize the event.

[SUMMARY]
"""


def get_template() -> PromptTemplate:
    template = BASE_TEMPLATE
    prompt_template = PromptTemplate(
        input_variables=["thoughts", "action", "result"], template=template)
    return prompt_template


def get_chat_templatez() -> ChatPromptTemplate:
    messages = []
    messages.append(SystemMessagePromptTemplate.from_template(BASE_TEMPLATE))
    return ChatPromptTemplate.from_messages(messages)
