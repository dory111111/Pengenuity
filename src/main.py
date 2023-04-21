import os
from dotenv import load_dotenv
from agent import Agent
from tools.base import AgentTool
from ui.cui import CommandlineUserInterface
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI


# Set API Keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
assert OPENAI_API_KEY, "OPENAI_API_KEY environment variable is missing from .env"
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "")
assert GOOGLE_CSE_ID, "GOOGLE_CSE_ID environment variable is missing from .env"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Set Agent Settings
AGENT_NAME = os.getenv("AGENT_NAME", "")
assert AGENT_NAME, "AGENT_NAME variable is missing from .env"
AGENT_ROLE = os.getenv("AGENT_ROLE", "")
assert AGENT_ROLE, "AGENT_ROLE variable is missing from .env"
AGENT_OBJECTIVE = os.getenv("AGENT_OBJECTIVE", "")
assert AGENT_OBJECTIVE, "AGENT_OBJECTIVE variable is missing from .env"
AGENT_DIRECTORY = os.getenv("AGENT_DIRECTORY", "")
assert AGENT_DIRECTORY, "AGENT_DIRECTORY variable is missing from .env"

llm = OpenAI(temperature=0.0)
openaichat = ChatOpenAI(temperature=0.0)  # Optional

### 1.Create Agent ###
dir = AGENT_DIRECTORY

agent = Agent(
    name=AGENT_NAME,
    role=AGENT_ROLE,
    goal=AGENT_OBJECTIVE,
    ui=CommandlineUserInterface(),
    openai_api_key=OPENAI_API_KEY,
    llm=llm,
    openaichat=openaichat,
    dir=dir
)

### 2. Set up tools for agent ###
search = GoogleSearchAPIWrapper()
search_tool = AgentTool(
    name="google_search",
    func=search.run,
    description="""
        "With this tool, you can search the web using Google search engine"
        "It is a great way to quickly find information on the web.""",
    user_permission_required=False
)

### 3. Momoize usage of tools to agent ###
agent.prodedural_memory.memorize_tools([search_tool])

### 4.Run agent ###
agent.run()
