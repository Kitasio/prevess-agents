from typing import Any, List, Union
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain_community.tools.convert_to_openai import format_tool_to_openai_tool
from langchain.agents import AgentExecutor, tool
from langchain.chat_models import ChatOpenAI
from langserve import add_routes
from langserve.pydantic_v1 import BaseModel, Field
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, FunctionMessage, HumanMessage
from langchain.prompts import MessagesPlaceholder
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

app = FastAPI()

llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)
search = TavilySearchResults()
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are very powerful assistant, but bad at calculating lengths of words. "
            "Talk with the user as normal. "
            "If they ask you to calculate the length of a word, use a tool"
            "If they provide information like age, name, etc. use a tool, but you MUST ask for confirmation first",
        ),
        # Please note the ordering of the fields in the prompt!
        # The correct ordering is:
        # 1. history - the past messages between the user and the agent
        # 2. user - the user's current input
        # 3. agent_scratchpad - the agent's working space for thinking and
        #    invoking tools to respond to the user's input.
        # If you change the ordering, the agent will not work correctly since
        # the messages will be shown to the underlying LLM in the wrong order.
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

@tool
def word_length(word: str) -> int:
    """Returns a counter word"""
    return len(word)


@tool
def profile_update(age=None, name=None) -> str:
    """Updates the profile of the user in the database"""
    print(age)
    print(name)
    return "profile updated"


tools = [
    search,
    word_length,
    profile_update,
]
llm_with_tools = llm.bind(tools=[format_tool_to_openai_tool(tool) for tool in tools])

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    # | prompt_trimmer # See comment above.
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

class Input(BaseModel):
    input: str
    # The field extra defines a chat widget.
    # Please see documentation about widgets in the main README.
    # The widget is used in the playground.
    # Keep in mind that playground support for agents is not great at the moment.
    # To get a better experience, you'll need to customize the streaming output
    # for now.
    chat_history: List[Union[HumanMessage, AIMessage, FunctionMessage]] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "input", "output": "output"}},
    )


class Output(BaseModel):
    output: Any


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


# Edit this to add the chain you want to add
add_routes(
    app,
    agent_executor.with_types(input_type=Input, output_type=Output).with_config(
        {"run_name": "agent"}
    ),
    path="/openai",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
