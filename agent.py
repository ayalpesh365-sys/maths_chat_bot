from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from tools import (
    theory_search_tool,
    taylor_series_tool,
    symbolic_simplify_tool,
    plot_taylor_approximation_tool,
    physics_application_tool,
    practice_question_tool
)

from prompts import SYSTEM_PROMPT

load_dotenv()

LLM_MODEL = "gpt-4o-mini"

model = ChatOpenAI(
    model=LLM_MODEL,
    temperature=0
)

agent = create_agent(
    model=model,
    tools=[
        theory_search_tool,
        taylor_series_tool,
        symbolic_simplify_tool,
        plot_taylor_approximation_tool,
        physics_application_tool,
        practice_question_tool
    ],
    system_prompt=SYSTEM_PROMPT
)


def run_agent(messages) -> str:
    """
    Accepts either:
    1. A plain string question, or
    2. A full message history list for cross-questioning.

    Example history:
    [
        {"role": "user", "content": "Derive Euler formula"},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "Why step 3?"}
    ]
    """
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    result = agent.invoke({
        "messages": messages
    })

    return result["messages"][-1].content
