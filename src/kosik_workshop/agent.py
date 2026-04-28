"""LangGraph ReAct agent pro Košík assistenta.

Explicitní graf se dvěma nody (`call_model`, `call_tools`) a podmíněnou hranou
podle `tool_calls` v poslední AI zprávě. Záměrně bez `create_react_agent` —
pro workshop je didakticky cennější vidět stav, nody a routing.

Tok:

    START → call_model ──tool_calls?──> call_tools ──> call_model ...
                          └── jinak ──> END
"""

from __future__ import annotations

import os
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from kosik_workshop.config import AGENT_MODEL, AGENT_TEMPERATURE
from kosik_workshop.prompts.loader import load_kosik_prompt
from kosik_workshop.tools import ALL_TOOLS


def _extract_system_template(prompt: object) -> str:
    """Vytáhne system prompt template z LangSmith ChatPromptTemplate.

    LangSmith hub vrací `ChatPromptTemplate` se seznamem `messages`. System
    template je první zpráva s neprázdným `prompt.template`. Při nečekané
    struktuře vyhoď srozumitelnou chybu místo `StopIteration`.
    """
    messages = getattr(prompt, "messages", None)
    if not messages:
        raise TypeError(f"Prompt nemá atribut `messages` nebo je prázdný: {type(prompt).__name__}")
    for msg in messages:
        inner = getattr(msg, "prompt", None)
        template = getattr(inner, "template", None) if inner is not None else None
        if isinstance(template, str) and template.strip():
            return template
    raise TypeError(
        "V LangSmith promptu nebyla nalezena žádná zpráva s neprázdným `prompt.template`."
    )


class AgentState(TypedDict):
    """Stav grafu: konverzace jako append-only list zpráv.

    `add_messages` reducer řeší slučování (deduplikace podle id, append jinak),
    takže jednotlivé nody jen vracejí nové zprávy a runtime se postará o merge.
    """

    messages: Annotated[list[AnyMessage], add_messages]


def build_agent(
    model: str = AGENT_MODEL,
    temperature: float = AGENT_TEMPERATURE,
    checkpointer: BaseCheckpointSaver | None = None,
    system_text: str | None = None,
):
    """Build a LangGraph ReAct agent with the prompt pulled from LangSmith Hub.

    The prompt is pulled according to `ENVIRONMENT` (dev/prod) or the
    `KOSIK_PROMPT_COMMIT` pin. Returns a compiled graph — invoke it with
    `.invoke({"messages": [HumanMessage(...)]}, config={"configurable": {"thread_id": ...}})`.

    A `MemorySaver` checkpointer is used by default so that calls sharing the
    same `thread_id` accumulate conversation state. Pass a different checkpointer
    (e.g. `SqliteSaver`) for persistence beyond the process lifetime.
    """
    if system_text is None:
        prompt = load_kosik_prompt()
        system_text = _extract_system_template(prompt)
    system_message = SystemMessage(content=system_text)
    llm = ChatOpenAI(model=model, temperature=temperature).bind_tools(ALL_TOOLS)

    def call_model(state: AgentState) -> dict:
        response = llm.invoke([system_message, *state["messages"]])
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    graph = StateGraph(AgentState)
    graph.add_node("call_model", call_model)
    graph.add_node("tools", ToolNode(ALL_TOOLS))
    graph.add_edge(START, "call_model")
    graph.add_conditional_edges("call_model", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "call_model")
    compiled = graph.compile(
        checkpointer=checkpointer or MemorySaver(),
        name="kosik-agent",
    )

    env = os.getenv("ENVIRONMENT", "development").strip().lower()
    env_tag = "prod" if env in ("production", "prod") else "dev"
    base_tags = [f"env:{env_tag}", f"model:{model}"]
    return compiled.with_config({"tags": base_tags, "metadata": {"env": env_tag, "model": model}})


def studio_graph():
    """Zero-arg factory entrypoint for `langgraph dev` / LangSmith Studio.

    The Studio CLI requires factories with 0/1/2 args, so this wraps `build_agent`
    with sensible defaults (no checkpointer — the runtime injects its own).
    """
    return build_agent()
