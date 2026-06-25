

import os
import re
import json
from typing import Optional, Annotated
from typing_extensions import TypedDict

from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage,
)
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

import mlflow

from app.tools import TOOLS

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
# Default to the most capable Opus model; override via the ANTHROPIC_MODEL env
# var. For a cheaper/faster agent loop (closer to the original Bedrock Haiku),
# set ANTHROPIC_MODEL=claude-haiku-4-5.
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-8")
MAX_ITERATIONS = 3          # max self-reflection loops before we stop
CONFIDENCE_THRESHOLD = 6    # >= this ends the graph; < this triggers a re-loop

SYSTEM_PROMPT = """You are an industrial maintenance diagnostic assistant for a \
predictive-maintenance copilot. For every diagnosis you MUST work in this order:

(a) FIRST call check_equipment_anomaly on the equipment to read its current \
anomaly signal, RUL estimate and severity. You may also call get_sensor_trend \
to see whether readings are degrading or stable.
(b) If a maintenance-log entry is provided, call classify_fault to identify the \
fault type.
(c) Call retrieve_similar_incidents to ground your diagnosis in real historical \
events and their resolutions. Prefer filtering by the classified fault_category.
(d) NEVER fabricate a resolution. If no sufficiently similar past incident is \
found, say so explicitly and recommend what an engineer should inspect instead.

When you have gathered enough evidence, write a concise diagnosis: the likely \
fault, the supporting evidence (anomaly/sensor/classification/retrieved cases), \
and a recommended resolution grounded in the retrieved incidents."""

REFLECT_PROMPT = """You are reviewing your own diagnostic hypothesis. Score your \
confidence from 1-10. List any evidence gaps. If confidence is below 6, specify \
what additional information would help."""


# --------------------------------------------------------------------------- #
# Graph state
# --------------------------------------------------------------------------- #
class AgentState(TypedDict):
    # The running conversation. add_messages appends new messages instead of
    # overwriting, which is what makes the tool/agent loop accumulate context.
    messages: Annotated[list, add_messages]
    equipment_id: str
    log_entry: Optional[str]
    iterations: int            # how many reflect loops have completed
    confidence_score: int      # latest self-assessed confidence (1-10)
    evidence_gaps: str         # gaps from the last reflection
    final_answer: str          # latest draft diagnosis


# --------------------------------------------------------------------------- #
# Bedrock models (one with tools for the agent, one plain for reflection)
# --------------------------------------------------------------------------- #
def _make_llm():
    # Reads ANTHROPIC_API_KEY from the environment. We deliberately do NOT set
    # temperature/top_p: Opus 4.7/4.8 reject sampling params with a 400. If you
    # switch ANTHROPIC_MODEL to an older model and want determinism, add it back.
    return ChatAnthropic(
        model=ANTHROPIC_MODEL,
        max_tokens=1024,
        timeout=60,
        max_retries=2,
    )


_LLM = _make_llm()
_LLM_WITH_TOOLS = _LLM.bind_tools(TOOLS)


def _message_text(msg: BaseMessage) -> str:
    """Extract plain text from a message whose content may be a list of blocks.

    The Bedrock Converse API can return content as a list of typed blocks
    (e.g. [{'type': 'text', 'text': '...'}]) rather than a bare string.
    """
    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(parts).strip()
    return str(content)


def _parse_confidence(text: str) -> int:
    """Pull a 1-10 confidence score out of the reflection text.

    Tries 'X/10' first, then 'confidence ... N', then any standalone 1-10.
    Falls back to 5 (neutral) if nothing parses, so the loop can still proceed.
    """
    m = re.search(r"\b(10|[1-9])\s*/\s*10\b", text)
    if m:
        return int(m.group(1))
    m = re.search(r"confidence[^0-9]{0,20}(10|[1-9])\b", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(10|[1-9])\b", text)
    if m:
        return int(m.group(1))
    return 5


# --------------------------------------------------------------------------- #
# Node 1 — classify_and_retrieve (the tool-using agent)
# --------------------------------------------------------------------------- #
def classify_and_retrieve(state: AgentState) -> dict:
    """Invoke Bedrock with tools bound. It either requests a tool or drafts."""
    response = _LLM_WITH_TOOLS.invoke(state["messages"])
    return {"messages": [response]}


# --------------------------------------------------------------------------- #
# Node 2 — tools (prebuilt executor)
# --------------------------------------------------------------------------- #
tool_node = ToolNode(TOOLS)


# --------------------------------------------------------------------------- #
# Node 3 — reflect (separate self-critique)
# --------------------------------------------------------------------------- #
def reflect(state: AgentState) -> dict:
    """Critique the latest draft diagnosis and score confidence 1-10."""
    # The most recent AI message (with no tool calls) is the draft hypothesis.
    draft = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            draft = _message_text(msg)
            break

    review = _LLM.invoke([
        SystemMessage(content=REFLECT_PROMPT),
        HumanMessage(content=f"Hypothesis under review:\n\n{draft}"),
    ])
    review_text = _message_text(review)
    confidence = _parse_confidence(review_text)
    iterations = state["iterations"] + 1

    update = {
        "final_answer": draft,
        "confidence_score": confidence,
        "evidence_gaps": review_text,
        "iterations": iterations,
    }

    # If we're going to loop (low confidence and budget left), feed the gaps
    # back in as a new instruction so the next pass can close them.
    if confidence < CONFIDENCE_THRESHOLD and iterations < MAX_ITERATIONS:
        update["messages"] = [HumanMessage(content=(
            "Your previous diagnosis was reviewed and found low-confidence. "
            "Address these evidence gaps by gathering more information with the "
            f"tools, then revise your diagnosis:\n\n{review_text}"
        ))]
    return update


# --------------------------------------------------------------------------- #
# Conditional routing
# --------------------------------------------------------------------------- #
def route_after_agent(state: AgentState) -> str:
    """If the agent asked for a tool, run it; otherwise go reflect on the draft."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "reflect"


def route_after_reflect(state: AgentState) -> str:
    """End if confident enough or out of budget; otherwise loop back."""
    if (state["confidence_score"] >= CONFIDENCE_THRESHOLD
            or state["iterations"] >= MAX_ITERATIONS):
        return END
    return "classify_and_retrieve"


# --------------------------------------------------------------------------- #
# Build the graph
# --------------------------------------------------------------------------- #
def _build_graph():
    g = StateGraph(AgentState)
    g.add_node("classify_and_retrieve", classify_and_retrieve)
    g.add_node("tools", tool_node)
    g.add_node("reflect", reflect)

    g.add_edge(START, "classify_and_retrieve")
    g.add_conditional_edges(
        "classify_and_retrieve", route_after_agent,
        {"tools": "tools", "reflect": "reflect"},
    )
    g.add_edge("tools", "classify_and_retrieve")  # tools always return to agent
    g.add_conditional_edges(
        "reflect", route_after_reflect,
        {"classify_and_retrieve": "classify_and_retrieve", END: END},
    )
    return g.compile()


_GRAPH = _build_graph()

# Enable MLflow tracing of the LangChain/LangGraph run (LLMOps tracing).
# Wrapped so a missing/misconfigured MLflow never blocks a diagnosis.
try:
    mlflow.langchain.autolog()
except Exception as exc:  # noqa: BLE001
    print(f"[agent] MLflow autolog unavailable: {exc}")


# --------------------------------------------------------------------------- #
# Post-run extraction helpers
# --------------------------------------------------------------------------- #
def _extract_run_facts(messages: list) -> dict:
    """Mine the message history for fault_category, retrieval count, tool calls."""
    fault_category = None
    similar_incidents_found = 0
    tool_calls = []

    for msg in messages:
        # Record every tool the agent requested (name + args).
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({"name": tc["name"], "args": tc["args"]})
        # Parse tool RESULTS to pull out structured facts.
        if isinstance(msg, ToolMessage):
            try:
                payload = json.loads(msg.content) if isinstance(msg.content, str) \
                    else msg.content
            except (json.JSONDecodeError, TypeError):
                payload = None
            if msg.name == "classify_fault" and isinstance(payload, dict):
                fault_category = payload.get("fault_category", fault_category)
            elif msg.name == "retrieve_similar_incidents" and isinstance(payload, list):
                # Don't count an error-only payload as a found incident.
                real = [x for x in payload if isinstance(x, dict) and "error" not in x]
                similar_incidents_found = len(real)

    return {
        "fault_category": fault_category,
        "similar_incidents_found": similar_incidents_found,
        "tool_calls": tool_calls,
    }


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def run_agent(equipment_id: str, log_entry: Optional[str] = None) -> dict:
    """Run a full diagnosis for one machine and return a structured result.

    Returns: final_answer, confidence_score, fault_category,
    similar_incidents_found (count), iterations_taken.
    """
    # Seed the conversation. The human turn frames the task; the system prompt
    # enforces the (a)->(d) diagnostic procedure.
    human = f"Equipment ID: {equipment_id}\n"
    if log_entry:
        human += f"Maintenance log entry: {log_entry}\n"
    human += "Diagnose the likely fault and recommend a grounded resolution."

    initial_state: AgentState = {
        "messages": [SystemMessage(content=SYSTEM_PROMPT),
                     HumanMessage(content=human)],
        "equipment_id": equipment_id,
        "log_entry": log_entry,
        "iterations": 0,
        "confidence_score": 0,
        "evidence_gaps": "",
        "final_answer": "",
    }

    # MLflow run = one diagnosis. autolog adds the detailed LLM/tool trace;
    # here we log the high-level inputs/outputs as params/metrics/artifacts.
    with mlflow.start_run(run_name=f"diagnosis-{equipment_id}"):
        mlflow.log_params({
            "equipment_id": equipment_id,
            "log_entry": (log_entry or "")[:250],  # params have length limits
            "model_id": ANTHROPIC_MODEL,
        })

        # recursion_limit must comfortably exceed our 3 loops x (agent+tools).
        final_state = _GRAPH.invoke(initial_state, config={"recursion_limit": 50})

        facts = _extract_run_facts(final_state["messages"])
        result = {
            "final_answer": final_state.get("final_answer", ""),
            "confidence_score": final_state.get("confidence_score", 0),
            "fault_category": facts["fault_category"],
            "similar_incidents_found": facts["similar_incidents_found"],
            "iterations_taken": final_state.get("iterations", 0),
        }

        # Trace-level logging for LLMOps: metrics + the full structured record.
        mlflow.log_metric("confidence_score", result["confidence_score"])
        mlflow.log_metric("iterations_taken", result["iterations_taken"])
        mlflow.log_metric("similar_incidents_found", result["similar_incidents_found"])
        mlflow.log_dict(
            {**result, "tool_calls": facts["tool_calls"],
             "equipment_id": equipment_id, "log_entry": log_entry},
            "agent_run.json",
        )

    return result


if __name__ == "__main__":
    # Manual smoke test (requires AWS creds + Bedrock + Weaviate to fully run).
    out = run_agent(
        equipment_id="1",
        log_entry="DE bearing running hot at 82C, growl at 1480 rpm, vibration up.",
    )
    print(json.dumps(out, indent=2))
