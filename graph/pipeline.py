from langgraph.graph import StateGraph, END

from graph.state import RAGState
from graph.nodes import check_cache, retrieve_chunks, generate_answer, format_response

_graph = None


def _route_after_cache(state: RAGState) -> str:
    return END if state["cache_hit"] else "retrieve"


def build_graph():
    sg = StateGraph(RAGState)

    sg.add_node("cache",    check_cache)
    sg.add_node("retrieve", retrieve_chunks)
    sg.add_node("generate", generate_answer)
    sg.add_node("format",   format_response)

    sg.set_entry_point("cache")
    sg.add_conditional_edges("cache", _route_after_cache, {END: END, "retrieve": "retrieve"})
    sg.add_edge("retrieve", "generate")
    sg.add_edge("generate", "format")
    sg.add_edge("format",   END)

    return sg.compile()


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
