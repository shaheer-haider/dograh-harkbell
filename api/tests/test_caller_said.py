"""The caller's words, as transcribed, where a tool can read them.

The Mirror Labs copy, 2026-11-19: the smaller realtime model filed a
nine-digit order number given in two halves wrong on six runs out of six,
and the transcript had both halves right every time. ``note_caller_turn``
keeps the last turns in the run's context, and a preset hands them to a tool.
"""

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.tests.mock_llm_service import MockLLMService

from api.services.workflow.dto import ReactFlowDTO
from api.services.workflow.pipecat_engine import PipecatEngine
from api.services.workflow.tools.custom_tool import _resolve_preset_parameters
from api.services.workflow.workflow_graph import WorkflowGraph
from api.tests.test_take_pathway_without_model import screening_definition


def _engine():
    graph = WorkflowGraph(ReactFlowDTO.model_validate(screening_definition()))
    return PipecatEngine(
        llm=MockLLMService(mock_steps=[], chunk_delay=0.001),
        context=LLMContext(),
        workflow=graph,
        call_context_vars={},
        workflow_run_id=1,
    )


PRESET = {
    "preset_parameters": [
        {
            "name": "caller_said",
            "type": "string",
            "value_template": "{{gathered_context.caller_said}}",
            "required": False,
        }
    ]
}


def _handed_to_a_tool(engine):
    return _resolve_preset_parameters(
        PRESET, engine._call_context_vars, engine._gathered_context
    )


def test_a_number_given_in_two_halves_reaches_the_tool_as_it_was_said():
    engine = _engine()
    # Nothing said yet: an optional preset that renders empty is left out.
    assert _handed_to_a_tool(engine) == {}
    engine.note_caller_turn("Hi, StubHub support, calling about an order.")
    engine.note_caller_turn("The order number is six five eight seven.")
    engine.note_caller_turn("  Four four   nine five seven. ")
    assert _handed_to_a_tool(engine) == {
        "caller_said": "Hi, StubHub support, calling about an order."
        " | The order number is six five eight seven."
        " | Four four nine five seven."
    }


def test_a_key_press_and_an_empty_turn_are_not_something_they_said():
    engine = _engine()
    engine.note_caller_turn("[keypad] 1")
    engine.note_caller_turn("")
    engine.note_caller_turn(None)
    assert "caller_said" not in engine._gathered_context
    engine.note_caller_turn("Order six four four one.")
    assert engine._gathered_context["caller_said"] == "Order six four four one."


def test_only_the_last_turns_are_kept():
    engine = _engine()
    for n in range(20):
        engine.note_caller_turn(f"turn {n}")
    said = engine._gathered_context["caller_said"].split(" | ")
    assert said == [f"turn {n}" for n in range(12, 20)]
    # A long ramble is cut from the front: the number is at the end of it.
    engine.note_caller_turn("word " * 400 + "six five eight seven")
    assert len(engine._gathered_context["caller_said"]) <= PipecatEngine.CALLER_SAID_CHARS
    assert engine._gathered_context["caller_said"].endswith("six five eight seven")
