from nabard_qa.generation.llm import ExtractiveGenerator, GenerationRequest
from nabard_qa.generation.prompts import SYSTEM_PROMPT, build_messages, build_user_message
from nabard_qa.schemas import ScoredChunk, Turn

from .conftest import CORPUS


def _sources():
    return [ScoredChunk(chunk=c, score=1.0) for c in CORPUS[:2]]


def test_user_message_escapes_and_labels_sources():
    msg = build_user_message("What is <RIDF>?", _sources(), profile="woman & SC")
    assert '<source id="S1" report="NABARD Annual Report 2017-18" pages="p. 100"' in msg
    assert "Question: What is <RIDF>?" in msg
    assert "woman &amp; SC" in msg


def test_build_messages_alternates_roles_and_starts_with_user():
    hist = [Turn(role="assistant", content="hi"), Turn(role="user", content="a"), Turn(role="user", content="b"),
            Turn(role="assistant", content="c")]
    msgs = build_messages("next?", _sources(), hist)
    roles = [m["role"] for m in msgs]
    assert roles[0] == "user" and roles[-1] == "user"
    assert all(a != b for a, b in zip(roles, roles[1:]))


def test_system_prompt_is_static():
    assert "{" not in SYSTEM_PROMPT  # no unfilled template slots / volatile values


def test_extractive_quotes_relevant_sentence_with_citation():
    req = GenerationRequest(system="", messages=[], question="Long Term Irrigation Fund interest rate", sources=_sources())
    out = ExtractiveGenerator().complete(req)
    assert "6 per cent" in out and "[S2]" in out


def test_extractive_declines_without_sources():
    out = ExtractiveGenerator().complete(GenerationRequest(system="", messages=[], question="x", sources=[]))
    assert "could not find" in out
