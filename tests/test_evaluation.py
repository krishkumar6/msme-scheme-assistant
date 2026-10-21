"""Answer scoring: out-of-scope questions pass only on an explicit decline."""

from types import SimpleNamespace

from nabard_qa.evaluation import GoldenItem, evaluate_answers


class _FakeService:
    def __init__(self, answer: str, grounded: bool) -> None:
        self.generator = SimpleNamespace(name="fake")
        self._res = SimpleNamespace(answer=answer, grounded=grounded, provider="fake")

    def answer(self, _req):  # noqa: ANN001, ANN202
        return self._res


OUT_OF_SCOPE = [GoldenItem(id="na", question="What is SBI's home loan rate?", answerable=False)]


def test_invented_ungrounded_answer_is_not_a_decline():
    r = evaluate_answers(_FakeService("The current home loan interest rate at SBI is 8.5%. [S1]", False), OUT_OF_SCOPE)
    assert r["decline_rate"] == 0.0


def test_explicit_decline_passes():
    r = evaluate_answers(_FakeService("The NABARD reports I have do not contain this information.", False),
                         OUT_OF_SCOPE)
    assert r["decline_rate"] == 1.0
