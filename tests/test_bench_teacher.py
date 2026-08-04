"""The teacher arm's benchmark: standard pricing, and a label that prevents misuse.

Two acceptance criteria live in this file's subject and nowhere else:

- `results/bench/teacher.json` carries `"measurement": "api_wall_clock"`, so nobody
  reads a network round-trip as if it were a local `generate()`;
- it is priced at **standard** rates. F2 labels the corpus through the Batch API at
  a 50% discount, which is right for a bulk offline job and wrong for a
  latency-sensitive one that cannot wait 24 hours. Reusing `teacher.cost_of` here
  would understate the figure by exactly a factor of two, which is why
  `standard_cost` exists and why this test pins the ratio.

No network: the client is a fake. `run` never touches `make_client` when one is
injected, so this costs nothing and needs no key.
"""

from __future__ import annotations

import pytest

from _fakes import doc
from sxl.bench_teacher import TEACHER_BENCH_KEYS, run, standard_cost
from sxl.config import TEACHER_MODEL, TEACHER_PRICE_USD
from sxl.gpu.bench import BenchError


class FakeUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class FakeResponse:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.usage = FakeUsage(prompt_tokens, completion_tokens)


class FakeCompletions:
    def __init__(self, owner: FakeChatClient) -> None:
        self.owner = owner

    def create(self, **body):
        self.owner.calls.append(body)
        if self.owner.fail_on and len(self.owner.calls) in self.owner.fail_on:
            raise RuntimeError("simulated API failure")
        return FakeResponse(1200, 150)


class FakeChat:
    def __init__(self, owner: FakeChatClient) -> None:
        self.completions = FakeCompletions(owner)


class FakeChatClient:
    """The one surface `bench_teacher` touches: `chat.completions.create`."""

    def __init__(self, fail_on: set[int] | None = None) -> None:
        self.calls: list[dict] = []
        self.fail_on = fail_on or set()
        self.chat = FakeChat(self)


def docs(n: int) -> list[dict]:
    return [doc(i) for i in range(n)]


def silent(_message: str) -> None:
    pass


# --- pricing ------------------------------------------------------------------


def test_standard_cost_is_exactly_double_the_batch_rate_f2_labels_at():
    # The whole reason this function is not `teacher.cost_of`.
    from sxl.teacher import cost_of

    usage = {"input_tokens": 1_200_000, "output_tokens": 350_000}
    assert standard_cost(usage) == pytest.approx(cost_of(usage) * 2)


def test_standard_cost_matches_the_published_rate_card_by_hand():
    price = TEACHER_PRICE_USD[TEACHER_MODEL]
    # 1M in + 1M out at gpt-4o-mini standard rates = $0.15 + $0.60.
    assert standard_cost({"input_tokens": 1_000_000, "output_tokens": 1_000_000}) == pytest.approx(
        price["in"] + price["out"]
    )


# --- the record ---------------------------------------------------------------


def test_the_record_has_exactly_the_contracted_keys_in_order():
    record = run(docs(5), n=5, client=FakeChatClient(), echo=silent)
    assert tuple(record) == TEACHER_BENCH_KEYS


def test_the_record_is_labelled_as_a_network_measurement_not_a_gpu_one():
    # Comparing an API round-trip to a local generate() is apples-to-oranges, and
    # the file has to say so itself -- F8 reads the file, not this test.
    record = run(docs(5), n=5, client=FakeChatClient(), echo=silent)
    assert record["measurement"] == "api_wall_clock"
    assert record["gpu_name"] is None
    assert record["dtype"] is None
    assert record["cache_implementation"] is None


def test_the_gpu_hourly_rate_is_null_because_no_gpu_is_rented():
    # And therefore the cost is NOT derivable via cost_per_1k: for an API arm it is
    # a token quantity, not a rented-seconds quantity.
    record = run(docs(5), n=5, client=FakeChatClient(), echo=silent)
    assert record["gpu_hourly_usd"] is None
    assert record["cost_per_1k_docs_usd"] > 0


def test_cost_per_1k_is_the_mean_per_document_cost_scaled_by_a_thousand():
    record = run(docs(10), n=10, client=FakeChatClient(), echo=silent)
    per_doc = standard_cost({"input_tokens": 1200, "output_tokens": 150})
    assert record["cost_per_1k_docs_usd"] == pytest.approx(per_doc * 1000, abs=1e-6)
    assert record["total_usd"] == pytest.approx(per_doc * 10, abs=1e-6)


def test_the_record_states_that_no_batch_discount_was_applied():
    record = run(docs(5), n=5, client=FakeChatClient(), echo=silent)
    assert record["pricing"]["batch_discount"] is False
    assert record["pricing"]["rates_usd_per_1m"] == TEACHER_PRICE_USD[TEACHER_MODEL]


def test_no_accuracy_metric_appears_in_the_teacher_bench_record():
    assert {"macro_f1", "schema_valid_rate"}.isdisjoint(TEACHER_BENCH_KEYS)


# --- call behaviour -----------------------------------------------------------


def test_calls_are_sequential_and_use_the_same_body_the_labeling_run_used():
    # Same model, prompt, temperature and strict-JSON response format as F2, so the
    # timing describes the configuration that produced results/metrics/teacher.json.
    client = FakeChatClient()
    run(docs(5), n=5, client=client, echo=silent)
    assert len(client.calls) == 5
    for body in client.calls:
        assert body["model"] == TEACHER_MODEL
        assert body["response_format"]["json_schema"]["strict"] is True
        assert [m["role"] for m in body["messages"]] == ["system", "user"]


def test_a_flaky_call_is_counted_and_skipped_rather_than_losing_the_whole_run():
    client = FakeChatClient(fail_on={3})
    record = run(docs(10), n=10, client=client, echo=silent)
    assert record["n_calls_failed"] == 1
    assert record["n_calls_ok"] == 9
    # `n_docs` reports what was actually measured, not what was requested.
    assert record["n_docs"] == 9


def test_a_run_where_every_call_fails_raises_rather_than_reporting_zeros():
    client = FakeChatClient(fail_on=set(range(1, 6)))
    with pytest.raises(BenchError, match="all"):
        run(docs(5), n=5, client=client, echo=silent)


def test_asking_for_more_documents_than_exist_raises_before_spending_money():
    with pytest.raises(BenchError, match="only 3"):
        run(docs(3), n=50, client=FakeChatClient(), echo=silent)


def test_an_unpriced_model_raises_before_making_any_call():
    # A cost figure derived from a guessed rate is worse than no cost figure.
    client = FakeChatClient()
    with pytest.raises(BenchError, match="pricing"):
        run(docs(5), n=5, model="gpt-9-imaginary", client=client, echo=silent)
    assert client.calls == []
