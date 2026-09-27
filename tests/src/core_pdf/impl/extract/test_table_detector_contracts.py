from copy import replace

from core_pdf.impl.extract_contracts import ObservationBatch
from core_pdf.impl.extract_table_core import (
    TableAnalysis,
    TableCandidate,
    TableContext,
    TableFacts,
    TableRule,
    digit_bearing_cell,
    numeric_cell,
    short_digit_cell,
)
from core_pdf.impl.extract_table_detection import TableDetector
from core_pdf.impl.output_model import Table, TableCell


def make_table(order: int, top: float, texts: list[list[str]]) -> Table:
    rows = tuple(
        tuple(TableCell(row, column, text) for column, text in enumerate(values))
        for row, values in enumerate(texts)
    )
    return Table(order=order, rows=rows, bbox=(0.0, top - 10.0, 50.0, top))


class FixedSource:
    def __init__(self, *tables: Table) -> None:
        self.tables = tables
        self.start_orders: list[int] = []

    def detect(self, context: TableContext, start_order: int) -> tuple[TableCandidate, ...]:
        self.start_orders.append(start_order)
        return tuple(TableCandidate(table) for table in self.tables)

    def admit(
        self, accepted: list[TableCandidate], found: tuple[TableCandidate, ...]
    ) -> list[TableCandidate]:
        return [*accepted, *found]


def context() -> TableContext:
    return TableContext(None, TableAnalysis.build(ObservationBatch.empty(), 100.0))  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


def test_candidates_compute_facts_once_and_keep_them_while_rows_are_unchanged() -> None:
    table = make_table(0, 100.0, [["a", "1"], ["b", "2"]])
    candidate = TableCandidate(table)
    facts = candidate.facts
    assert facts == TableFacts.from_rows(table.rows)
    assert candidate.facts is facts
    assert candidate.with_table(table) is candidate
    assert candidate.with_table(replace(table, order=4)).facts is facts
    assert candidate.with_table(replace(table, rows=table.rows[:1])).cached_facts is None


def test_supplements_bypass_the_filters_and_sources_number_from_accepted_tables() -> None:
    kept = make_table(0, 100.0, [["a", "b"]])
    rejected = make_table(1, 80.0, [["reject", "me"]])
    supplement = make_table(-1, 60.0, [["reject", "chart"]])
    first = FixedSource(kept, rejected)
    second = FixedSource()
    extra = FixedSource(supplement)
    detector = TableDetector(
        None,
        (first, second),
        filters=(TableRule("reject", lambda item: item.table.rows[0][0].text == "reject"),),
        supplements=(extra,),
    )
    assert detector.detect(context()) == (kept, supplement)
    assert (first.start_orders, second.start_orders, extra.start_orders) == ([0], [2], [1])


def test_a_closed_gate_skips_detection() -> None:
    source = FixedSource(make_table(0, 100.0, [["a", "b"]]))
    detector = replace(TableDetector(None, (source,)), gate=lambda capture: False)
    assert detector.extract(None, ObservationBatch.empty()) == ()  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
    assert source.start_orders == []


def test_the_three_numeric_cell_definitions_stay_distinct() -> None:
    assert (numeric_cell("a1"), digit_bearing_cell("a1"), short_digit_cell("a1")) == (
        True,
        True,
        True,
    )
    assert (numeric_cell("ab1"), digit_bearing_cell("ab1"), short_digit_cell("ab1")) == (
        False,
        True,
        True,
    )
    long_text = "a" * 30 + "1"
    assert (digit_bearing_cell(long_text), short_digit_cell(long_text)) == (True, False)
