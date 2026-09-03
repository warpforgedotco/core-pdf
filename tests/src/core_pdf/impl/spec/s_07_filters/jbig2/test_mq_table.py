# SPDX-License-Identifier: AGPL-3.0-only
"""Invariants of the MQ-coder probability table, ITU-T T.88 Table E.1.

The decoder indexes four parallel columns per pixel, so an entry inserted or
dropped in one of them shifts it against the other three and corrupts every
decode without raising. The table is authored as rows and the columns derived
from it, which makes that impossible; these pin the properties that would
otherwise have to be checked by eye against the standard.
"""

from __future__ import annotations

from core_pdf.impl.spec.s_07_filters.jbig2.codec import (
    MQ_NLPS,
    MQ_NMPS,
    MQ_QE,
    MQ_SWITCH,
    internal_MQ_STATES,
)

STATE_COUNT = 47


def test_the_columns_are_the_rows() -> None:
    assert len(internal_MQ_STATES) == STATE_COUNT
    assert tuple(state[0] for state in internal_MQ_STATES) == MQ_QE
    assert tuple(state[1] for state in internal_MQ_STATES) == MQ_NMPS
    assert tuple(state[2] for state in internal_MQ_STATES) == MQ_NLPS
    assert tuple(state[3] for state in internal_MQ_STATES) == MQ_SWITCH


def test_every_transition_lands_on_a_real_state() -> None:
    for index, (qe, nmps, nlps, switch) in enumerate(internal_MQ_STATES):
        assert 0 < qe <= 0xFFFF, index
        assert 0 <= nmps < STATE_COUNT, index
        assert 0 <= nlps < STATE_COUNT, index
        assert switch in (0, 1), index


def test_only_the_three_documented_states_switch_the_mps() -> None:
    """Table E.1 sets SWITCH on states 0, 6 and 14 and nowhere else."""
    assert [i for i, switch in enumerate(MQ_SWITCH) if switch] == [0, 6, 14]


def test_the_two_backward_mps_transitions_survive() -> None:
    """Most NMPS entries are index + 1; these two are not, and a column that
    slipped by one would quietly turn them into their neighbours.
    """
    assert MQ_NMPS[5] == 38
    assert MQ_NMPS[13] == 29
    off_by_one = [i for i, n in enumerate(MQ_NMPS) if n != i + 1]
    assert off_by_one == [5, 13, 45, 46]


def test_the_last_state_is_the_fixed_point() -> None:
    """State 46 is T.88's non-adapting state: both transitions return to it."""
    assert internal_MQ_STATES[46] == (0x5601, 46, 46, 0)
