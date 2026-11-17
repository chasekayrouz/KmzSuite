"""Calibration provenance tests using invented coefficients and slot layouts."""
import math

import pytest

from kmzconvert_core.binary import FormatError
from kmzconvert_core.reader import validate_calibration_records


def startup(n_exp=3, first_cycle=2):
    pairs = [(0.001 + exp*0.0001, 2. + exp) for exp in range(n_exp)]
    table = [(*pairs[slot % n_exp], 0) for slot in range((first_cycle+3)*n_exp)]
    embedded = {slot: table[slot][:2] for slot in range(first_cycle*n_exp, len(table))}
    # An earlier snapshot in the first populated cycle, followed by the updated
    # experiment-specific calibration. No measured arrays or filenames needed.
    for exp in range(n_exp):
        embedded[first_cycle*n_exp+exp] = (pairs[exp][0]/2, pairs[exp][1]-1)
    return table, embedded


@pytest.mark.parametrize('n_exp, first_cycle', [(2, 1), (3, 4), (5, 2)])
def test_startup_table_is_corroborated_per_experiment(n_exp, first_cycle):
    table, embedded = startup(n_exp, first_cycle)
    resolved = validate_calibration_records(table, embedded, n_exp)
    assert resolved == {slot: slot + n_exp for slot in range(first_cycle*n_exp, (first_cycle+1)*n_exp)}


def test_empty_dependent_slots_do_not_shift_calibration_association():
    table, embedded = startup()
    del embedded[7], embedded[8]
    assert validate_calibration_records(table, embedded, 3) == {6: 9}


def test_agreeing_records_can_change_each_cycle_without_startup_exception():
    table = [(.001, 2., 0), (.002, 3., 0), (.003, 4., 0), (.004, 5., 0)]
    assert validate_calibration_records(table, {i: p[:2] for i, p in enumerate(table)}, 2) == {}


def stale_run(n_exp=3, first_cycle=0, stale_cycles=4, later_cycles=3):
    """Survey-scan startup run sharing one stale preamble; dependent slots empty."""
    current, stale = (.001, 2.), (.0011, 2.5)
    total = first_cycle + stale_cycles + later_cycles
    table = [(*current, 0) if slot % n_exp == 0 else (0., 0., 0) for slot in range(total*n_exp)]
    embedded = {cycle*n_exp: stale for cycle in range(first_cycle, first_cycle+stale_cycles)}
    following = first_cycle + stale_cycles
    for cycle in range(following, total):
        # After the corroborating cycle, agreeing records may change each cycle.
        if cycle > following:
            table[cycle*n_exp] = (current[0] + cycle*1e-9, current[1], 0)
        embedded[cycle*n_exp] = table[cycle*n_exp][:2]
    return table, embedded


@pytest.mark.parametrize('n_exp, first_cycle, stale_cycles', [(4, 0, 92), (4, 0, 1), (3, 2, 5), (2, 1, 2)])
def test_consecutive_stale_startup_run_is_corroborated(n_exp, first_cycle, stale_cycles):
    table, embedded = stale_run(n_exp, first_cycle, stale_cycles)
    following = (first_cycle + stale_cycles)*n_exp
    assert validate_calibration_records(table, embedded, n_exp) == {
        cycle*n_exp: following for cycle in range(first_cycle, first_cycle+stale_cycles)}


@pytest.mark.parametrize('case', [
    'gap_in_run', 'second_stale_value', 'table_changes_within_run', 'run_starts_late',
    'following_table_changed', 'following_copy_differs', 'stale_through_end', 'prefix_table_changed',
])
def test_stale_startup_run_violations_fail_closed(case):
    table, embedded = stale_run(n_exp=3, first_cycle=1)
    if case == 'gap_in_run':
        embedded[6] = table[6][:2]
    elif case == 'second_stale_value':
        embedded[9] = (.0012, 2.5)
    elif case == 'table_changes_within_run':
        table[9] = (.002, 2., 0)
    elif case == 'run_starts_late':
        embedded[3] = table[3][:2]
    elif case == 'following_table_changed':
        table[15] = (.002, 2., 0)
        embedded[15] = table[15][:2]
    elif case == 'following_copy_differs':
        embedded[15] = (.002, 2.)
    elif case == 'stale_through_end':
        for slot in (15, 18, 21):
            embedded[slot] = embedded[3]
    elif case == 'prefix_table_changed':
        table[0] = (.002, 2., 0)
    with pytest.raises(FormatError, match='Calibration disagreement'):
        validate_calibration_records(table, embedded, 3)


@pytest.mark.parametrize('case', [
    'later_cycle', 'late_first_scan_of_experiment', 'missing_following_scan',
    'stale_without_following_agreement', 'following_table_changed', 'earlier_prefix_changed',
    'wrong_experiment_corroboration', 'one_ulp_disagreement',
])
def test_unexplained_disagreements_still_fail_closed(case):
    table, embedded = startup()
    if case == 'later_cycle':
        embedded[12] = (.008, 9.)
    elif case == 'late_first_scan_of_experiment':
        del embedded[7]
        embedded[10] = (.008, 9.)
    elif case == 'missing_following_scan':
        del embedded[9]
    elif case == 'stale_without_following_agreement':
        embedded[9] = embedded[12] = embedded[6]
    elif case == 'following_table_changed':
        table[9] = (.008, 9., 0)
        embedded[9] = table[9][:2]
    elif case == 'earlier_prefix_changed':
        table[0] = (.008, 9., 0)
    elif case == 'wrong_experiment_corroboration':
        embedded[9] = table[10][:2]
    elif case == 'one_ulp_disagreement':
        a, b = embedded[9]
        embedded[9] = (math.nextafter(a, math.inf), b)
    with pytest.raises(FormatError, match='Calibration disagreement'):
        validate_calibration_records(table, embedded, 3)


@pytest.mark.parametrize('pair', [(0., 2.), (-.001, 2.), (1., 2.), (math.inf, 2.), (.001, math.nan), (.001, math.inf)])
@pytest.mark.parametrize('source', ['table', 'scan'])
def test_invalid_coefficients_are_never_accepted(source, pair):
    table, embedded = startup()
    if source == 'table':
        table[6] = (*pair, 0)
    else:
        embedded[6] = pair
    with pytest.raises(FormatError, match='Invalid .*calibration record'):
        validate_calibration_records(table, embedded, 3)


@pytest.mark.parametrize('slot', [0, 6, 9])
def test_nonzero_reserved_field_cannot_corroborate_startup(slot):
    table, embedded = startup()
    table[slot] = (*table[slot][:2], 1)
    with pytest.raises(FormatError):
        validate_calibration_records(table, embedded, 3)
