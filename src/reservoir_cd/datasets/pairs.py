"""Year-pair policy: same season (Jan-Apr medoid), consecutive years."""


def consecutive_pairs(years: list[int], step: int = 1) -> list[tuple[int, int]]:
    """(T1, T2) with T2 = T1 + step, both present in ``years``. 2015..2026, step 1 -> 11 pairs."""
    ys = sorted(set(years))
    return [(y, y + step) for y in ys if y + step in ys]
