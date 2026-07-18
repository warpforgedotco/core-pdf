# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

T1_CTXNO_ZC = 0
T1_CTXNO_SC = 9
T1_CTXNO_MAG = 14
T1_CTXNO_AGG = 17
T1_CTXNO_UNI = 18

T1_ORIENT_LL = 0
T1_ORIENT_HL = 1
T1_ORIENT_LH = 2
T1_ORIENT_HH = 3


class JpxTier1State:
    __slots__ = ("width", "height", "data", "significant", "visited", "refined")

    def __init__(self, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("invalid JPX Tier-1 block dimensions")
        self.width = width
        self.height = height
        self.data = [0] * (width * height)
        self.significant = [False] * (width * height)
        self.visited = [False] * (width * height)
        self.refined = [False] * (width * height)

    def index(self, x: int, y: int) -> int:
        return y * self.width + x

    def is_significant(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height and self.significant[self.index(x, y)]

    def ignore_north_neighbors(
        self,
        y: int,
        vertical_stripe_causal: bool,
    ) -> bool:
        return vertical_stripe_causal and y % 4 == 0

    def neighbor_counts(
        self,
        x: int,
        y: int,
        vertical_stripe_causal: bool = False,
    ) -> tuple[int, int, int]:
        width = self.width
        height = self.height
        significant = self.significant
        row = y * width
        ignore_north = self.ignore_north_neighbors(y, vertical_stripe_causal)
        horizontal = 0
        if x > 0 and significant[row + x - 1]:
            horizontal += 1
        if x + 1 < width and significant[row + x + 1]:
            horizontal += 1
        vertical = 0
        if y > 0 and not ignore_north and significant[row - width + x]:
            vertical += 1
        if y + 1 < height and significant[row + width + x]:
            vertical += 1
        diagonal = 0
        if y > 0 and not ignore_north:
            prev = row - width
            if x > 0 and significant[prev + x - 1]:
                diagonal += 1
            if x + 1 < width and significant[prev + x + 1]:
                diagonal += 1
        if y + 1 < height:
            nxt = row + width
            if x > 0 and significant[nxt + x - 1]:
                diagonal += 1
            if x + 1 < width and significant[nxt + x + 1]:
                diagonal += 1
        return horizontal, vertical, diagonal

    def has_significant_neighbor(
        self,
        x: int,
        y: int,
        vertical_stripe_causal: bool = False,
    ) -> bool:
        width = self.width
        height = self.height
        significant = self.significant
        row = y * width
        ignore_north = self.ignore_north_neighbors(y, vertical_stripe_causal)
        if x > 0 and significant[row + x - 1]:
            return True
        if x + 1 < width and significant[row + x + 1]:
            return True
        if y > 0 and not ignore_north:
            prev = row - width
            if significant[prev + x]:
                return True
            if x > 0 and significant[prev + x - 1]:
                return True
            if x + 1 < width and significant[prev + x + 1]:
                return True
        if y + 1 < height:
            nxt = row + width
            if significant[nxt + x]:
                return True
            if x > 0 and significant[nxt + x - 1]:
                return True
            if x + 1 < width and significant[nxt + x + 1]:
                return True
        return False

    def sign_context(
        self,
        x: int,
        y: int,
        vertical_stripe_causal: bool = False,
    ) -> tuple[int, int]:
        width = self.width
        height = self.height
        significant = self.significant
        data = self.data
        row = y * width
        ignore_north = self.ignore_north_neighbors(y, vertical_stripe_causal)
        positive_horizontal = 0
        negative_horizontal = 0
        if x > 0 and significant[row + x - 1]:
            if data[row + x - 1] < 0:
                negative_horizontal += 1
            else:
                positive_horizontal += 1
        if x + 1 < width and significant[row + x + 1]:
            if data[row + x + 1] < 0:
                negative_horizontal += 1
            else:
                positive_horizontal += 1
        positive_vertical = 0
        negative_vertical = 0
        if y > 0 and not ignore_north and significant[row - width + x]:
            if data[row - width + x] < 0:
                negative_vertical += 1
            else:
                positive_vertical += 1
        if y + 1 < height and significant[row + width + x]:
            if data[row + width + x] < 0:
                negative_vertical += 1
            else:
                positive_vertical += 1

        horizontal = min(positive_horizontal, 1) - min(negative_horizontal, 1)
        vertical = min(positive_vertical, 1) - min(negative_vertical, 1)
        context_horizontal = horizontal
        context_vertical = vertical
        if context_horizontal < 0:
            context_horizontal = -context_horizontal
            context_vertical = -context_vertical
        if context_horizontal == 0:
            context_offset = 0 if context_vertical == 0 else 1
        elif context_vertical < 0:
            context_offset = 2
        elif context_vertical == 0:
            context_offset = 3
        else:
            context_offset = 4
        predicted_sign = 0
        if horizontal != 0 or vertical != 0:
            predicted_sign = int(not (horizontal > 0 or (horizontal == 0 and vertical > 0)))
        return T1_CTXNO_SC + context_offset, predicted_sign

    def update_significance(self, x: int, y: int, value: int) -> None:
        index = y * self.width + x
        self.data[index] = value
        self.significant[index] = True

    def reset_pass_flags(self) -> None:
        self.visited = [False] * (self.width * self.height)


def t1_zero_coding_context(
    state: JpxTier1State,
    x: int,
    y: int,
    orientation: int,
    vertical_stripe_causal: bool = False,
) -> int:
    horizontal, vertical, diagonal = state.neighbor_counts(
        x,
        y,
        vertical_stripe_causal,
    )
    return t1_zero_coding_context_from_counts(horizontal, vertical, diagonal, orientation)


def t1_zero_coding_context_from_counts(
    horizontal: int,
    vertical: int,
    diagonal: int,
    orientation: int,
) -> int:
    if orientation == T1_ORIENT_HL:
        horizontal, vertical = vertical, horizontal
    if orientation == T1_ORIENT_HH:
        hv = horizontal + vertical
        if diagonal == 0:
            if hv == 0:
                offset = 0
            elif hv == 1:
                offset = 1
            else:
                offset = 2
        elif diagonal == 1:
            if hv == 0:
                offset = 3
            elif hv == 1:
                offset = 4
            else:
                offset = 5
        elif diagonal == 2:
            offset = 6 if hv == 0 else 7
        else:
            offset = 8
        return T1_CTXNO_ZC + offset
    if horizontal == 0:
        if vertical == 0:
            if diagonal == 0:
                offset = 0
            elif diagonal == 1:
                offset = 1
            else:
                offset = 2
        elif vertical == 1:
            offset = 3
        else:
            offset = 4
    elif horizontal == 1:
        if vertical == 0:
            offset = 5 if diagonal == 0 else 6
        else:
            offset = 7
    else:
        offset = 8
    return T1_CTXNO_ZC + offset


def t1_magnitude_context(
    state: JpxTier1State,
    x: int,
    y: int,
    vertical_stripe_causal: bool = False,
) -> int:
    index = state.index(x, y)
    if state.refined[index]:
        return T1_CTXNO_MAG + 2
    if state.has_significant_neighbor(x, y, vertical_stripe_causal):
        return T1_CTXNO_MAG + 1
    return T1_CTXNO_MAG
