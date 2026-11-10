# SPDX-License-Identifier: AGPL-3.0-only

import cython
from cython.cimports.cpython.mem import PyMem_Free, PyMem_Malloc, PyMem_Realloc

# One queued pair: (skip_between, distance, first id, second id), as pdfminer's
# group_textboxes keeps it in a Python tuple.
Entry = cython.struct(
    skip=cython.int,
    distance=cython.double,
    first=cython.Py_ssize_t,
    second=cython.Py_ssize_t,
)

# CPython's heapq turns to its cache-friendly heapify above this many items.
CACHE_FRIENDLY_HEAPIFY = cython.declare(cython.Py_ssize_t, 2500)


@cython.cfunc
@cython.inline
@cython.exceptval(check=False)
def entry_lt(a: Entry, b: Entry) -> cython.bint:
    # The tuples' < : the first element that is not == decides, so a NaN distance,
    # != everything, decides by its < , which is false.
    if a.skip != b.skip:
        return a.skip < b.skip
    if a.distance != b.distance:
        return a.distance < b.distance
    if a.first != b.first:
        return a.first < b.first
    return a.second < b.second


@cython.cfunc
@cython.exceptval(check=False)
def sift_down(
    heap: cython.pointer[Entry], start: cython.Py_ssize_t, pos: cython.Py_ssize_t
) -> cython.void:
    # _heapqmodule.c's siftdown.
    item: Entry = heap[pos]
    parent: cython.Py_ssize_t
    while pos > start:
        parent = (pos - 1) >> 1
        if entry_lt(item, heap[parent]):
            heap[pos] = heap[parent]
            pos = parent
            continue
        break
    heap[pos] = item


@cython.cfunc
@cython.exceptval(check=False)
def sift_up(
    heap: cython.pointer[Entry], size: cython.Py_ssize_t, pos: cython.Py_ssize_t
) -> cython.void:
    # _heapqmodule.c's siftup: move the smaller child up until a leaf, then sift down.
    start: cython.Py_ssize_t = pos
    limit: cython.Py_ssize_t = size >> 1
    child: cython.Py_ssize_t
    swap: Entry
    while pos < limit:
        child = 2 * pos + 1
        if child + 1 < size and not entry_lt(heap[child], heap[child + 1]):
            child += 1
        swap = heap[child]
        heap[child] = heap[pos]
        heap[pos] = swap
        pos = child
    sift_down(heap, start, pos)


@cython.cfunc
@cython.exceptval(check=False)
def keep_top_bit(n: cython.Py_ssize_t) -> cython.Py_ssize_t:
    shift: cython.int = 0
    while n > 1:
        n >>= 1
        shift += 1
    return n << shift


@cython.cfunc
@cython.exceptval(check=False)
def heapify(heap: cython.pointer[Entry], size: cython.Py_ssize_t) -> cython.void:
    # _heapqmodule.c's heapify_internal, with its cache_friendly_heapify for large heaps.
    i: cython.Py_ssize_t
    j: cython.Py_ssize_t
    m: cython.Py_ssize_t
    leftmost: cython.Py_ssize_t
    if size <= CACHE_FRIENDLY_HEAPIFY:
        i = (size >> 1) - 1
        while i >= 0:
            sift_up(heap, size, i)
            i -= 1
        return
    m = size >> 1
    leftmost = keep_top_bit(m + 1) - 1
    i = leftmost - 1
    while i >= (m >> 1):
        j = i
        while True:
            sift_up(heap, size, j)
            if not (j & 1):
                break
            j >>= 1
        i -= 1
    i = m - 1
    while i >= leftmost:
        j = i
        while True:
            sift_up(heap, size, j)
            if not (j & 1):
                break
            j >>= 1
        i -= 1


@cython.cclass
class AreaGapQueue:
    """The pair queue of pdfminer's group_textboxes, in C.

    Holds every pair of the given boxes by their distance, the area of their union box
    less both their areas, computed in double precision as Python computes it. pop()
    yields the next pair whose two ids are both still active, as heapq.heappop over the
    Python tuples would; merge() retires a pair, adds their group under a new id and
    queues its distance to every active id in the order they became active.
    """

    heap: cython.pointer[Entry]
    size: cython.Py_ssize_t
    capacity: cython.Py_ssize_t
    bounds: cython.pointer[cython.double]
    active: cython.pointer[cython.char]
    order: cython.pointer[cython.Py_ssize_t]
    order_size: cython.Py_ssize_t
    id_capacity: cython.Py_ssize_t
    active_count: cython.Py_ssize_t
    compact_at: cython.Py_ssize_t
    last: Entry

    def __cinit__(self, boxes: list):
        count: cython.Py_ssize_t = len(boxes)
        i: cython.Py_ssize_t
        j: cython.Py_ssize_t
        self.heap = cython.NULL
        self.bounds = cython.NULL
        self.active = cython.NULL
        self.order = cython.NULL
        self.size = 0
        self.order_size = 0
        # Every merge retires two ids and adds one, so ids stop below 2 * count.
        self.id_capacity = 2 * count if count else 1
        self.capacity = count * (count - 1) // 2 + count + 1
        self.heap = cython.cast(
            cython.pointer[Entry], PyMem_Malloc(self.capacity * cython.sizeof(Entry))
        )
        self.bounds = cython.cast(
            cython.pointer[cython.double],
            PyMem_Malloc(5 * self.id_capacity * cython.sizeof(cython.double)),
        )
        self.active = cython.cast(cython.pointer[cython.char], PyMem_Malloc(self.id_capacity))
        self.order = cython.cast(
            cython.pointer[cython.Py_ssize_t],
            PyMem_Malloc(self.id_capacity * cython.sizeof(cython.Py_ssize_t)),
        )
        if not self.heap or not self.bounds or not self.active or not self.order:
            raise MemoryError()
        for i in range(self.id_capacity):
            self.active[i] = 0
        for i in range(count):
            x0, y0, x1, y1 = boxes[i]
            self.set_bounds(i, x0, y0, x1, y1)
            self.active[i] = 1
            self.order[i] = i
        self.order_size = count
        self.active_count = count
        for i in range(count):
            for j in range(i + 1, count):
                self.heap[self.size] = self.gap(i, j)
                self.size += 1
        heapify(self.heap, self.size)
        self.compact_at = count // 2

    def __dealloc__(self):
        PyMem_Free(self.heap)
        PyMem_Free(self.bounds)
        PyMem_Free(self.active)
        PyMem_Free(self.order)

    @cython.cfunc
    def set_bounds(
        self,
        index: cython.Py_ssize_t,
        x0: cython.double,
        y0: cython.double,
        x1: cython.double,
        y1: cython.double,
    ) -> cython.void:
        base: cython.pointer[cython.double] = self.bounds + 5 * index
        base[0] = x0
        base[1] = y0
        base[2] = x1
        base[3] = y1
        base[4] = (x1 - x0) * (y1 - y0)

    @cython.cfunc
    @cython.exceptval(check=False)
    def gap(self, first: cython.Py_ssize_t, other: cython.Py_ssize_t) -> Entry:
        # min(first, other) is other only when other < first, and max likewise.
        f: cython.pointer[cython.double] = self.bounds + 5 * first
        o: cython.pointer[cython.double] = self.bounds + 5 * other
        x0: cython.double = o[0] if o[0] < f[0] else f[0]
        y0: cython.double = o[1] if o[1] < f[1] else f[1]
        x1: cython.double = o[2] if o[2] > f[2] else f[2]
        y1: cython.double = o[3] if o[3] > f[3] else f[3]
        entry = cython.declare(Entry)
        entry.skip = 0
        entry.distance = (x1 - x0) * (y1 - y0) - f[4] - o[4]
        entry.first = first
        entry.second = other
        return entry

    @cython.cfunc
    def push(self, entry: Entry) -> cython.void:
        if self.size == self.capacity:
            grown = cython.cast(
                cython.pointer[Entry],
                PyMem_Realloc(self.heap, 2 * self.capacity * cython.sizeof(Entry)),
            )
            if not grown:
                raise MemoryError()
            self.heap = grown
            self.capacity *= 2
        self.heap[self.size] = entry
        self.size += 1
        sift_down(self.heap, 0, self.size - 1)

    def pop(self):
        """(skip_between, first, second) of the next pair of active ids, or None."""
        entry: Entry
        while self.size:
            self.size -= 1
            if self.size:
                entry = self.heap[0]
                self.heap[0] = self.heap[self.size]
                sift_up(self.heap, self.size, 0)
            else:
                entry = self.heap[0]
            if self.active[entry.first] and self.active[entry.second]:
                self.last = entry
                return (entry.skip != 0, entry.first, entry.second)
        return None

    def push_skipped(self):
        """Queue the pair last popped again, to be grouped without the between test."""
        entry: Entry = self.last
        entry.skip = 1
        self.push(entry)

    def merge(
        self,
        first: cython.Py_ssize_t,
        second: cython.Py_ssize_t,
        group: cython.Py_ssize_t,
        bbox: tuple,
    ):
        """Retire first and second, and queue group, whose box is bbox, against the rest."""
        i: cython.Py_ssize_t
        other: cython.Py_ssize_t
        kept: cython.Py_ssize_t
        if group >= self.id_capacity:
            raise IndexError("group id out of range")
        self.active[first] = 0
        self.active[second] = 0
        self.active_count -= 2
        x0, y0, x1, y1 = bbox
        self.set_bounds(group, x0, y0, x1, y1)
        for i in range(self.order_size):
            other = self.order[i]
            if self.active[other]:
                self.push(self.gap(group, other))
        self.active[group] = 1
        self.order[self.order_size] = group
        self.order_size += 1
        self.active_count += 1
        if self.active_count <= self.compact_at:
            # As the Python loop compacts its list: keep live pairs in heap order, heapify.
            if self.size > self.active_count * (self.active_count - 1):
                kept = 0
                for i in range(self.size):
                    if self.active[self.heap[i].first] and self.active[self.heap[i].second]:
                        self.heap[kept] = self.heap[i]
                        kept += 1
                self.size = kept
                heapify(self.heap, self.size)
            self.compact_at = self.active_count // 2
