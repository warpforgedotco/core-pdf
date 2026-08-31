# SPDX-License-Identifier: AGPL-3.0-only
"""Layout heuristics: line grouping, spatial indexing, and geometry quality.

Everything here runs *downstream* of content-stream interpretation and consumes the
capture records in ``model/``. Import from the owning module rather than from
this package; it deliberately re-exports nothing.
"""
