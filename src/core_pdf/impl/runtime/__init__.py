# SPDX-License-Identifier: AGPL-3.0-only
"""Engine-independent runtime infrastructure: buffers, caches, and bounded execution.

Nothing here may import from ``capture_model/``, ``engine/`` or ``spec/``; these
modules sit beneath the whole implementation so every higher layer can depend on
them downward.
"""
