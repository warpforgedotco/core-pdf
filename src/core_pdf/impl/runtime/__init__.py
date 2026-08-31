# SPDX-License-Identifier: AGPL-3.0-only
"""Engine-independent runtime infrastructure: buffers, caches, and bounded execution.

Nothing here may import from ``spec/`` or the derived-processing packages beside it;
these modules sit beneath the whole engine so those packages can depend on them downward.
"""
