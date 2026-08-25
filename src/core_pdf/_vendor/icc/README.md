# Vendored ICC profiles

## `SWOP2006_Coated5v2.icc`

The built-in DeviceCMYK profile. PDF 32000-1 leaves DeviceCMYK device-dependent
and defines no CMYK-to-RGB conversion, so a renderer has to pick one; every
serious implementation picks a characterized press condition rather than the
uncalibrated ink formula, and this is a SWOP one.

- **Name** (`desc` tag): `SWOP2006_Coated5v2.icc`
- **Copyright** (`cprt` tag): "Copyright X-Rite, Inc.. This profile is made
  available by IDEAlliance, with permission of X-Rite, Inc., and may be used,
  embedded, exchanged, and shared without restriction. It may not be altered,
  or sold without written permission of IDEAlliance."
- **Origin**: <https://www.color.org/registry/profiles/SWOP2006_Coated5v2.icc>
- **License**: permissive, and explicitly allows embedding and redistribution.
  It forbids alteration, so this file is byte-for-byte as published -- do not
  strip the unused `A2B0`/`A2B2`/`B2A0`/`B2A2`/`gamt` tags to save space.
- **Shape**: ICC v2.0, CMYK device space, Lab PCS. `A2B*` are `mft2` tags over a
  17x17x17x17 grid, `B2A*` are `mft2` over 33x33x33. We read `A2B1` and `B2A1`,
  the relative colorimetric pair, because PDF 32000-1 8.6.5.8 makes
  RelativeColorimetric the default rendering intent. The `B2A1` tag is what the
  black point detection in `s_08_graphics/icc_profiles.py` runs against; without
  it, black point compensation has no source black to scale from.
- **SHA-256**: `0ac00fe6f03901bfd06ef70e72ec2c55fa3c043c6c34c0b6d70f06cc7a40a822`

Replacing this file with another CMYK profile needs no code change beyond the
filename in `s_08_graphics/device_profiles.py`, as long as it is an ICC v2
profile whose tables are `mft1`/`mft2`; the v4 `mAB `/`mBA ` tag types are not
parsed. Note that MuPDF and Ghostscript default to a *different* SWOP profile
("Artifex CMYK SWOP Profile", AGPL), so our DeviceCMYK output is close to but
not identical with theirs. Swapping profiles moves rendered output for every
DeviceCMYK page, so regenerate `tests/snapshots/raster/first_page_scale1.json`.
