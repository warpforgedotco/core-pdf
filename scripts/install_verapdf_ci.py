# SPDX-License-Identifier: AGPL-3.0-only
"""Install the checksum-pinned CI engine into a new private directory.

This maintenance script is never imported or run by core-pdf-validate.
Installer format: https://docs.verapdf.org/install/#automated-installation
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

internal_VERSION = "1.30.2"
internal_URL = "https://software.verapdf.org/rel/1.30/verapdf-greenfield-1.30.2-installer.zip"
# Downloaded from the exact official release URL above and checked on 2026-09-11.
internal_SHA256 = "6cc6341cb1af644044054b81f00a6590a7918abb18f762243de115258bcad838"


def internal_configuration(destination: Path) -> bytes:
    root = ElementTree.Element("AutomatedInstallation", langpack="eng")
    prefix = "com.izforge.izpack.panels."
    ElementTree.SubElement(root, prefix + "htmlhello.HTMLHelloPanel", id="welcome")
    target = ElementTree.SubElement(root, prefix + "target.TargetPanel", id="install_dir")
    ElementTree.SubElement(target, "installpath").text = str(destination)
    packs = ElementTree.SubElement(root, prefix + "packs.PacksPanel", id="sdk_pack_select")
    for index, name in enumerate(
        (
            "veraPDF GUI",
            "veraPDF Mac and *nix Scripts",
            "veraPDF Validation model",
            "veraPDF Documentation",
            "veraPDF Sample Plugins",
        )
    ):
        # The GUI pack also contains the CLI binaries; the other packs are optional.
        ElementTree.SubElement(
            packs, "pack", index=str(index), name=name, selected=str(index < 2).lower()
        )
    ElementTree.SubElement(root, prefix + "install.InstallPanel", id="install")
    ElementTree.SubElement(root, prefix + "finish.FinishPanel", id="finish")
    return ElementTree.tostring(root, encoding="utf-8", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="new installation directory")
    parser.add_argument(
        "--java", default="java", help="installed Java executable (CI uses Java 17)"
    )
    parser.add_argument(
        "--archive", type=Path, help="reuse a local installer; checksum still required"
    )
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination.exists():
        parser.error("destination must not already exist; the installer may overwrite files")

    with tempfile.TemporaryDirectory(prefix="core-pdf-verapdf-install-") as temporary:
        work = Path(temporary)
        archive = args.archive
        if archive is None:
            archive = work / "installer.zip"
            with (
                urllib.request.urlopen(internal_URL, timeout=120) as response,
                archive.open("wb") as output,
            ):
                shutil.copyfileobj(response, output)
        with archive.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        if digest != internal_SHA256:
            raise ValueError(f"veraPDF installer SHA256 mismatch: {digest}")

        jar_name = f"verapdf-izpack-installer-{internal_VERSION}.jar"
        jar = work / jar_name
        with ZipFile(archive) as package:
            jar.write_bytes(package.read(f"verapdf-greenfield-{internal_VERSION}/{jar_name}"))
        configuration = work / "install.xml"
        configuration.write_bytes(internal_configuration(destination))
        destination.mkdir(parents=True)
        subprocess.run(
            [
                args.java,
                f"-Duser.home={destination}",
                "-Djava.awt.headless=true",
                "-jar",
                str(jar),
                str(configuration),
            ],
            check=True,
            timeout=120,
        )
        executable = destination / "verapdf"
        if not executable.is_file():
            raise RuntimeError("veraPDF installer did not produce the CLI executable")
        print(f"Installed veraPDF {internal_VERSION}: {executable}")


if __name__ == "__main__":
    main()
