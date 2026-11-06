# Ported from x-ray 0.4.0, Copyright (c) Free Law Project, under the BSD 2-Clause License.
import asyncio

from . import cli


def main() -> None:
    asyncio.run(cli())


if __name__ == "__main__":
    main()
