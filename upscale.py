"""Entry point that arc-upscale runs. The code is in the arc_upscale package."""

import sys

from arc_upscale.cli import main

if __name__ == "__main__":
    sys.exit(main())
