import os

import polygraphy
from tests.tools.common import run_polygraphy


class TestPolygraphyBin(object):
    def test_version(self):
        status = run_polygraphy(["-v"])
        assert status.stdout.strip().replace("\n", " ").replace(
            "  ", " "
        ) == "Polygraphy | Version: {:} | Path: {:}".format(
            polygraphy.__version__, list(map(os.path.realpath, polygraphy.__path__))
        )
