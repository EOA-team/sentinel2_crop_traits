# `name` is the name of the package as used for `pip install package`
name = "rtm_inv"
# `path` is the name of the package for `import package`
path = name.lower().replace("-", "_").replace(" ", "_")
# Your version number should follow https://python.org/dev/peps/pep-0440 and
# https://semver.org
# version = "0.1.dev0"
author = (
    "Lukas Valentin Graf (Crop Science, ETH Zurich and EOA-Team Agroscope Reckenholz, Zurich, Switzerland)"
)
author_email = ""
description = "Python library for radiative transfer model inversion"  # One-liner
url = "https://github.com/EOA-team/rtm_inv"  # your project home-page
license = "GNU General Public License version 3"  # See https://choosealicense.com
version = "0.1"
