import os
import shutil
import sys


def start(name: str = "project"):
    # Get the current working directory
    pkgdir = sys.modules["aidd_codebase"].__path__[0]  # type: ignore
    src = os.path.join(pkgdir, "new_project")
    dest = os.path.join(os.getcwd(), name)

    # Check if the project directory already exists
    if os.path.exists(dest):
        raise FileExistsError("Project directory already exists")

    # Copy the content of
    # source to destination
    shutil.copytree(src, dest)
