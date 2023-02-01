import os
import shutil
import sys


# python -c "from aidd_codebase.start_project import start; start('name', 'dir_path')"
def start(name: str = "project", dest: str = os.getcwd()):
    # Get the current working directory
    pkgdir = sys.modules["aidd_codebase"].__path__[0]  # type: ignore
    src = os.path.join(pkgdir, "new_project")
    dest = os.path.join(dest, name)

    # Check if the project directory already exists
    if os.path.exists(dest):
        raise FileExistsError("Project directory already exists")

    # Copy the content of
    # source to destination
    shutil.copytree(src, dest)
    
    
if __name__ == "__main__":
    start(*sys.argv[1:])
