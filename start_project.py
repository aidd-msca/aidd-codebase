import sys
import os 
import shutil 
from abstract_codebase.directories import validate_or_create_dir

def start(name: str ="project"):
    # Get the current working directory
    pkgdir = sys.modules['aidd_codebase'].__path__[0]
    src = os.path.join(pkgdir, "new_project")
    dest = os.path.join(os.getcwd(), name)
    
    # Check if the project directory already exists
    if os.path.exists(dest):
        raise FileExistsError("Project directory already exists")
    
    # Copy the content of 
    # source to destination 
    shutil.copytree(src, dest) 

if __name__ == "__main__":
    # find name in command line arguments
    for i in range(1, len(sys.argv)):
        if sys.argv[i] == "--name":
            name = sys.argv[i+1]
            start(name)

    # start()