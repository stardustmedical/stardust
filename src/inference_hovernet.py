import sys
import site
sys.path.extend(site.getsitepackages())

import hover_net

if __name__ == "__main__":
    print("hover_net version: ", hover_net.__version__)