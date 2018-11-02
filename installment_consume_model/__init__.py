import os
import sys
project_root_dir = os.path.dirname(os.path.realpath(__file__))
if not project_root_dir in sys.path:
    sys.path.append(project_root_dir)