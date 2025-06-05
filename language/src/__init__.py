import os
from .params import HF_HOME

# Setup environment variables
os.environ["HF_HOME"] = str(HF_HOME)
