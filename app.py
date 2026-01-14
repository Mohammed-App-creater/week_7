import sys
from pathlib import Path

# Add project root to path (though if running from root, it usually is)
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import the streamlit app from ui/app.py
# This will execute the code in ui/app.py since it has top-level Streamlit calls
import ui.app
