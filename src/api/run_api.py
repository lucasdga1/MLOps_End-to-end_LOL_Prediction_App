import sys
from pathlib import Path
import os
import uvicorn
from main import app
sys.path.append(str(Path(__file__).resolve().parents[1]))
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)