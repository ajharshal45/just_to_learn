"""
Launch script for SHAP-SMOTE Research Platform FastAPI Backend.
Starts uvicorn on http://127.0.0.1:8000
"""

import sys
from pathlib import Path
import uvicorn

# Ensure repository root is in python path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    print("=" * 70)
    print("Starting SHAP-SMOTE Research Web Application Backend")
    print("API documentation: http://127.0.0.1:8000/docs")
    print("API health check : http://127.0.0.1:8000/api/health")
    print("=" * 70)
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=False)
