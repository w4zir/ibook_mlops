#!/usr/bin/env python
"""List MLflow model versions for a registered model. Usage: python scripts/mlflow_list_model_versions.py [MODEL_NAME]"""
import os
import sys

# Optional: use MLFLOW_TRACKING_URI env or default to local server
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")

from mlflow.tracking import MlflowClient

def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "fraud_detection"
    client = MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{name}'")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    if not versions:
        print(f"No versions found for model '{name}'.")
        return
    print(f"Model '{name}' ({len(versions)} version(s)):")
    for mv in sorted(versions, key=lambda m: int(m.version), reverse=True):
        print(f"  version {mv.version}  stage={mv.current_stage or 'None'}  run_id={mv.run_id}")

if __name__ == "__main__":
    main()
