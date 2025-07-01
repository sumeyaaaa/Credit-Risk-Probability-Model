import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path

# Set tracking URI
mlruns_path = Path(
    r"C:\Users\ABC\Desktop\10Acadamy\Week 5\Credit-Risk-"
    r"Probability-Model\mlruns"
).absolute()
mlflow.set_tracking_uri(f"file:///{mlruns_path.as_posix()}")

# Define model variables
model_name = "best_model"
run_id = "1bed56713a694528a9571bb00576059c"
artifact_path = "models"
model_uri = f"runs:/{run_id}/{artifact_path}"

client = MlflowClient()

# Register the model (will raise exception if already exists)
try:
    client.create_registered_model(model_name)
except Exception:
    pass  # model already exists

# Create new version
mv = client.create_model_version(
    name=model_name,
    source=model_uri,
    run_id=run_id
)

# Transition to Staging
client.transition_model_version_stage(
    name=model_name,
    version=mv.version,
    stage="Staging"
)

print(f"✅ Re-registered as models:/{model_name}/Staging")
