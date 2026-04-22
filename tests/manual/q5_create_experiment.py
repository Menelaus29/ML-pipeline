"""Q5: Create an experiment (triggers background notebook generation)."""
import urllib.request, json

ds_id  = json.loads(urllib.request.urlopen("http://localhost:8000/api/datasets/").read())[0]["id"]
cfg_id = json.loads(urllib.request.urlopen(f"http://localhost:8000/api/preprocessing/configs/{ds_id}").read())[0]["id"]

models_config = json.dumps([
    {"name": "LogisticRegression", "parameters": {"C": 1, "max_iter": 200}},
])
payload_bytes = json.dumps({
    "dataset_id": ds_id,
    "preprocessing_config_id": cfg_id,
    "models_config_json": models_config,
}).encode()

req = urllib.request.Request("http://localhost:8000/api/experiments/", data=payload_bytes, method="POST")
req.add_header("Content-Type", "application/json")
exp = json.loads(urllib.request.urlopen(req).read())
print("experiment_id:", exp["id"], "| status:", exp["status"])
