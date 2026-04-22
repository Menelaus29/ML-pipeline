"""Q4: Create a preprocessing config for the first dataset."""
import urllib.request, json

ds_id = json.loads(urllib.request.urlopen("http://localhost:8000/api/datasets/").read())[0]["id"]
print("Using dataset_id:", ds_id)

config_payload = {
    "dataset_id": ds_id,
    "label": "test-config",
    "config": {
        "columns": {
            "age":    {"type": "numerical",   "strategy": "standardize", "imputation": "none", "imputation_fill_value": None, "is_target": False},
            "salary": {"type": "numerical",   "strategy": "normalize",   "imputation": "none", "imputation_fill_value": None, "is_target": False},
            "label":  {"type": "categorical", "strategy": "label",       "imputation": "none", "imputation_fill_value": None, "is_target": True},
        },
        "outlier_treatment": {"method": "none", "threshold": 1.5},
        "feature_selection": {"method": "none", "k": 10, "score_func": "f_classif", "variance_threshold": 0.0},
        "class_balancing":   {"method": "none"},
    },
}
payload_bytes = json.dumps(config_payload).encode()
req = urllib.request.Request("http://localhost:8000/api/preprocessing/configs", data=payload_bytes, method="POST")
req.add_header("Content-Type", "application/json")
cfg = json.loads(urllib.request.urlopen(req).read())
print("config_id:", cfg["id"])
