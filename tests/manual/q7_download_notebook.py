"""Q7: Download the generated notebook for the first experiment."""
import urllib.request, json

exp_id = json.loads(urllib.request.urlopen("http://localhost:8000/api/experiments/").read())[0]["id"]
r = urllib.request.urlopen(f"http://localhost:8000/api/experiments/{exp_id}/notebook")
content = r.read()
print("notebook download status:", r.status, "| bytes:", len(content))
assert r.status == 200 and len(content) > 100, "Notebook download failed"
print("PASS: notebook downloaded OK")
