"""Q6: Wait then check experiment notebook_generated status."""
import urllib.request, json, time

print("Waiting 5 seconds for background notebook generation...")
time.sleep(5)

exps = json.loads(urllib.request.urlopen("http://localhost:8000/api/experiments/").read())
if not exps:
    print("FAIL: no experiments found — run q5_create_experiment.py first")
else:
    exp = exps[0]
    print("status     :", exp["status"])
    print("notebook   :", exp.get("notebook_path"))
    assert exp["status"] == "notebook_generated", f"Expected notebook_generated, got {exp['status']}"
    assert exp.get("notebook_path"), "notebook_path should not be None"
    print("PASS: notebook generation confirmed")
