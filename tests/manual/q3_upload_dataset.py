"""Q3: Upload a test CSV and print the dataset_id."""
import urllib.request, json

boundary = "BOUNDARY_Q3"
csv_data = b"age,salary,label\n25,50000,yes\n30,60000,no\n35,70000,yes\n40,80000,no\n"
body = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="file"; filename="test.csv"\r\n'
    f"Content-Type: text/csv\r\n\r\n"
).encode() + csv_data + f"\r\n--{boundary}--\r\n".encode()

req = urllib.request.Request("http://localhost:8000/api/datasets/upload", data=body, method="POST")
req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
d = json.loads(urllib.request.urlopen(req).read())
print("dataset_id:", d["id"], "| rows:", d["row_count"])
