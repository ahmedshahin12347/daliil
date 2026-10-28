import random, re, statistics, sys, time
import requests
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from main import extract_pages  # نفس الـ cleaning اللي بتستخدمه في الـ ingest

pdf_path, label = sys.argv[1], sys.argv[2]
reverse = len(sys.argv) > 3 and sys.argv[3] == "rev"   # لو كنت رافع بـ reverse_digits=true

pages = extract_pages(pdf_path, open(pdf_path, "rb").read(), reverse)
text = "\n".join(t for _, t in pages)
numbers = sorted({int(n) for n in re.findall(r"مادة \((\d+)\)\s*:", text)})
print(f"Found {len(numbers)} article headings in the PDF")

random.seed(42)  # نفس العينة في كل تشغيل، عشان المقارنة عادلة
sample = random.sample(numbers, min(30, len(numbers)))

requests.post("http://127.0.0.1:8000/ask", json={"question": "ما نص المادة 1؟", "top_k": 4})  # warm-up
hit1 = hit4 = 0
times = []
for n in sample:
    t = time.perf_counter()
    r = requests.post("http://127.0.0.1:8000/ask",
                      json={"question": f"ما نص المادة {n}؟", "top_k": 4}).json()
    times.append(time.perf_counter() - t)
    heading = re.compile(rf"مادة \({n}\)\s*:")
    found = [bool(heading.search(s["text"])) for s in r["sources"]]
    hit4 += any(found)
    hit1 += bool(found and found[0])

times.sort()
k = len(sample)
print(f"[{label}] Hit@1: {hit1}/{k} ({hit1/k:.0%}) | Hit@4: {hit4}/{k} ({hit4/k:.0%})")
print(f"[{label}] avg: {statistics.mean(times):.2f}s | p95: {times[int(0.95*(k-1))]:.2f}s")