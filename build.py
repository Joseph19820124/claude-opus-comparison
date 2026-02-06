#!/usr/bin/env python3
"""Build the static site by embedding benchmark data into index.html."""

import json
import re

# Read results
with open("/home/ec2-user/opus-comparison/benchmark_results.json") as f:
    data = json.load(f)

# Strip raw_json from results to reduce size
for r in data["results"]:
    r.pop("raw_json", None)

# Read template
with open("/home/ec2-user/opus-comparison/index.html") as f:
    html = f.read()

# Embed data — replace existing const DATA = ...; or BENCHMARK_DATA_PLACEHOLDER
json_str = json.dumps(data, indent=None)

# Try replacing existing embedded data first
pattern = r'const DATA = \{.*?\};'
match = re.search(pattern, html, re.DOTALL)
if match:
    html = html[:match.start()] + f'const DATA = {json_str};' + html[match.end():]
else:
    # Fall back to placeholder
    html = html.replace("BENCHMARK_DATA_PLACEHOLDER", json_str, 1)

# Write final output
with open("/home/ec2-user/opus-comparison/index.html", "w") as f:
    f.write(html)

print(f"Built index.html with {len(data['results'])} results ({len(json_str)} bytes of data)")
