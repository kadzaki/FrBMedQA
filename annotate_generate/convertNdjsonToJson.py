import json

file = open('../articles_v2.ndjson', 'r', encoding="utf8")
ndjson_content = file.read()

result = []

for ndjson_line in ndjson_content.splitlines():
    if not ndjson_line.strip():
        continue  # ignore empty lines
    json_line = json.loads(ndjson_line)
    result.append(json_line)

with open('../articles_v2.json', 'w', encoding="utf8") as outfile:
    json.dump(result, outfile)
