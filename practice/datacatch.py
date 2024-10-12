import json

with open("data01.json", "r", encoding="UTF-8") as data:
    json_data = json.load(data)

contents = {
    "reply": []
}
for comment_info in json_data["result"]["commentList"]:
    content = comment_info["contents"]
    if content != "":
        contents["reply"].insert(len(contents["reply"]), content)

with open("made_data01.json", "w", encoding="UTF-8") as data:
    json.dump(contents, data, indent=4, ensure_ascii=False)
