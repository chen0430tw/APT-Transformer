import os, json

def build_tree(path):
    tree = {}
    for name in os.listdir(path):
        if name in [".git", "__pycache__", ".venv"]:
            continue
        full = os.path.join(path, name)
        if os.path.isdir(full):
            tree[name] = build_tree(full)
        else:
            tree[name] = "file"
    return tree

index = build_tree(".")
with open("repo_index.json", "w", encoding="utf-8") as f:
    json.dump(index, f, indent=2, ensure_ascii=False)

print("repo_index.json generated.")
