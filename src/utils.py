import json

def load_jsonl(path):
    # Windows PowerShell `Set-Content -Encoding utf8` often writes a UTF-8 BOM.
    # Using utf-8-sig transparently strips it so json.loads works.
    lines = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for lineno, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                lines.append(json.loads(raw))
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"{e.msg} (in {path} at line {lineno})",
                    e.doc,
                    e.pos,
                ) from None
    return lines

def add_jsonl(new_line, path):
    with open(path, "a", encoding="utf-8") as f:
        json.dump(new_line, f)
        f.write("\n")
