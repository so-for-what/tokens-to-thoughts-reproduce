#!/usr/bin/env python3
"""
Push files to GitHub via API.
Usage: python3 push_to_github.py <commit_message> <path1> [<path2> ...]
Paths relative to /tmp/ttt/.
Requires the token in /tmp/ttt/.github_token
"""
import base64, json, os, sys, subprocess

TOKEN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".github_token")
if os.path.exists(TOKEN_FILE):
    TOKEN = open(TOKEN_FILE).read().strip()
else:
    TOKEN = os.environ.get("GITHUB_TOKEN", "")

if not TOKEN:
    print("ERROR: No GitHub token found. Set GITHUB_TOKEN env or create .github_token")
    sys.exit(1)

API = "https://api.github.com/repos/so-for-what/tokens-to-thoughts-reproduce"

def _api(method, ep, data=None):
    cmd = [
        "curl", "-s", "--noproxy", "*", "--max-time", "60",
        "-X", method,
        "-H", f"Authorization: token {TOKEN}",
        "-H", "Accept: application/vnd.github.v3+json",
        "-H", "Content-Type: application/json",
    ]
    if data:
        cmd += ["--data-raw", json.dumps(data)]
    cmd += [f"{API}{ep}"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    out = r.stdout.strip()
    if not out:
        return {}
    return json.loads(out)

def push(files, message):
    """files: {github_path: local_path, ...}"""
    ref = _api("GET", "/git/refs/heads/main")
    latest = ref["object"]["sha"]
    commit = _api("GET", f"/git/commits/{latest}")
    tree_data = _api("GET", f"/git/trees/{commit['tree']['sha']}?recursive=1")
    entries = [e for e in tree_data["tree"] if e["type"] != "tree"]

    new_blobs = {}
    for gp, lp in files.items():
        if not os.path.exists(lp):
            print(f"  SKIP {gp} (not found)")
            continue
        with open(lp, "rb") as f:
            ct = f.read()
        result = _api("POST", "/git/blobs", {
            "content": base64.b64encode(ct).decode(),
            "encoding": "base64"
        })
        sha = result.get("sha")
        if not sha:
            print(f"  FAIL {gp}: {json.dumps(result)[:200]}")
            continue
        new_blobs[gp] = sha
        print(f"  OK  {gp}: {len(ct)/1024:.0f}KB")

    if not new_blobs:
        print("  Nothing to push")
        return False

    tree_items = [
        {"path": e["path"], "mode": e["mode"], "type": e["type"], "sha": e["sha"]}
        for e in entries
    ]
    for p, s in new_blobs.items():
        tree_items = [e for e in tree_items if e["path"] != p]
        tree_items.append({"path": p, "mode": "100644", "type": "blob", "sha": s})

    new_tree = _api("POST", "/git/trees", {
        "base_tree": commit["tree"]["sha"], "tree": tree_items
    })
    new_commit = _api("POST", "/git/commits", {
        "message": message,
        "tree": new_tree["sha"],
        "parents": [latest],
    })
    result = _api("PATCH", "/git/refs/heads/main", {"sha": new_commit["sha"], "force": False})
    print(f"\nPushed: {result['object']['sha'][:12]} ({len(new_blobs)} files)")
    return True

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 2:
        print("Usage: push_to_github.py <message> <path1> [<path2> ...]")
        sys.exit(1)
    message = args[0]
    files = {}
    for p in args[1:]:
        local = os.path.join("/tmp/ttt", p)
        if os.path.exists(local):
            files[p] = local
        else:
            print(f"  SKIP {p} (not found)")
    if files:
        push(files, message)
    else:
        print("  Nothing to push")