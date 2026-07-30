"""Hugging Face Space bootstrap for Lipsync.

Paste this into a Space as `app.py`. It is deliberately tiny — small enough to
type into the web editor from a phone — because it does not contain the app. It
clones the real one from GitHub at startup, so the Space never needs the ~60
vendored files uploaded by hand, and it picks up changes on every restart.

Nothing here needs editing unless you forked the repository.
"""

import os
import subprocess
import sys

REPO = "https://github.com/TridentIntelFree/Lipsync-.git"
BRANCHES = ["claude/lipsync-speech-inference-k649jo", "main"]
CHECKOUT = "/home/user/app/_lipsync"

if not os.path.isdir(CHECKOUT):
    for branch in BRANCHES:
        done = subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", branch, REPO, CHECKOUT],
            capture_output=True,
            text=True,
        )
        if done.returncode == 0:
            print(f"Cloned {REPO} @ {branch}")
            break
    else:
        raise SystemExit(f"Could not clone {REPO} from any of {BRANCHES}")

sys.path.insert(0, CHECKOUT)

# Spaces with persistent storage keep the ~1 GB model between restarts. Without
# it the model is re-fetched on each cold start, which still works.
os.environ.setdefault(
    "LIPSYNC_CACHE", "/data/lipsync" if os.path.isdir("/data") else "/tmp/lipsync-cache"
)

import app as lipsync_app  # noqa: E402

demo = lipsync_app.build()

if __name__ == "__main__":
    demo.launch()
