"""Deploy the public demo to HuggingFace Spaces.

Uploads only what the demo needs (code + gallery + Dockerfile), never
the raw corpora, .env, logs or chat history. Also configures the Space's
secrets and variables.

Usage (from project root, with HF_TOKEN and GOOGLE_API_KEY in .env):
    .venv/bin/python scripts/deploy_space.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")
os.environ.pop("HF_ENDPOINT", None)  # official endpoint for authenticated ops

from huggingface_hub import HfApi

SPACE_ID = os.getenv("SPACE_ID", "vae01/multiagent-investment-debate")
KB_DATASET = os.getenv("KB_DATASET", "vae01/investment-debate-kb")

ALLOW_PATTERNS = [
    "main.py",
    "llm_provider.py",
    "web_search.py",
    "requirements.txt",
    "Dockerfile",
    "LICENSE",
    "rag/*.py",
    "web/*.py",
    "web/gallery/*.json",
]

# Non-secret runtime configuration
VARIABLES = {
    "DEMO_MODE": "1",
    "DEMO_LLM_MODEL": "gemini-2.5-flash",
    "DEMO_DAILY_DEBATE_LIMIT": "30",
    "DEMO_SESSION_DEBATE_LIMIT": "2",
    "DEMO_AGENT_SEARCH_CAP": "1",
    "LLM_PROVIDER": "gemini",
    "LLM_FAST_MODEL": "gemini-2.5-flash",
    "WEB_SEARCH_MODEL": "gemini-2.5-flash",
    "ENABLE_AGENT_SEARCH": "1",
    "EMBEDDING_PROVIDER": "local",
    "KB_DATASET": KB_DATASET,
}


def main() -> None:
    token = os.environ["HF_TOKEN"]
    google_key = os.environ["GOOGLE_API_KEY"]
    api = HfApi(token=token)

    print(f"== Deploying to {SPACE_ID} ==")
    api.create_repo(SPACE_ID, repo_type="space", space_sdk="docker",
                    private=False, exist_ok=True)

    print("-- setting variables --")
    for k, v in VARIABLES.items():
        api.add_space_variable(SPACE_ID, k, v)

    print("-- setting secrets (GOOGLE_API_KEY, HF_TOKEN) --")
    api.add_space_secret(SPACE_ID, "GOOGLE_API_KEY", google_key)
    # Read access to the private KB dataset (build-time + runtime fetch)
    api.add_space_secret(SPACE_ID, "HF_TOKEN", token)

    print("-- uploading Space README --")
    api.upload_file(
        path_or_fileobj=str(ROOT / "deploy" / "space_README.md"),
        path_in_repo="README.md",
        repo_id=SPACE_ID,
        repo_type="space",
        commit_message="Space README",
    )

    print("-- uploading code --")
    api.upload_folder(
        folder_path=str(ROOT),
        repo_id=SPACE_ID,
        repo_type="space",
        allow_patterns=ALLOW_PATTERNS,
        commit_message="Deploy demo code",
    )

    print(f"DEPLOY_OK → https://huggingface.co/spaces/{SPACE_ID}")


if __name__ == "__main__":
    main()
