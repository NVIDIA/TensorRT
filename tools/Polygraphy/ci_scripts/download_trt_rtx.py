#!/usr/bin/env python3
"""
Download the latest TensorRT-RTX build from NVIDIA Artifactory.

Usage:
    python3 ci_scripts/download_trt_rtx.py [dest_dir]

Required environment variables:
    ARTIFACTORY_USERNAME     - Artifactory username
    AWS_ARTIFACTORY_PASSWORD - Artifactory password/token
"""

import os
import sys

import requests

ARTIFACTORY_BASE = "https://artifactory.nvidia.com/artifactory"
REPO_PATH = "sw-tensorrt-generic-local/cicd/main/L1_Nightly"
FILE_PREFIX = "trt_build_x86_64_rocky8_cuda"
FILE_SUFFIX = "_full_optimized_winjit_cmake.tar"
NUM_VERSIONS_TO_TRY = 10


def get_auth():
    user = os.environ.get("ARTIFACTORY_USERNAME")
    password = os.environ.get("AWS_ARTIFACTORY_PASSWORD")
    if not user or not password:
        print(
            "ERROR: ARTIFACTORY_USERNAME and AWS_ARTIFACTORY_PASSWORD must be set",
            file=sys.stderr,
        )
        sys.exit(1)
    return (user, password)


def list_recent_versions(auth):
    url = f"{ARTIFACTORY_BASE}/api/storage/{REPO_PATH}"
    resp = requests.get(url, auth=auth, timeout=30)
    resp.raise_for_status()
    children = resp.json().get("children", [])
    versions = sorted(
        [
            int(c["uri"].strip("/"))
            for c in children
            if c.get("folder") and c["uri"].strip("/").isdigit()
        ],
        reverse=True,
    )
    return versions[:NUM_VERSIONS_TO_TRY]


def find_artifact(version, auth):
    url = f"{ARTIFACTORY_BASE}/api/storage/{REPO_PATH}/{version}"
    resp = requests.get(url, auth=auth, timeout=30)
    if not resp.ok:
        return None
    for child in resp.json().get("children", []):
        name = child["uri"].strip("/")
        if name.startswith(FILE_PREFIX) and name.endswith(FILE_SUFFIX):
            return name
    return None


def download(version, file_name, auth, dest_dir):
    url = f"{ARTIFACTORY_BASE}/{REPO_PATH}/{version}/{file_name}"
    dest = os.path.join(dest_dir, file_name)
    print(f"Downloading: {url}")
    with requests.get(url, auth=auth, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
    print(f"Saved to: {dest}")


def main():
    dest_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(dest_dir, exist_ok=True)

    auth = get_auth()

    print("Listing recent nightly versions...")
    versions = list_recent_versions(auth)
    if not versions:
        print("ERROR: No versions found at Artifactory path", file=sys.stderr)
        sys.exit(1)
    print(f"Checking versions: {versions}")

    for version in versions:
        print(f"Checking version {version}...")
        file_name = find_artifact(version, auth)
        if not file_name:
            print(f"  No matching artifact found")
            continue
        print(f"  Found: {file_name}")
        try:
            download(version, file_name, auth, dest_dir)
            print(f"Successfully downloaded TensorRT-RTX from version {version}")
            return
        except requests.HTTPError as e:
            print(f"  Download failed: {e}")

    print(
        f"ERROR: No TensorRT-RTX build found in recent versions: {versions}",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
