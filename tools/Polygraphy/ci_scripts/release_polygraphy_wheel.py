#!/usr/bin/env python3
# Get a Kitmaker API token by logging in to https://kitmaker-portal.nvidia.com
# with NVIDIA SAML, opening your profile settings, selecting API Tokens, and
# creating a new token. Copy it immediately; the portal will not show it again.

"""
Submit the staged Polygraphy wheel to Kitmaker for PyPI release.

Dry run, which runs Kitmaker checks without publishing:
    python3 ci_scripts/release_polygraphy_wheel.py --token "$KITMAKER_API_TOKEN"

Publish after the dry run passes:
    python3 ci_scripts/release_polygraphy_wheel.py --token "$KITMAKER_API_TOKEN" --upload
"""

import argparse
import ast
import json
import posixpath
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


ARTIFACTORY_DOWNLOAD_ROOT = "https://urm.nvidia.com/artifactory/sw-tensorrt-pypi"
ARTIFACTORY_STORAGE_ROOT = (
    "https://urm.nvidia.com/artifactory/api/storage/sw-tensorrt-pypi"
)
KITMAKER_PORTAL_API_BASE = "https://kitmaker-portal.nvidia.com/api/v0"
PIC_EMAIL = "pranavm@nvidia.com"
POLL_INTERVAL_SECONDS = 30
PROJECT_NAME = "polygraphy"
VERSION_FILE = "polygraphy/__init__.py"


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--token",
        required=True,
        help="Kitmaker API token.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Publish after checks pass. Omit this flag for a dry run.",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification if not enrolled with NVIDIA's IT CA.",
    )
    return parser.parse_args()


def request_json(method, url, token=None, body=None, insecure=False):
    data = None if body is None else json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    context = None
    if insecure:
        import ssl

        context = ssl._create_unverified_context()

    try:
        with urllib.request.urlopen(request, context=context, timeout=60) as response:
            response_body = response.read().decode("utf-8")
            return json.loads(response_body) if response_body else {}
    except urllib.error.HTTPError as err:
        response_body = err.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"HTTP request failed with HTTP {err.code}: {response_body}"
        ) from err
    except urllib.error.URLError as err:
        raise RuntimeError(f"HTTP request failed: {err}") from err


def join_url(base_url, *parts):
    parsed = urllib.parse.urlparse(base_url)
    path_parts = [parsed.path.rstrip("/")]
    path_parts.extend(urllib.parse.quote(str(part).strip("/")) for part in parts)
    path = posixpath.join(*path_parts)
    return urllib.parse.urlunparse(parsed._replace(path=path))


def infer_version():
    with open(VERSION_FILE, "r", encoding="utf-8") as version_file:
        module = ast.parse(version_file.read(), filename=VERSION_FILE)

    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Name)
                and target.id == "__version__"
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                return node.value.value

    raise RuntimeError(f"Could not infer __version__ from {VERSION_FILE}")


def normalize_project_name(name):
    return name.replace("_", "-").lower()


def extract_projects(response):
    if isinstance(response, list):
        return response
    if not isinstance(response, dict):
        raise RuntimeError(f"Unexpected projects response: {response!r}")

    for key in ("projects", "items", "data", "results"):
        projects = response.get(key)
        if isinstance(projects, list):
            return projects

    raise RuntimeError(f"Could not find project list in response: {response!r}")


def get_project_id(token, insecure):
    response = request_json(
        "GET", f"{KITMAKER_PORTAL_API_BASE}/projects", token=token, insecure=insecure
    )
    matches = [
        project
        for project in extract_projects(response)
        if normalize_project_name(project.get("name", "")) == PROJECT_NAME
    ]

    if not matches:
        raise RuntimeError(f"Could not find Kitmaker project {PROJECT_NAME!r}")
    if len(matches) > 1:
        raise RuntimeError(f"Found multiple Kitmaker projects named {PROJECT_NAME!r}")

    project_id = matches[0].get("id")
    if project_id is None:
        raise RuntimeError(f"Kitmaker project {PROJECT_NAME!r} has no id field")

    return str(project_id)


def get_staged_wheel_urls(version, insecure):
    storage_url = join_url(ARTIFACTORY_STORAGE_ROOT, PROJECT_NAME, version)
    download_url = join_url(ARTIFACTORY_DOWNLOAD_ROOT, PROJECT_NAME, version)

    response = request_json("GET", storage_url, insecure=insecure)
    wheel_urls = []
    for child in response.get("children", []):
        if child.get("folder", True):
            continue
        name = child.get("uri", "").lstrip("/")
        if name.endswith(".whl"):
            wheel_urls.append(join_url(download_url, name))

    if not wheel_urls:
        raise RuntimeError(f"No staged wheel files found at {download_url}")

    return wheel_urls


def create_release(token, project_id, wheel_urls, upload, insecure):
    release_jobs = [
        {
            "pic": PIC_EMAIL,
            "job_type": "wheel-release-job",
            "url": wheel_url,
            "upload": upload,
        }
        for wheel_url in wheel_urls
    ]
    body = {"project_name": PROJECT_NAME, "payload": release_jobs}
    url = f"{KITMAKER_PORTAL_API_BASE}/projects/{project_id}/releases"
    return request_json("POST", url, token=token, body=body, insecure=insecure)


def get_release_status(token, release_uuid, insecure):
    url = f"{KITMAKER_PORTAL_API_BASE}/status/{release_uuid}"
    return request_json("GET", url, token=token, insecure=insecure)


def wait_for_completion(token, release_uuid, insecure):
    while True:
        status_response = get_release_status(token, release_uuid, insecure)
        print(json.dumps(status_response, indent=2, sort_keys=True))

        status = status_response.get("status")
        if status not in {"pending", "building"}:
            return status_response

        time.sleep(POLL_INTERVAL_SECONDS)


def main():
    args = parse_args()

    try:
        version = infer_version()
        project_id = get_project_id(args.token, args.insecure)
        wheel_urls = get_staged_wheel_urls(version, args.insecure)

        print(f"Project: {PROJECT_NAME} ({project_id})", file=sys.stderr)
        print(f"Version: {version}", file=sys.stderr)
        print(f"Upload: {args.upload}", file=sys.stderr)
        for wheel_url in wheel_urls:
            print(f"Wheel: {wheel_url}", file=sys.stderr)

        response = create_release(
            args.token, project_id, wheel_urls, args.upload, args.insecure
        )
        print(json.dumps(response, indent=2, sort_keys=True))

        release_uuid = response.get("release_uuid")
        if not release_uuid:
            print("ERROR: Response did not include release_uuid", file=sys.stderr)
            return 1

        final_status = wait_for_completion(args.token, release_uuid, args.insecure)
        return 0 if final_status.get("status") == "completed" else 1
    except RuntimeError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
