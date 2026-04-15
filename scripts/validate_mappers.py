"""Validate mappers.json beatsaver_ids against BeatSaver API.

Writes validation results back alongside mappers.json as mappers_validated.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

API_BASE = "https://api.beatsaver.com"
UA = "beatsaber-automapper/0.1.0 (https://github.com/Kwoolford/beatsaber_automapper)"
HEADERS = {"User-Agent": UA}


def fetch_user(user_id: str) -> dict | None:
    """GET /users/id/{id} — returns user object or None on 404."""
    r = requests.get(f"{API_BASE}/users/id/{user_id}", headers=HEADERS, timeout=15)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def fetch_user_maps_count(user_id: str) -> int:
    """Fetch first page of uploader maps; return total docs count."""
    r = requests.get(f"{API_BASE}/maps/uploader/{user_id}/0", headers=HEADERS, timeout=15)
    if r.status_code == 404:
        return 0
    r.raise_for_status()
    data = r.json()
    return len(data.get("docs", []))


def main() -> int:
    root = Path(__file__).parent.parent
    src = root / "data" / "reference" / "mappers.json"
    data = json.loads(src.read_text(encoding="utf-8"))

    results = []
    for m in data["mappers"]:
        name = m["display_name"]
        bid = m["beatsaver_id"]
        stated_uname = m["beatsaver_username"]
        print(f"  checking {name:22s} id={bid} ...", end=" ", flush=True)
        try:
            user = fetch_user(bid)
            time.sleep(0.25)
        except Exception as e:
            print(f"ERROR {e}")
            results.append(
                {"display_name": name, "beatsaver_id": bid, "status": "error", "error": str(e)}
            )
            continue
        if user is None:
            print("NOT FOUND")
            results.append({"display_name": name, "beatsaver_id": bid, "status": "not_found"})
            continue
        api_uname = user.get("name", "")
        uname_match = api_uname.lower() == stated_uname.lower()
        # Peek first page of uploader maps to confirm they have uploads
        try:
            first_page = fetch_user_maps_count(bid)
            time.sleep(0.25)
        except Exception:
            first_page = -1
        print(
            f"OK  api_name={api_uname!r}  match={uname_match}  first_page_maps={first_page}"
        )
        results.append(
            {
                "display_name": name,
                "beatsaver_id": bid,
                "stated_username": stated_uname,
                "api_username": api_uname,
                "username_match": uname_match,
                "first_page_map_count": first_page,
                "status": "ok" if uname_match else "mismatch",
            }
        )

    out = root / "data" / "reference" / "mappers_validation.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")

    ok = sum(1 for r in results if r["status"] == "ok")
    mism = sum(1 for r in results if r["status"] == "mismatch")
    miss = sum(1 for r in results if r["status"] == "not_found")
    err = sum(1 for r in results if r["status"] == "error")
    print(f"Summary: ok={ok} mismatch={mism} not_found={miss} error={err}")
    return 0 if (miss == 0 and err == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
