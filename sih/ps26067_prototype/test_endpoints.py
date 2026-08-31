"""Quick integration tests for the OceanViz backend."""

import requests

BASE = "http://localhost:8765"

def test():
    r = requests.get(f"{BASE}/api/variables", timeout=10)
    assert r.status_code == 200
    data = r.json()
    assert "variables" in data
    print(f"variables: {len(data['variables'])} variables, {len(data['depths'])} depths, {len(data['times'])} times")

    r = requests.get(f"{BASE}/api/instruments", timeout=10)
    assert r.status_code == 200
    print(f"instruments: {len(r.json()['instruments'])}")

    r = requests.get(f"{BASE}/api/slice?variable=temperature&depth=0&time=0", timeout=30)
    assert r.status_code == 200
    print(f"slice: min={r.json()['min']}, max={r.json()['max']}")

    r = requests.get(f"{BASE}/api/image?variable=temperature&depth=0&time=0&width=512&height=512", timeout=60)
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    print(f"image: {len(r.content)} bytes")

    r = requests.get(f"{BASE}/api/colorbar?variable=salinity", timeout=30)
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    print(f"colorbar: {len(r.content)} bytes")

    r = requests.get(f"{BASE}/api/vectors?depth=0&time=0", timeout=30)
    assert r.status_code == 200
    print(f"vectors: {r.json()['count']}")

    r = requests.get(f"{BASE}/api/profile/argo_1?variable=temperature&time=0", timeout=10)
    assert r.status_code == 200
    print(f"profile: {r.json()['instrument_id']} has {len(r.json()['profile'])} points")

    r = requests.get(f"{BASE}/", timeout=10)
    assert r.status_code == 200 and "Cesium" in r.text
    print("index.html loads and references Cesium")

    print("\nAll endpoint tests passed.")


if __name__ == "__main__":
    test()
