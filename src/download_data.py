"""Download ca-HepPh dataset (SNAP)."""
import os, urllib.request, gzip

URL = "https://snap.stanford.edu/data/ca-HepPh.txt.gz"
RAW = os.path.join(os.path.dirname(__file__), "../data/raw/ca-HepPh.txt.gz")

def main():
    os.makedirs(os.path.dirname(RAW), exist_ok=True)
    if os.path.exists(RAW):
        print(f"Already exists: {RAW}")
        return
    print(f"Downloading {URL} ...")
    urllib.request.urlretrieve(URL, RAW)
    print(f"Saved to {RAW}")

if __name__ == "__main__":
    main()
