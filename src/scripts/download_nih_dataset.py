import argparse
import os
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from tqdm import tqdm

LINKS = (
    "https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz",
    "https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz",
    "https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz",
    "https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz",
    "https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz",
    "https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz",
    "https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz",
    "https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz",
    "https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz",
    "https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz",
    "https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz",
    "https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz",
)

METADATA_LINK = "https://nihcc.app.box.com/v/ChestXray-NIHCC/file/219760887468"

def download_and_extract(args):
    idx, url, out_dir, pbar = args
    arc_name = f"images_{idx+1:02d}.tar.gz"
    arc_path = out_dir / arc_name

    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            block_size = 1024

            download_pbar = tqdm(
                total=total_size,
                unit='iB',
                unit_scale=True,
                desc=f"Downloading {arc_name}",
                leave=False
            )
            with open(arc_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=block_size):
                    download_pbar.update(len(chunk))
                    f.write(chunk)
            download_pbar.close()

        with tarfile.open(arc_path, "r:gz") as tar:
            tar.extractall(path=out_dir)

        os.remove(arc_path)
    finally:
        pbar.update(1)

import json

def main():
    with open('config.jsonc') as f:
        config = json.load(f)['download']

    parser = argparse.ArgumentParser(description="Download and extract the NIH Chest X-ray dataset.")
    parser.add_argument("--out-dir", type=str, default=config['out_dir'], help="Path to the directory to save the dataset.")
    parser.add_argument("--workers", type=int, default=config['workers'], help="Number of worker threads to use for downloading.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Downloading and extracting {len(LINKS)} archives into {out_dir} using {args.workers} workers...")

    with tqdm(total=len(LINKS), desc="Overall Progress") as pbar:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            tasks = [(i, url, out_dir, pbar) for i, url in enumerate(LINKS)]
            executor.map(download_and_extract, tasks)

    print("\n[INFO] All archives fetched and unpacked.")
    print("[INFO] Please manually download the metadata file (Data_Entry_2017.csv).")
    print(f"[INFO] You can find it here: {METADATA_LINK}")

if __name__ == "__main__":
    main()
