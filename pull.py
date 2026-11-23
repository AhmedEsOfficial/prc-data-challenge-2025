"""Download the PRC Data Challenge 2025 datasets from the OpenSky S3 bucket.

Credentials are issued to each registered team by the challenge organisers.
Provide them through the environment — never commit them:

    export OPENSKY_ACCESS_KEY=...
    export OPENSKY_SECRET_KEY=...
    python pull.py
"""

import os
import sys

from minio import Minio

BUCKET = "prc-2025-datasets"


def main() -> None:
    access_key = os.environ.get("OPENSKY_ACCESS_KEY")
    secret_key = os.environ.get("OPENSKY_SECRET_KEY")
    if not access_key or not secret_key:
        sys.exit("Set OPENSKY_ACCESS_KEY and OPENSKY_SECRET_KEY before running pull.py")

    client = Minio(
        os.environ.get("OPENSKY_S3_ENDPOINT", "s3.opensky-network.org"),
        access_key=access_key,
        secret_key=secret_key,
        secure=True,
    )

    for obj in client.list_objects(BUCKET, recursive=True):
        print(f"Downloading: {obj.object_name}")
        client.fget_object(BUCKET, obj.object_name, obj.object_name)


if __name__ == "__main__":
    main()
