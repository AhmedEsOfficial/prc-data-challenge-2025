from minio import Minio

# Initialize client
client = Minio(
    "s3.opensky-network.org",
    access_key="REDACTED",
    secret_key="REDACTED",
    secure=True
)

# List and download objects
objects = client.list_objects("prc-2025-datasets", recursive=True)
for obj in objects:
    print(f"Downloading: {obj.object_name}")
    client.fget_object("prc-2025-datasets", obj.object_name, obj.object_name)