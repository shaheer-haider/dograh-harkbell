"""Harkbell: one storage instance per backend for the life of the process.

Building a fresh S3 filesystem per request built a fresh aioboto3 session,
whose first client parses the S3 service model on the event loop. With three
calls on the one-vCPU engine box that took 1.2 s, and every other call on the
box lost that much audio (copy runs 6660, 6668, 6669).
"""

from api.enums import StorageBackend
from api.services import storage


def test_each_backend_is_built_once_and_then_reused(monkeypatch):
    built: list[str] = []

    def build(backend: str):
        built.append(backend)
        return object()

    monkeypatch.setattr(storage, "_storage_by_backend", {})
    monkeypatch.setattr(storage, "_build_storage_for_backend", build)

    s3 = StorageBackend.S3.value
    minio = StorageBackend.MINIO.value
    first = storage.get_storage_for_backend(s3)
    assert storage.get_storage_for_backend(s3) is first
    assert storage.get_storage_for_backend(minio) is not first
    assert storage.get_storage_for_backend(minio) is storage.get_storage_for_backend(minio)
    assert built == [s3, minio]
