import tempfile
import unittest
from pathlib import Path

from model_artifacts import ModelArtifact, sha256_file, verify_model_artifact


class ModelArtifactTests(unittest.TestCase):
    def test_artifact_size_and_sha256_are_verified(self):
        content = b"small model fixture"
        artifact = ModelArtifact(
            kind="fixture",
            display_name="fixture",
            filename="fixture.bin",
            url="https://example.invalid/fixture.bin",
            size_bytes=len(content),
            sha256="0fecb51f65f96420ff8ff743d637cf1017f877c8002646ea208ccb8297d44eae",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / artifact.filename
            path.write_bytes(content)
            self.assertEqual(sha256_file(path), artifact.sha256)
            self.assertEqual(verify_model_artifact(path, artifact), artifact.sha256)

    def test_wrong_artifact_digest_is_rejected(self):
        artifact = ModelArtifact(
            kind="fixture",
            display_name="fixture",
            filename="fixture.bin",
            url="https://example.invalid/fixture.bin",
            size_bytes=3,
            sha256="0" * 64,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / artifact.filename
            path.write_bytes(b"abc")
            with self.assertRaisesRegex(ValueError, "SHA-256 no coincide"):
                verify_model_artifact(path, artifact)

    def test_wrong_artifact_size_is_rejected_before_hashing(self):
        artifact = ModelArtifact(
            kind="fixture",
            display_name="fixture",
            filename="fixture.bin",
            url="https://example.invalid/fixture.bin",
            size_bytes=4,
            sha256="0" * 64,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / artifact.filename
            path.write_bytes(b"abc")
            with self.assertRaisesRegex(ValueError, "Tamaño incorrecto"):
                verify_model_artifact(path, artifact)


if __name__ == "__main__":
    unittest.main()
