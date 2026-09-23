import unittest
from pathlib import Path

from model_contract import (
    DEFAULT_MODEL_NAMES,
    normalize_class_name,
    read_model_metadata_names,
    validate_cell_model_contract,
)


class ModelContractTests(unittest.TestCase):
    def test_bundled_openvino_labels_match_the_app(self):
        validate_cell_model_contract("segment", DEFAULT_MODEL_NAMES)

    def test_included_openvino_files_and_metadata_match_the_contract(self):
        repo_root = Path(__file__).resolve().parents[1]
        model_dir = repo_root / "models" / "best_openvino_model"
        metadata = (model_dir / "metadata.yaml").read_text(encoding="utf-8")
        self.assertTrue(list(model_dir.glob("*.xml")))
        self.assertTrue(list(model_dir.glob("*.bin")))
        self.assertIn("task: segment", metadata)
        metadata_names = read_model_metadata_names(model_dir)
        validate_cell_model_contract("segment", metadata_names)
        self.assertEqual(metadata_names, DEFAULT_MODEL_NAMES)

    def test_swapped_class_order_is_valid_and_kept_semantically_distinct(self):
        names = {0: "celula_sana", 1: "celula_enferma"}
        validate_cell_model_contract("segment", names)
        self.assertEqual(normalize_class_name(names[0]), "sana")
        self.assertEqual(normalize_class_name(names[1]), "enferma")

    def test_common_english_labels_are_supported(self):
        validate_cell_model_contract("segment", {0: "diseased cell", 1: "healthy cell"})

    def test_detection_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "se requiere segmentación"):
            validate_cell_model_contract("detect", DEFAULT_MODEL_NAMES)

    def test_missing_or_unknown_class_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactamente las clases"):
            validate_cell_model_contract("segment", {0: "unknown", 1: "celula_sana"})


if __name__ == "__main__":
    unittest.main()
