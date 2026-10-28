import unittest
from pathlib import Path

from model_contract import (
    DEFAULT_MODEL_NAMES,
    build_class_id_map,
    normalize_class_name,
    read_model_metadata_names,
    select_mask2former_candidates,
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

    def test_rfdetr_class_mapping_preserves_reversed_label_order(self):
        self.assertEqual(
            build_class_id_map(["celula_sana", "celula_enferma"]),
            {0: 1, 1: 0},
        )

    def test_rfdetr_class_mapping_rejects_unknown_names(self):
        with self.assertRaisesRegex(ValueError, "exactamente las clases"):
            build_class_id_map(["celula_enferma", "objeto"])

    def test_mask2former_uses_at_most_one_label_for_each_query(self):
        candidates = select_mask2former_candidates(
            class_probabilities=[[0.43, 0.42], [0.2, 0.8], [0.5, 0.5]],
            mask_confidences=[0.95, 0.9, 0.2],
            confidence_threshold=0.35,
        )
        self.assertEqual([query_id for query_id, _, _ in candidates], [0, 1])
        self.assertEqual([class_id for _, class_id, _ in candidates], [0, 1])

    def test_mask2former_applies_threshold_after_mask_confidence(self):
        candidates = select_mask2former_candidates(
            class_probabilities=[[0.8, 0.1], [0.7, 0.2]],
            mask_confidences=[0.9, 0.2],
            confidence_threshold=0.35,
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0][:2], (0, 0))
        self.assertAlmostEqual(candidates[0][2], 0.72)

    def test_mask2former_candidate_inputs_must_align(self):
        with self.assertRaisesRegex(ValueError, "una confianza de máscara"):
            select_mask2former_candidates([[0.8, 0.2]], [], 0.35)


if __name__ == "__main__":
    unittest.main()
