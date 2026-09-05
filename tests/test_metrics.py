import csv
import io
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics import (lexical, evaluate_pair, checks, diff_html, parse_csv, results_csv, aggregate)


class MetricsTests(unittest.TestCase):
    def test_identical_korean_is_full_overlap(self):
        r = lexical("보고서의 근거가 부족합니다.", "보고서의 근거가 부족합니다.")
        self.assertAlmostEqual(r["bleu"], 100)
        self.assertAlmostEqual(r["chrf"], 100)
        self.assertEqual(r["rouge_l"], 1)

    def test_unicode_rouge_does_not_drop_hangul(self):
        self.assertAlmostEqual(lexical("가 나 다", "가 나 라")["rouge_l"], 2/3)

    def test_unrelated_text_has_zero_overlap(self):
        r = lexical("사과 바나나", "로봇 우주")
        self.assertEqual(r["bleu"], 0)
        self.assertEqual(r["rouge_l"], 0)

    def test_empty_and_punctuation_inputs_rejected(self):
        for text in ("", "  ", "!!!"):
            with self.assertRaises(ValueError):
                evaluate_pair("문장", text)

    def test_numeric_and_negation_changes_flagged(self):
        r = checks("나는 3개 구매에 동의하지 않는다.", "나는 5개 구매에 동의한다.", "구매")
        self.assertEqual(len(r["messages"]), 2)
        self.assertEqual(r["keyword_retention"], 1)

    def test_keyword_denominator_excludes_absent_terms(self):
        r = checks("보고서 근거", "근거", "보고서, 근거, 사과")
        self.assertEqual(r["keyword_retention"], .5)
        self.assertEqual(r["keywords_not_in_source"], ["사과"])

    def test_unavailable_models_remain_missing(self):
        r = evaluate_pair("원문 내용", "수정된 내용")
        self.assertIsNone(r["sim"])
        self.assertIsNone(r["j_proxy"])
        self.assertEqual(set(r["errors"]), {"sim", "toxicity", "ppl"})

    def test_legacy_csv_extra_column_ignored_in_evaluation_and_export(self):
        data = 'source,candidate,reference,keywords\n거친 말,부드러운 말,예전 기준문,말\n'.encode("utf-8")
        row = parse_csv(data)[0]
        self.assertEqual(set(row), {"source", "candidate", "keywords"})
        r = evaluate_pair(**row, use_sim=False, use_toxicity=False, use_ppl=False)
        self.assertAlmostEqual(r["source_overlap"]["bleu"], lexical(row["source"], row["candidate"])["bleu"])
        self.assertNotIn("reference", r)
        self.assertNotIn("reference_overlap", r)
        exported = next(csv.DictReader(io.StringIO(results_csv([r]).decode("utf-8-sig"))))
        self.assertFalse(any(key.startswith("reference") for key in exported))
        self.assertFalse(any(key.startswith("reference") for key in aggregate([r])))

    def test_csv_encoding_and_quotes(self):
        data = 'source,candidate\n"원문, 내용",순화문\n'.encode("utf-8-sig")
        self.assertEqual(parse_csv(data)[0]["source"], "원문, 내용")
        with self.assertRaises(ValueError):
            parse_csv(b"wrong,header\na,b")

    def test_csv_formula_protection(self):
        r = {"source": "=HYPERLINK(1)", "candidate": "안녕", "errors": {}}
        row = next(csv.DictReader(io.StringIO(results_csv([r]).decode("utf-8-sig"))))
        self.assertTrue(row["source"].startswith("'="))

    def test_html_user_text_is_escaped(self):
        rendered = diff_html("<script>alert(1)</script>", "문장")
        self.assertNotIn("<script>", rendered)
        self.assertIn("&lt;", rendered)

    def test_joint_aggregated_per_sentence_not_product_of_means(self):
        r = aggregate([{"sim": .5, "sta": 1, "j_proxy": .1}, {"sim": 1, "sta": 0, "j_proxy": 0}, {"errors": {"input": "empty"}}])
        self.assertEqual(r["j_proxy_macro_mean"], .05)
        self.assertEqual(r["sta_valid_n"], 2)
        self.assertEqual(r["total_rows"], 3)


if __name__ == "__main__":
    unittest.main()
