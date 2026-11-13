import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from docintel.chunking import STRATEGIES, fixed_window, is_relevant, paragraph  # noqa: E402
from docintel.data import FIELDS, generate_docs, generate_queries  # noqa: E402
from docintel.evaluation import extraction_eval, retrieval_eval  # noqa: E402
from docintel.extraction import LLMExtractor, extract_v1, extract_v2  # noqa: E402
from docintel.pipeline import QAPipeline, classify_field  # noqa: E402
from docintel.retrieval import BM25, CharNgramTfidf, Hybrid  # noqa: E402
from docintel.service import Service, docs_from_directory  # noqa: E402
from docintel.textutil import num_to_words, parse_date, parse_money, words_to_num  # noqa: E402


class TextUtil(unittest.TestCase):
    def test_number_words_roundtrip(self):
        for n in (1, 12, 20, 45, 60, 90, 99):
            self.assertEqual(words_to_num(num_to_words(n)), n)

    def test_dates(self):
        self.assertEqual(parse_date("March 3, 2024"), "2024-03-03")
        self.assertEqual(parse_date("3 March 2024"), "2024-03-03")
        self.assertEqual(parse_date("2024-03-03"), "2024-03-03")
        self.assertIsNone(parse_date("someday"))

    def test_money(self):
        self.assertEqual(parse_money("$250,000"), 250000)
        self.assertEqual(parse_money("1,000,000 US dollars"), 1000000)


class Data(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(generate_docs(5, 1)[2].text, generate_docs(5, 1)[2].text)

    def test_gold_and_evidence_consistent(self):
        for d in generate_docs(10, 3):
            for f in FIELDS:
                s, e = d.evidence[f]
                self.assertTrue(0 <= s < e <= len(d.text))

    def test_unique_entities(self):
        docs = generate_docs(60, 7)
        names = [d.client for d in docs] + [d.vendor for d in docs]
        self.assertEqual(len(names), len(set(names)))


class Chunking(unittest.TestCase):
    def test_offsets_match_text(self):
        d = generate_docs(1, 1)[0]
        for name, fn in STRATEGIES.items():
            for c in fn(d):
                self.assertEqual(d.text[c.start:c.end].strip(), c.text.strip(), name)

    def test_window_overlap_creates_more_chunks(self):
        d = generate_docs(1, 1)[0]
        self.assertGreater(len(fixed_window(d, 50, 20)), len(fixed_window(d, 50, 0)))

    def test_relevance(self):
        d = generate_docs(1, 1)[0]
        ch = paragraph(d)[1]
        self.assertTrue(is_relevant(ch, d.doc_id, (ch.start, ch.end)))
        self.assertFalse(is_relevant(ch, "other-doc", (ch.start, ch.end)))


class Retrieval(unittest.TestCase):
    def test_bm25_finds_distinctive_term(self):
        r = BM25(["alpha beta", "gamma delta", "zebra crossing"])
        self.assertEqual(r.search("zebra", 1)[0][0], 2)

    def test_char_ngram_handles_morphology(self):
        r = CharNgramTfidf(["terminating the contract", "unrelated payment clause"])
        self.assertEqual(r.search("termination", 1)[0][0], 0)

    def test_hybrid_returns_k(self):
        texts = ["alpha beta", "gamma delta", "zebra crossing"]
        self.assertLessEqual(len(Hybrid([BM25(texts), CharNgramTfidf(texts)]).search("zebra", 2)), 2)

    def test_contextual_header_helps(self):
        docs = generate_docs(30, 5)
        qs = generate_queries(docs)
        plain = retrieval_eval(docs, qs, "fixed-50", "bm25", False)["recall@5"]
        ctx = retrieval_eval(docs, qs, "fixed-50", "bm25", True)["recall@5"]
        self.assertGreater(ctx, plain + 0.2)


class Extraction(unittest.TestCase):
    def test_v2_handles_formats_and_distractors(self):
        text = ("Client may audit records upon 5 days' written notice.\n"
                "Either party may terminate this Agreement upon forty-five (45) days' written notice.\n"
                "A party in material breach has 30 days to cure after written notice before the other party may terminate.\n"
                "This Agreement shall remain in force for two (2) years from the Effective Date.\n"
                "Provider shall maintain insurance coverage of at least $2,000,000.\n"
                "In no event shall aggregate liability exceed USD 250,000.\n"
                "The laws of England and Wales shall govern this Agreement.\n"
                "Overdue invoices accrue a late fee of 1.5% per month after 10 days.\n"
                "Client shall pay each invoice net 30.\n"
                "This Agreement is entered into as of 3 March 2024 between A and B.")
        out = extract_v2(text)
        self.assertEqual(out, {"effective_date": "2024-03-03", "term_months": 24, "payment_days": 30,
                               "notice_days": 45, "governing_law": "England and Wales",
                               "liability_cap": 250000})

    def test_v1_is_fooled_by_distractor(self):
        text = "Client may audit records upon 5 days' written notice. Either party may terminate upon 30 days' written notice."
        self.assertEqual(extract_v1(text)["notice_days"], 5)   # the known weakness v2 fixes
        self.assertEqual(extract_v2(text)["notice_days"], 30)

    def test_v2_beats_v1_on_standard_set(self):
        docs = generate_docs(30, 7)
        self.assertGreater(extraction_eval(docs, extract_v2)["accuracy"],
                           extraction_eval(docs, extract_v1)["accuracy"] + 0.3)

    def test_llm_parse_is_tolerant(self):
        raw = 'Sure! ```json\n{"effective_date": "2024-03-03", "term_months": 12, "payment_days": "30", ' \
              '"notice_days": null, "governing_law": "Texas.", "liability_cap": "250,000"}\n```'
        out = LLMExtractor.parse(raw)
        self.assertEqual(out["payment_days"], 30)
        self.assertEqual(out["governing_law"], "Texas")
        self.assertEqual(out["liability_cap"], 250000)
        self.assertIsNone(out["notice_days"])

    def test_llm_parse_rejects_garbage(self):
        self.assertTrue(all(v is None for v in LLMExtractor.parse("I cannot help with that").values()))
        self.assertTrue(all(v is None for v in LLMExtractor.parse("{not json}").values()))

    def test_llm_extractor_with_fake_backend(self):
        fake = lambda prompt: '{"term_months": 6}'  # noqa: E731
        self.assertEqual(LLMExtractor(complete=fake)("any text")["term_months"], 6)


class Pipeline(unittest.TestCase):
    def test_classifier_regression_company_name_contains_cap(self):
        # Bug found by the harness: substring match on 'cap' misrouted 'Falcon Capital' questions.
        self.assertEqual(classify_field("What is the effective date of the Falcon Capital agreement?"),
                         "effective_date")
        self.assertEqual(classify_field("What is the liability cap in the Acme Labs agreement?"), "liability_cap")

    def test_classifier_covers_all_generated_queries(self):
        qs = generate_queries(generate_docs(30, 7))
        self.assertTrue(all(classify_field(q.text) == q.field for q in qs))

    def test_answers_with_citation_in_correct_doc(self):
        docs = generate_docs(20, 7)
        pipe = QAPipeline(docs)
        d = docs[3]
        a = pipe.ask(f"What notice period applies to terminating the {d.client} agreement?")
        self.assertEqual(a.value, d.gold["notice_days"])
        self.assertEqual(a.citations[0]["doc_id"], d.doc_id)

    def test_abstains_on_unknown_question(self):
        a = QAPipeline(generate_docs(5, 7)).ask("What is the meaning of life?")
        self.assertTrue(a.abstained)

    def test_entity_scoping_prevents_wrong_doc_answers(self):
        docs = generate_docs(60, 7)
        qs = generate_queries(docs)[:120]
        by = {d.doc_id: d for d in docs}
        scoped = QAPipeline(docs, scope_by_entity=True)
        for q in qs:
            a = scoped.ask(q.text)
            if not a.abstained:
                self.assertEqual(a.citations[0]["doc_id"], q.doc_id)

    def test_docs_without_entity_names_do_not_match_everything(self):
        with tempfile.TemporaryDirectory() as tmp:
            (pathlib.Path(tmp) / "a.txt").write_text("Either party may terminate upon 30 days' written notice.")
            (pathlib.Path(tmp) / "b.txt").write_text("Either party may terminate upon 60 days' written notice.")
            pipe = QAPipeline(docs_from_directory(tmp))
            self.assertIsNone(pipe.resolve_doc("What notice period applies?"))


class ServiceTests(unittest.TestCase):
    def test_service_roundtrip(self):
        svc = Service(generate_docs(10, 7))
        self.assertEqual(svc.extract("The initial term of this Agreement is 12 months from the Effective Date.")["term_months"], 12)
        self.assertIn("value", svc.ask("What is the liability cap in the x agreement?"))
        with self.assertRaises(ValueError):
            svc.extract("x", method="nope")


if __name__ == "__main__":
    unittest.main()


class Citations(unittest.TestCase):
    def test_citation_is_the_supporting_sentence_with_exact_offsets(self):
        docs = generate_docs(20, 7)
        pipe = QAPipeline(docs)
        d = docs[5]
        a = pipe.ask(f"What is the liability cap in the {d.client} agreement?")
        c = a.citations[0]
        self.assertEqual(d.text[c["start"]:c["end"]], c["text"])
        self.assertLess(len(c["text"]), 140)                       # a sentence, not a page
        self.assertIn("liability", c["text"].lower())
        s, e = d.evidence["liability_cap"]
        self.assertTrue(s <= c["start"] and c["end"] <= e)        # inside the gold clause
