import sys
import unittest
import tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from normalization import normalize, canonical, load_dictionary
from audit_data import make_groups, prepare, grouped_split, validate_splits


class DataProtocolTests(unittest.TestCase):
    def test_dictionary_ambiguity_identity_and_invalid_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"lexicon.csv"
            p.write_text("slang,formal\ngak,tidak\nGAK,bukan\na,b\na,b\nx,x\nmulti word,formal\nempty,\n",encoding="utf-8")
            mapping,stats,ambiguous=load_dictionary(p)
            self.assertEqual(mapping,{"a":"b"})
            self.assertEqual(ambiguous,{"gak":["bukan","tidak"]})
            self.assertEqual(stats["identity_keys_skipped"],1)

    def test_boundaries_unknowns_negation_and_single_pass(self):
        mapping={"gak":"tidak","bgt":"banget","a":"b","b":"c"}
        o,n,changes=normalize("  GAK bagus,bgt! 😃 gaknya a  ",mapping)
        self.assertEqual(n,"tidak bagus,banget! 😃 gaknya b")
        self.assertEqual(len(changes),3)
        self.assertEqual(o[changes[0]["start"]:changes[0]["end"]],"GAK")

    def test_conflicting_originals_excluded_and_test_duplicate_retained(self):
        rows=[dict(id=str(i),text=text,label=label,source_split=split,source_order=i) for i,(text,label,split) in enumerate([
            ("BAIK","positive","train"),("baik","positive","test"),
            ("buruk","negative","train"),("BURUK","positive","test")])]
        kept,excluded=prepare(rows,["negative","positive"],{})
        self.assertEqual([r["id"] for r in kept],["1"])
        self.assertEqual(len(excluded),3)

    def test_cross_form_transitive_group_and_conflicting_collision_preserved(self):
        rows=[dict(id=str(i),original=o,normalized=n,label=label) for i,(o,n,label) in enumerate([
            ("a","b","positive"),("b","c","negative"),("d","c","positive"),("z","z","negative")])]
        groups=make_groups(rows)
        self.assertEqual(sorted(map(len,groups)),[1,3])
        config={"prdec_split":{"ratios":[.7,.15,.15],"seed":42}}
        assigned=grouped_split(groups,["negative","positive"],config)
        self.assertEqual(len(assigned),4)
        validate_splits(assigned)
        assigned[1]["split"]="test" if assigned[0]["split"]!="test" else "train"
        with self.assertRaises(AssertionError): validate_splits(assigned)


if __name__=="__main__":unittest.main()
