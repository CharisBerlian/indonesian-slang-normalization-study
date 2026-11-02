"""Audit public resources, construct reproducible splits, and measure input changes.

No classifiers are fitted and no test predictions are produced. Test labels are
used only for deterministic conflict checks and descriptive class counts.
"""
from collections import Counter, defaultdict
from pathlib import Path
import csv
import hashlib
import json
import platform
import sys

from tokenizers import BertWordPieceTokenizer, __version__ as tokenizers_version
from normalization import canonical, cleanup, load_dictionary, normalize, word_tokens

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
OUT = ROOT / "data/processed"
REPORT = ROOT / "reports"
SPLITS = ("train", "valid", "test")
PRIORITY = {"test":0,"valid":1,"train":2,"unassigned":3}


def sha(value):
    return hashlib.sha256(value if isinstance(value,bytes) else value.encode()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")


def read_rows(dataset):
    result=[]
    if dataset == "prdec":
        with (RAW / "prdec.csv").open(encoding="utf-8-sig",newline="") as f:
            for i,row in enumerate(csv.DictReader(f)):
                result.append(dict(id=f"prdec:{i+1}",text=row["Customer Review"],label=row["Sentiment"].strip().lower(),source_split="unassigned",source_order=i))
    else:
        for split in SPLITS:
            # The release uses one text/label pair per line, without a header.
            for i,line in enumerate((RAW/f"smsa_{split}.tsv").read_text(encoding="utf-8").splitlines()):
                text,label=line.rsplit("\t",1)
                result.append(dict(id=f"smsa:{split}:{i+1}",text=text,label=label.strip().lower(),source_split=split,source_order=i))
    return result


def prepare(rows, labels, mapping):
    excluded=[]
    buckets=defaultdict(list)
    for row in rows:
        if not cleanup(row["text"]) or row["label"] not in labels:
            excluded.append(dict(id=row["id"],source_split=row["source_split"],reason="missing_text_or_invalid_label"))
        else:
            buckets[canonical(row["text"])].append(row)
    kept=[]
    for copies in buckets.values():
        copies.sort(key=lambda r:(PRIORITY[r["source_split"]],r["source_order"]))
        conflicting=len({r["label"] for r in copies})>1
        for i,row in enumerate(copies):
            if conflicting or i>0:
                excluded.append(dict(id=row["id"],source_split=row["source_split"],reason="conflicting_original_labels" if conflicting else "canonical_original_duplicate"))
            else:
                original,normalized,changes=normalize(row.pop("text"),mapping)
                kept.append(dict(row,original=original,normalized=normalized,replacements=changes))
    return kept,excluded


def make_groups(rows):
    parent=list(range(len(rows)))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    seen={}
    for i,row in enumerate(rows):
        for key in {canonical(row["original"]),canonical(row["normalized"])}:
            if key in seen:
                a,b=root(i),root(seen[key])
                parent[max(a,b)]=min(a,b)
            else: seen[key]=i
    groups=defaultdict(list)
    for i,row in enumerate(rows): groups[root(i)].append(row)
    for group in groups.values():
        gid=sha("\n".join(sorted(r["id"] for r in group)))[:20]
        for row in group: row["group_id"]=gid
    return list(groups.values())


def grouped_split(groups, labels, config):
    ratios=config["prdec_split"]["ratios"]
    seed=config["prdec_split"]["seed"]
    total=Counter(r["label"] for g in groups for r in g)
    n=sum(total.values())
    counts={s:Counter() for s in SPLITS}
    sizes=Counter()
    ordered=sorted(groups,key=lambda g:(-len(g),sha(f'{seed}:{g[0]["group_id"]}')))
    for group in ordered:
        gc=Counter(r["label"] for r in group)
        scores=[]
        for candidate in SPLITS:
            score=0.0
            for split,ratio in zip(SPLITS,ratios):
                for label in labels:
                    target=ratio*total[label]
                    actual=counts[split][label]+(gc[label] if split==candidate else 0)
                    score+=(actual-target)**2/max(target,1)
                actual=sizes[split]+(len(group) if split==candidate else 0)
                score+=0.25*(actual-ratio*n)**2/(ratio*n)
            scores.append(score)
        chosen=SPLITS[min(range(3),key=lambda i:(scores[i],i))]
        counts[chosen].update(gc)
        sizes[chosen]+=len(group)
        for row in group:row["split"]=chosen
    return [r for g in groups for r in g]


def validate_splits(rows):
    owners={}
    for row in rows:
        for key in [row["group_id"],canonical(row["original"]),canonical(row["normalized"])]:
            if key in owners and owners[key]!=row["split"]:
                raise AssertionError("Cross-split original/normalized overlap")
            owners[key]=row["split"]


def audit(dataset, mapping, tokenizer, config):
    raw=read_rows(dataset)
    labels=["negative","positive"] if dataset=="prdec" else ["negative","neutral","positive"]
    released=Counter(r["source_split"] for r in raw)
    rows,excluded=prepare(raw,labels,mapping)
    groups=make_groups(rows)
    before_group_count=len(groups)
    if dataset=="prdec": rows=grouped_split(groups,labels,config)
    else:
        rows=[]
        for group in groups:
            preferred=min((r["source_split"] for r in group),key=PRIORITY.get)
            for row in group:
                if row["source_split"]==preferred:
                    row["split"]=preferred
                    rows.append(row)
                else:
                    excluded.append(dict(id=row["id"],source_split=row["source_split"],reason="cross_partition_equivalence",group_id=row["group_id"]))
    rows.sort(key=lambda r:(SPLITS.index(r["split"]),r["source_order"],r["id"]))
    validate_splits(rows)
    lengths={}
    input_changes={}
    for r in rows:
        enc=[tokenizer.encode(r[k]).ids for k in ["original","normalized"]]
        lengths[r["id"]]=[len(x) for x in enc]
        input_changes[r["id"]]=enc[0]!=enc[1]
    train=[r for r in rows if r["split"]=="train"]
    length=512
    for candidate in config["sequence_lengths"]:
        if all(sum(lengths[r["id"]][i]<=candidate for r in train)/len(train)>=config["training_length_coverage"] for i in [0,1]):
            length=candidate
            break
    tokenizer.enable_truncation(max_length=length)
    stats={}
    for split in SPLITS:
        subset=[r for r in rows if r["split"]==split]
        changed=[r for r in subset if r["original"]!=r["normalized"]]
        trunc_changes=0
        for r in subset:
            ids=[tokenizer.encode(r[k]).ids for k in ["original","normalized"]]
            r["bert_input_changed"]=ids[0]!=ids[1]
            trunc_changes+=r["bert_input_changed"]
            r["original_token_length"],r["normalized_token_length"]=lengths[r["id"]]
        stats[split]={
            "reviews":len(subset),"groups":len({r["group_id"] for r in subset}),
            "classes":dict(Counter(r["label"] for r in subset)),
            "changed_reviews":len(changed),"changed_classes":dict(Counter(r["label"] for r in changed)),
            "unchanged_classes":dict(Counter(r["label"] for r in subset if r["original"]==r["normalized"])),
            "replacement_tokens":sum(len(r["replacements"]) for r in subset),
            "word_token_sequence_changed":sum(word_tokens(r["original"])!=word_tokens(r["normalized"]) for r in subset),
            "casefold_character_sequence_changed":sum(r["original"].casefold()!=r["normalized"].casefold() for r in subset),
            "bert_untruncated_ids_changed":sum(input_changes[r["id"]] for r in subset),
            "bert_truncated_ids_changed":trunc_changes,
            "truncated_original":sum(lengths[r["id"]][0]>length for r in subset),
            "truncated_normalized":sum(lengths[r["id"]][1]>length for r in subset),
            "max_group_size":max(Counter(r["group_id"] for r in subset).values(),default=0)
        }
    tokenizer.no_truncation()
    norm_groups=defaultdict(list)
    for row in rows:norm_groups[canonical(row["normalized"])].append(row)
    collisions=[v for v in norm_groups.values() if len(v)>1]
    summary={"release_rows":len(raw),"release_partitions":dict(released),"retained_rows":len(rows),
             "groups_before_partition_filter":before_group_count,"exclusions":dict(Counter(x["reason"] for x in excluded)),
             "exclusions_by_source_partition":dict(Counter(x["source_split"] for x in excluded)),
             "selected_max_length":length,"splits":stats,"normalized_collision_groups":len(collisions),
             "normalized_collision_conflicting_label_groups":sum(len({r["label"] for r in g})>1 for g in collisions),
             "cross_split_original_or_normalized_overlap":0}
    with (OUT/f"{dataset}.jsonl").open("w",encoding="utf-8",newline="\n") as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n")
    with (ROOT/f"data/splits/{dataset}.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["id","source_split","split","group_id","label"])
        writer.writeheader()
        writer.writerows({k:r[k] for k in writer.fieldnames} for r in rows)
    dump(REPORT/f"{dataset}_exclusions.json",excluded)
    return summary


def main():
    for p in [OUT,REPORT,ROOT/"data/splits"]:p.mkdir(parents=True,exist_ok=True)
    config=json.loads((ROOT/"configs/protocol.json").read_text())
    manifest=json.loads((ROOT/"data/manifest.json").read_text())
    for entry in manifest["files"]:
        assert sha((RAW/entry["path"]).read_bytes())==entry["sha256"],entry["path"]
    mapping,dictstats,ambiguous=load_dictionary(RAW/"alay.csv")
    dump(OUT/"normalization_mapping.json",mapping)
    dump(OUT/"ambiguous_dictionary_keys.json",ambiguous)
    tokenizer=BertWordPieceTokenizer(str(RAW/"bert_vocab.txt"),lowercase=True,strip_accents=True,clean_text=True,handle_chinese_chars=True)
    tokenizer.save(str(OUT/"tokenizer.json"))
    summary={"dictionary":dictstats,"datasets":{},"environment":{"python":sys.version,"platform":platform.platform(),"tokenizers":tokenizers_version}}
    for dataset in ["prdec","smsa"]:
        print("Auditing",dataset,flush=True)
        summary["datasets"][dataset]=audit(dataset,mapping,tokenizer,config)
    dump(REPORT/"audit_summary.json",summary)
    fingerprint_paths=["configs/protocol.json","configs/resources.json","src/normalization.py","src/audit_data.py","data/processed/normalization_mapping.json","data/processed/tokenizer.json","data/splits/prdec.csv","data/splits/smsa.csv"]
    hashes={p:sha((ROOT/p).read_bytes()) for p in fingerprint_paths}
    hashes.update({"data/raw/"+f["path"]:f["sha256"] for f in manifest["files"]})
    dump(ROOT/"data/protocol_lock.json",{"protocol_id":sha(json.dumps(hashes,sort_keys=True)),"files":hashes,"status":"Data and preprocessing audit complete; no classifier training or test predictions"})
    lines=["# Data and protocol audit", "", "Protocol established 2026-11-01. No classifier training or test predictions. This is a data audit, not a performance result.","", "## Dictionary",""]
    lines.extend(f"- {k}: {v:,}" for k,v in dictstats.items())
    for name,d in summary["datasets"].items():
        display_name="PRDECT-ID" if name=="prdec" else "SmSA"
        lines.extend(["",f"## {display_name}","",f"Released rows: {d['release_rows']:,}; retained: {d['retained_rows']:,}. Selected IndoBERT maximum length: {d['selected_max_length']} (training data only).",f"Exclusions: `{json.dumps(d['exclusions'])}`.","","| Split | Reviews | Groups | Classes | Text changed | BERT input changed after truncation | Truncated O / N |","|---|---:|---:|---|---:|---:|---|"])
        for split,s in d["splits"].items():
            lines.append(f"| {split} | {s['reviews']} | {s['groups']} | {s['classes']} | {s['changed_reviews']} | {s['bert_truncated_ids_changed']} | {s['truncated_original']} / {s['truncated_normalized']} |")
        lines.extend(["",f"Normalized collision groups retained: {d['normalized_collision_groups']}; groups with differing labels: {d['normalized_collision_conflicting_label_groups']}. Cross-partition original/normalized overlap: 0."])
        lines.append(f"Changed test-review class counts: {d['splits']['test']['changed_classes']}.")
    lines.extend(["","## Interpretation and remaining checks","",
        "- Counts reflect the frozen dictionary policy, not independently verified slang annotations. Original and normalized versions share exactly the same retained rows and split assignments.",
        "- Word and character change counts are before training-vocabulary filtering. They do not yet establish a TF-IDF vector change; the training implementation must log that diagnostic after fitting training-only vocabularies.",
        "- BERT input-change counts use the pinned vocabulary and the serialized lowercase BERT WordPiece tokenizer, including truncation and special tokens. Training must reuse this tokenizer or verify exact token-ID equivalence.",
        "- Labels were used for conflict checks, group-level stratification of PRDECT-ID, and descriptive class counts, not tuning or prediction evaluation. No manual test-example inspection informed the dictionary.",
        "- The dictionary, grouping code, tokenizer and assignments are jointly fingerprinted in data/protocol_lock.json. Any change requires re-audit before training; after test evaluation, protocol changes require disclosure and a new evaluation design.",
        "- PRDECT-ID: source declares CC BY 4.0. IndoNLU: repository includes Apache-2.0; retain original dataset attribution and verify any dataset-specific restrictions before redistribution. Kamus Alay: no explicit license file in the pinned repository; its README invites research use, but redistribution permission is not established. Raw/derived text and dictionary mappings stay local and are excluded by .gitignore.",
        "- This audit establishes usable inputs, not statistical power. Inspect class support and interval widths before drawing subgroup or no-effect conclusions.",
        "- No full training was launched. Hyperparameters remain proposed; model outcomes and computation costs are unknown.",""])
    (REPORT/"data_audit.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
