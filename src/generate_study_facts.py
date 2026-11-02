"""Generate aggregate-only dataset, selection and vocabulary-survival facts."""
import hashlib
import json
from pathlib import Path
from train_svm_pilot import ROOT, vector_diagnostic
from infer_svm_final import verify_selection, load_heldout
from normalization import word_tokens

OUTPUT=ROOT/'results/final_evaluation/study_facts.json'
REPORT=ROOT/'reports/study_facts.md'


def main():
    selection_hash,manifest,selected,models=verify_selection()
    audit=json.loads((ROOT/'reports/audit_summary.json').read_text(encoding='utf-8'))
    fullsearch=json.loads((ROOT/'results/svm_search/summary.json').read_text(encoding='utf-8'))
    result={'protocol_id':manifest['protocol_id'],'selection_sha256':selection_hash,
            'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'dictionary':audit['dictionary'],'datasets':{},
            'bert_tuning':manifest['indobert']['selected_tuning'],
            'bert_final':[{'dataset':m['dataset'],'condition':m['condition'],'seed':m['seed'],
                           'learning_rate':m['selected_learning_rate'],'best_epoch':m['best_epoch']}
                          for m in manifest['indobert']['models']],
            'svm_selected':[{'dataset':m['dataset'],'representation':m['representation'],
                             'condition':m['condition'],'C':m['selected_C']} for m in manifest['svm']['models']]}
    for dataset in ('prdec','smsa'):
        train,valid,_=selected[dataset]; test,_=load_heldout(dataset,train,valid)
        parts={'train':train,'valid':valid,'test':test}; item=audit['datasets'][dataset]
        dataset_record={'release_rows':item['release_rows'],'retained_rows':item['retained_rows'],
                        'exclusions':item['exclusions'],'split_summary':item['splits'],'vector_survival':{}}
        for split,rows in parts.items():
            summary=item['splits'][split]
            if len(rows)!=summary['reviews'] or sum(len(r['replacements']) for r in rows)!=summary['replacement_tokens']:
                raise ValueError('Retained count or replacement-event count differs from frozen audit')
            if sum(r['original']!=r['normalized'] for r in rows)!=summary['changed_reviews']:
                raise ValueError('Text-change count differs from frozen audit')
            if sum(word_tokens(r['original'])!=word_tokens(r['normalized']) for r in rows)!=summary['word_token_sequence_changed']:
                raise ValueError('Word-token change count differs from frozen audit')
            if sum(r['bert_input_changed'] for r in rows)!=summary['bert_truncated_ids_changed']:
                raise ValueError('BERT input-change count differs from frozen audit')
        for rep in ('word','character'):
            for condition in ('original','normalized'):
                entry,model=models[(dataset,rep,condition)]; name=f'{rep}_{condition}'
                diagnostics={s:vector_diagnostic(model['tfidf'],rows) for s,rows in parts.items()}
                old=fullsearch['datasets'][dataset]['models'][name]
                if diagnostics['train']!=old['vector_diagnostic_train'] or diagnostics['valid']!=old['vector_diagnostic_valid']:
                    raise ValueError('Validation/train vector results differ from search')
                dataset_record['vector_survival'][name]=diagnostics
        result['datasets'][dataset]=dataset_record
    OUTPUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# Dataset, selection, and coverage facts','',
           'All numbers come from the frozen audit, frozen selections and selected SVM vectorizers. These are aggregate descriptions of the studied data and saved models, not new training or performance-based selection. The dictionaries are operational matches, not gold slang labels or semantic-preservation annotations.', '',
           'PRDECT-ID contains Indonesian Tokopedia product reviews across 29 categories ([source](https://data.mendeley.com/datasets/574v66hf2v/1)). SmSA contains Indonesian comments and reviews from multiple online platforms ([IndoNLU paper](https://aclanthology.org/2020.aacl-main.85/)); the released SmSA input files were already preprocessed. Label sets, domains and preprocessing differ, so absolute scores should not be used to infer relative dataset difficulty.', '',
           '| Dataset | Split | Reviews | Groups | Negative | Neutral | Positive |', '|---|---|---:|---:|---:|---:|---:|']
    for ds,d in result['datasets'].items():
        for split,s in d['split_summary'].items():
            c=s['classes']; lines.append(f'| {ds} | {split} | {s["reviews"]} | {s["groups"]} | {c.get("negative",0)} | {c.get("neutral","—")} | {c.get("positive",0)} |')
    lines+=['','Released/retained: PRDECT-ID 5,400/5,288 (112 canonical duplicates excluded); SmSA 12,760/12,679 (81 canonical duplicates excluded). All SmSA removals were from training.', '',
            'SVM C candidates were 0.1, 1 and 10, selected by validation macro-F1 with smaller C on exact ties. All selected C values are 1 except SmSA word/normalized, where C=10. No validation examples were added to training.', '',
            'IndoBERT learning rates were tuned at seed 42 by validation macro-F1 with earlier epoch then lower rate on ties; final seeds 13, 37 and 73 were retrained at the selected rate and each selected its own best validation epoch (earlier epoch on ties).', '',
            '| Dataset | Condition | Tuning LR | Tuning best epoch | Final epochs for seeds 13 / 37 / 73 |', '|---|---|---:|---:|---:|']
    for ds in ('prdec','smsa'):
        for condition in ('original','normalized'):
            t=result['bert_tuning'][f'{ds}_{condition}']; fs=[next(x for x in result['bert_final'] if (x['dataset'],x['condition'],x['seed'])==(ds,condition,seed))['best_epoch'] for seed in (13,37,73)]
            lines.append(f'| {ds} | {condition} | {t["learning_rate"]:.0e} | {t["best_epoch"]} | {fs[0]} / {fs[1]} / {fs[2]} |')
    di=result['dictionary'];lines+=['',f'Dictionary: {di["rows"]:,} source rows; {di["eligible_unique_keys"]:,} eligible unique keys; {di["ambiguous_keys_skipped"]:,} ambiguous keys skipped; {di["active_mapping_keys"]:,} active mappings. Counted replacements are occurrences, not distinct terms or evidence that replacements are linguistically correct.', '',
            '| Dataset | Split | Replacement events | Text-changed reviews | Word-token sequence changed | BERT token IDs changed after truncation |', '|---|---|---:|---:|---:|---:|']
    for ds,d in result['datasets'].items():
        for split,s in d['split_summary'].items():
            lines.append(f'| {ds} | {split} | {s["replacement_tokens"]} | {s["changed_reviews"]} | {s["word_token_sequence_changed"]} | {s["bert_truncated_ids_changed"]} |')
    lines+=['','The following held-out TF-IDF vector survival counts compare original and normalized forms through each named fixed, training-fitted vectorizer. Denominators are text-changed reviews. A changed vector means some effect survives vocabulary filtering; it does not identify which replacement was responsible. These counts were calculated descriptively from the saved models after evaluation and were not used to select models.', '',
            '| Dataset | Fixed vectorizer | Test changed vectors / text-changed reviews |', '|---|---|---:|']
    for ds,d in result['datasets'].items():
        for name,records in d['vector_survival'].items():
            s=records['test']; lines.append(f'| {ds} | {name} | {s["changed_text_with_different_tfidf_vector"]} / {s["text_changed"]} |')
    lines+=['','Machine-readable aggregate results: `results/final_evaluation/study_facts.json`. Resource and sharing caveats are in `reports/resource_availability.md`.', '']
    REPORT.write_text('\n'.join(lines),encoding='utf-8')
    print('Aggregate facts ready.')

if __name__=='__main__':main()
