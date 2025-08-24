import json
from rouge_score import rouge_scorer

def save_jsonl(df, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            item = {
                "instruction": row['question'],
                "output": row['answer']
            }
            f.write(json.dumps(item, ensure_ascii=False) + '\n')



def rouge_lsum(pred: str, ref: str):
    scorer = rouge_scorer.RougeScorer(['rougeLsum'], use_stemmer=True)
    scores = scorer.score(ref, pred)
    return scores['rougeLsum']

