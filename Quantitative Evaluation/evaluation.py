import pandas as pd
import re
import os
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer
from scipy import stats


# configuration
LOCAL_MODEL_PATH = r'C:\Users\CVYHQ\models\all-MiniLM-L6-v2' # replace with local model path
FILE_NAME = 'Evaluation_Queries_and_Answers.xlsx' # replace with file path


# data clean
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\\', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def main():
    print(f"loading local model: {LOCAL_MODEL_PATH}...")
    try:
        # load local model
        model = SentenceTransformer(LOCAL_MODEL_PATH)
    except Exception as e:
        print(f" loading local model failed: {e}")
        print("please check the path or model availability.")
        return

    # initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    print(f"reading data: {FILE_NAME}...")
    try:
        df = pd.read_excel(FILE_NAME)
        df = df.dropna(subset=['golden_answer', 'baseline_answer_text', 'artifacts_implemented_answer_text'])
    except Exception as e:
        print(f" reading file failed: {e}")
        return

    # clean text columns
    df['golden_clean'] = df['golden_answer'].apply(clean_text)
    df['baseline_clean'] = df['baseline_answer_text'].apply(clean_text)
    df['artifacts_implemented_clean'] = df['artifacts_implemented_answer_text'].apply(clean_text)

    # computing embeddings
    print("computing embeddings...")
    #SBERT embeddings
    emb_golden = model.encode(df['golden_clean'].tolist(), convert_to_tensor=True)
    emb_base = model.encode(df['baseline_clean'].tolist(), convert_to_tensor=True)
    emb_impr = model.encode(df['artifacts_implemented_clean'].tolist(), convert_to_tensor=True)

    print("calculating metrics...")
    base_sem_scores = []
    impr_sem_scores = []
    base_rouge_scores = []
    impr_rouge_scores = []

    for i in range(len(df)):
        # compute the cosine of the angle between two SBERT vectors
        base_sem = util.cos_sim(emb_base[i], emb_golden[i]).item()
        impr_sem = util.cos_sim(emb_impr[i], emb_golden[i]).item()
        base_sem_scores.append(base_sem)
        impr_sem_scores.append(impr_sem)

        # compute ROUGE-L fmeasure (F1 score)
        b_rouge = scorer.score(df.iloc[i]['golden_clean'], df.iloc[i]['baseline_clean'])['rougeL'].fmeasure
        i_rouge = scorer.score(df.iloc[i]['golden_clean'], df.iloc[i]['artifacts_implemented_clean'])['rougeL'].fmeasure
        base_rouge_scores.append(b_rouge)
        impr_rouge_scores.append(i_rouge)

    # store in DataFrame
    df['Base_Semantic'] = base_sem_scores
    df['Arti_Semantic'] = impr_sem_scores
    df['Base_ROUGE_L'] = base_rouge_scores
    df['Arti_ROUGE_L'] = impr_rouge_scores

    # Wilcoxon Signed-Rank Test
    _, p_sem = stats.wilcoxon(df['Base_Semantic'], df['Arti_Semantic'])
    _, p_rouge = stats.wilcoxon(df['Base_ROUGE_L'], df['Arti_ROUGE_L'])

    # create report
    print("\n" + "="*50)
    print("DSR Evaluation Report")
    print("-" * 50)
    metrics = {
        "Metric": ["Semantic Similarity", "ROUGE-L"],
        "Baseline (Mean)": [df['Base_Semantic'].mean(), df['Base_ROUGE_L'].mean()],
        "Improved (Mean)": [df['Arti_Semantic'].mean(), df['Arti_ROUGE_L'].mean()],
        "P-Value": [p_sem, p_rouge]

    }
    report_df = pd.DataFrame(metrics)
    print(report_df.round(4).to_string(index=False))
    print("="*50)

    # store detailed results
    df.to_csv("Evaluation_Results_Final.csv", index=False)
    print("\n: Evaluation_Results_Final.csv")

if __name__ == "__main__":
    main()