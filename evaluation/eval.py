# Adapted from https://github.com/princeton-nlp/ALCE/main/utils.py

import argparse
import collections
import json
import re
import string
import torch
import copy

from nltk import sent_tokenize
import numpy as np
from rouge_score import rouge_scorer, scoring
from tqdm import tqdm
import sys
import logging
import os

log_file_path = os.environ["EVAL_LOG_PATH"]

log_fh = open(log_file_path, 'a')

sys.stderr = log_fh
sys.stdout = log_fh

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', 
                    handlers=[
                        logging.FileHandler(log_file_path, mode="a"),
                    ])
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig
)

from eval_utils import normalize_answer, get_max_memory, remove_citations

QA_MODEL="gaotianyu1350/roberta-large-squad"

global autoais_model, autoais_tokenizer, autoais_model_name, autoais_model_type
autoais_model, autoais_tokenizer, autoais_model_name, autoais_model_type = None, None, None, None

max_seq_len = 2048

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

def compute_f1(a_gold, a_pred):
    """Compute F1 score between two strings."""

    def _get_tokens(s):
        if not s:
            return []
        return normalize_answer(s).split()

    gold_toks = _get_tokens(a_gold)
    pred_toks = _get_tokens(a_pred)

    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def compute_exact(a_gold, a_pred):
    """Check whether two strings are equal up to normalization."""

    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def exact_presence(short_answers, context):
    """Verify if any of the answers is present in the given context.
    Args:
        short_answers: list of short answers to look for in the context
        context: a paragraph to search for short answers
    Returns:
        true if any of the short answers is present in the context
    """

    n_short_answers = [normalize_answer(sa) for sa in short_answers]
    n_context = normalize_answer(context)

    for ans in n_short_answers:
        if ans in n_context:
            return True

    return False


def compute_rouge(data):
    """Main function for rouge scoring.
    If two references are provided,
    the best score is chosen for each instance.
    Args:
        data: requires field `output` and `answer` (or `annotations` for ASQA)
        metrics: list of evaluation metrics
    Returns:
        dictionary representation of rouge scores
    """
    def _rouge_calculation(hypotheses,
                        references1,
                        references2=[],
                        metrics=['rougeLsum']):

        if references2 == []:
            references2 = references1

        scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
        aggregator = scoring.BootstrapAggregator()

        for i in range(len(hypotheses)):
            scores1 = scorer.score(references1[i], hypotheses[i])
            scores2 = scorer.score(references2[i], hypotheses[i])
            if scores1['rougeLsum'].fmeasure > scores2['rougeLsum'].fmeasure:
                aggregator.add_scores(scores1)
            else:
                aggregator.add_scores(scores2)

        scores = {m: [] for m in metrics}

        for m in metrics:
            fmeasure = aggregator.aggregate()[m].mid.fmeasure
            scores[m].append(fmeasure)

        for m in scores:
            scores[m] = 100 * sum(scores[m]) / len(scores[m])

        return scores

    hypotheses = {}
    references1 = {}
    references2 = {}

    for idx, item in enumerate(data):
        hypotheses[idx] = item["output"]
        if "annotations" in item and item['annotations'] is not None and len(item['annotations']) > 0: # For ASQA
            references1[idx] = item["annotations"][0]["long_answer"]
            references2[idx] = item["annotations"][1]["long_answer"]
        else:
            references1[idx] = item["answer"]
            references2[idx] = item["answer"]

    h, r1, r2 = [], [], []

    for key in references1:
        h.append(hypotheses[key])
        r1.append(references1[key])

        if references2 is not None:
            r2.append(references2[key])

    h = ['\n'.join(sent_tokenize(text.lower())) for text in h]
    r1 = ['\n'.join(sent_tokenize(text.lower())) for text in r1]
    r2 = ['\n'.join(sent_tokenize(text.lower())) for text in r2]
    scores = _rouge_calculation(h, r1, r2)

    return scores['rougeLsum']


def compute_str_em(data, doc_rec=None, dataset=None):
    """Compute STR-EM metric (only for ASQA)
    Args:
        data: requires field `qa_pairs/short_answers` and `output`
    Returns:
        STR-EM and STR-EM-HIT ()
    """

    if dataset != "asqa":
        return 0, 0, 0, 0

    acc = []
    hit = []

    doc_adj_acc = []
    # doc_adj_hit = []

    hit_count = 0
    ans_count = 0

    for i, item in enumerate(data):
        loc_acc = []
        for qa_pair in item['qa_pairs']:
            loc_acc.append(exact_presence(qa_pair['short_answers'], item["output"]))
        acc.append(np.mean(loc_acc))
        hit.append( int(np.mean(loc_acc) == 1) )

        hit_count = hit_count + np.sum(loc_acc)
        ans_count = ans_count + len(item['qa_pairs'])

        if doc_rec is not None:
            doc_rec_loc_acc = [loc_acc[idx] for idx in doc_rec[i]["doc_rec_ans"]]
            # If documents do not capture any answer, then set the adjusted recall to be 1
            curr_doc_adj_acc = np.mean(doc_rec_loc_acc) if len(doc_rec[i]["doc_rec_ans"]) > 0 else 1
            doc_adj_acc.append(curr_doc_adj_acc)
            # doc_adj_hit.append(int(curr_doc_adj_acc == 1))

    str_em_eq_w = hit_count / ans_count

    doc_adj_str_em = 100 * np.mean(doc_adj_acc) if doc_rec is not None else 0
    # doc_adj_str_hit = 100 * np.mean(doc_adj_hit) if doc_rec is not None else 0

    return 100 * np.mean(acc), 100 * np.mean(hit), 100 * str_em_eq_w, doc_adj_str_em


def compute_len(data):
    """Compute average length of predictions."""

    res, cntr = 0, 0
    for item in data:
        res += len(item["output"].split())
        cntr += 1
    return res / cntr


def compute_qa(data):
    """Compute QA-based accuracy.
    Args:
        data: requires filed `qa_pairs/short_answers` and `output`
    Returns:
        QA metrics (QA-EM, QA-F1, QA-Hit)
    """

    if 'qa_pairs' not in data[0] or data[0]['qa_pairs'] is None:
        logger.warn("Warning: no QA pairs found in data")
        return {
            'QA-EM': 0,
            'QA-F1': 0,
            'QA-Hit': 0,
        }

    # Load model
    logger.info("Loading the RoBERTa-large SQuAD model for QA-based accuracy...")
    qa_pipeline = pipeline("question-answering", model=QA_MODEL, device=0)
    logger.info("Done")

    # Get prediction
    logger.info("Computing the QA-based accuracy...")
    em, f1, bins = [], [], []
    for item in tqdm(data):
        question = [qa_pair['question'] for qa_pair in item['qa_pairs']]
        context = item['output'] if len(item['output']) > 0 else " "
        results = qa_pipeline(question=question, context=context, handle_impossible_answer=True)
        loc_counter, loc_em, loc_f1 = 0, 0, 0

        for idx, res in enumerate(results):
            answers = item["qa_pairs"][idx]["short_answers"]
            prediction = res["answer"]

            loc_em += max([compute_exact(a, prediction) for a in answers])
            loc_f1 += max([compute_f1(a, prediction) for a in answers])
            loc_counter += 1

        em.append(loc_em / loc_counter)
        f1.append(loc_f1 / loc_counter)
        bins.append(loc_em == loc_counter)

    return {
        'QA-EM': 100 * np.mean(em),
        'QA-F1': 100 * np.mean(f1),
        'QA-Hit': 100 * np.mean(bins)
    }


def compute_mauve(data):
    """Compute Mauve score."""

    logger.info("Computing MAUVE...")
    human_data = []
    model_data = []
    for item in data:
        # Remove ending punctuations
        # Remove any new lines
        # Truncate by 100 words
        human_data.append(' '.join((item['question'] + " " + item['answer'].strip()).split()[:100]).rstrip(string.punctuation))
        model_data.append(' '.join((item['question'] + " " + item['output'].strip()).split()[:100]).rstrip(string.punctuation))

    import mauve
    out = mauve.compute_mauve(
        p_text=human_data,
        q_text=model_data,
        device_id=0,
        max_text_length=512,
        verbose=True,
        batch_size=8,
        featurize_model_name="gpt2-large"
    )
    return out.mauve * 100


def _run_nli_autoais(passage, claim):
    """
    Run inference for assessing AIS between a premise and hypothesis.
    Adapted from https://github.com/google-research-datasets/Attributed-QA/blob/main/evaluation.py
    """
    global autoais_model, autoais_tokenizer, autoais_model_name, autoais_model_type
    
    if "true" in autoais_model_name:
        input_text = "premise: {} hypothesis: {}".format(passage, claim)
    elif "attrscore" in autoais_model_name:
        input_text = "As an Attribution Validator, your task is to verify whether a given reference can support the given claim. A claim can be either a plain sentence or a question followed by its answer. Specifically, your response should clearly indicate the relationship: Attributable, Contradictory or Extrapolatory. A contradictory error occurs when you can infer that the answer contradicts the fact presented in the context, while an extrapolatory error means that you cannot infer the correctness of the answer based on the information provided in the context. \n\nClaim: {} \n Reference: {}".format(claim, passage)
    else:
        raise Exception("Model {} is not defined".format(autoais_model_name))

    # May need to consider max_input_len
    if "true" in autoais_model_name:
        input_ids = autoais_tokenizer(input_text, return_tensors="pt", max_length=max_seq_len, truncation=True).input_ids.to(autoais_model.device)
    elif "attrscore" in autoais_model_name:
        input_ids = autoais_tokenizer(input_text, return_tensors="pt").input_ids.to(autoais_model.device)
    else:
        raise Exception("Model {} is not defined".format(autoais_model_name))
    with torch.inference_mode():
        outputs = autoais_model.generate(input_ids, max_new_tokens=10)
    result = autoais_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # print("Result: {}\n".format(result))

    if "true" in autoais_model_name:
        inference = 1 if result == "1" else 0
    elif "attrscore" in autoais_model_name:
        att_keys = [w.casefold() for w in ["Attributable"]]
        # Combine Extrapolatory and Contraditory into 1 single category
        inference = 1 if result.casefold() in att_keys else 0
    else:
        raise Exception("Model {} is not defined".format(autoais_model_name))
    return inference



def compute_claims(data, doc_rec=None):
    global autoais_model, autoais_tokenizer, autoais_model_name, autoais_model_type
    if autoais_model is None:
        logger.info("Initalizeing AutoAIS model {} with {}...".format(autoais_model_name, autoais_model_type))
        if autoais_model_type == "bf16":
            autoais_model = AutoModelForSeq2SeqLM.from_pretrained(autoais_model_name, torch_dtype=torch.bfloat16, max_memory=get_max_memory(), device_map="auto", offload_folder="offload", offload_state_dict = True)
        elif autoais_model_type == "4bit":
            autoais_model = AutoModelForSeq2SeqLM.from_pretrained(autoais_model_name, load_in_4bit=True, device_map=0)
        elif autoais_model_type == "bnb":
            autoais_model = AutoModelForSeq2SeqLM.from_pretrained(autoais_model_name, quantization_config=bnb_config)
        else:
            raise Exception("autoais_model_type {} is not valid".format(autoais_model_type))
        autoais_tokenizer = AutoTokenizer.from_pretrained(autoais_model_name, use_fast=False)

    logger.info("Computing claims...")
    scores = []

    doc_adj_scores = []

    for i in tqdm(range(len(data))):
        item = data[i]
        normalized_output = remove_citations(item['output'])
        # entail = 0
        claims = item["claims"]

        entails = []

        for claim in claims:
            if "attrscore" in autoais_model_name:
                claim = "{} {}".format(item["question"], claim)
            entails.append(_run_nli_autoais(normalized_output, claim))

        scores.append(sum(entails) / len(claims))

        if doc_rec is not None:
            doc_adj_entails = [entails[idx] for idx in doc_rec[i]["doc_rec_ans"]]
            doc_adj_score = sum(doc_adj_entails) / len(doc_rec[i]["doc_rec_ans"]) if len(doc_rec[i]["doc_rec_ans"]) > 0 else 1
            doc_adj_scores.append(doc_adj_score)

    doc_adj_claims_nli = 100 * np.mean(doc_adj_scores) if doc_rec is not None else 0
    return 100 * np.mean(scores), doc_adj_claims_nli


def compute_autoais(data,
                    decontext=False,
                    concat=False,
                    qampari=False,
                    at_most_citations=None,):
    """
    Compute AutoAIS score.

    Args:
        data: requires field `output` and `docs`
              - docs should be a list of items with fields `title` and `text` (or `phrase` and `sent` for QA-extracted docs)
        citation: check citations and use the corresponding references.
        decontext: decontextualize the output
    """

    global autoais_model, autoais_tokenizer, autoais_model_name, autoais_model_type
    if autoais_model is None:
        logger.info("Initalizeing AutoAIS model {} with {}...".format(autoais_model_name, autoais_model_type))
        # Replace torch_dtype=torch.bfloat16 with quantization_config=bnb_config to use QLoRA
        if autoais_model_type == "bf16":
            autoais_model = AutoModelForSeq2SeqLM.from_pretrained(autoais_model_name, torch_dtype=torch.bfloat16, max_memory=get_max_memory(), device_map="auto", offload_folder="offload", offload_state_dict = True)
        elif autoais_model_type == "4bit":
            autoais_model = AutoModelForSeq2SeqLM.from_pretrained(autoais_model_name, load_in_4bit=True, device_map=0)
        elif autoais_model_type == "bnb":
            autoais_model = AutoModelForSeq2SeqLM.from_pretrained(autoais_model_name, quantization_config=bnb_config)
        else:
            raise Exception("autoais_model_type {} is not valid".format(autoais_model_type))
        autoais_tokenizer = AutoTokenizer.from_pretrained(autoais_model_name, use_fast=False)

    logger.info(f"Running AutoAIS...")

    def _format_document(doc):
        """Format document for AutoAIS."""

        if "sent" in doc:
            # QA-extracted docs
            return "Title: %s\n%s" % (doc['title'], doc['sent'])
        else:
            return "Title: %s\n%s" % (doc['title'], doc['text'])

    ais_scores = []
    num_sents = []
    ais_scores_prec = []

    num_cits = []

    sent_total = 0
    sent_mcite = 0
    sent_mcite_support = 0
    sent_mcite_overcite = 0
    autoais_log = []
    for item in tqdm(data):
        # Get sentences by using NLTK
        if qampari:
            sents = [item['question'] + " " + x.strip() for x in item['output'].rstrip().rstrip(".").rstrip(",").split(",")]
        else:
            if "true" in autoais_model_name:
                sents = sent_tokenize(item['output'])
            elif "attrscore" in autoais_model_name:
                sents = ["{} {}".format(item["question"], sent) for sent in sent_tokenize(item['output'])]
            else:
                raise Exception("Model {} is not defined".format(autoais_model_name))

        if len(sents) == 0:
            continue

        target_sents = [remove_citations(sent).strip() for sent in sents]

        entail = 0
        entail_prec = 0
        total_citations = 0
        for sent_id, sent in enumerate(sents):
            target_sent = target_sents[sent_id] # Citation removed and (if opted for) decontextualized
            joint_entail = -1 # Undecided

            # Find references
            ref = [int(r[1:])-1 for r in re.findall(r"\[\d+", sent)] # In text citation id starts from 1
            logger.info(f"For `{sent}`, find citations {ref}")
            if len(ref) == 0:
                # No citations
                joint_entail = 0
            elif any([ref_id >= len(item['docs']) for ref_id in ref]):
                # Citations out of range
                joint_entail = 0
            else:
                if at_most_citations is not None:
                    ref = ref[:at_most_citations]
                total_citations += len(ref)
                joint_passage = '\n'.join([_format_document(item['docs'][psgs_id]) for psgs_id in ref])

            # If not directly rejected by citation format error, calculate the recall score
            if joint_entail == -1: 
                joint_entail = _run_nli_autoais(joint_passage, target_sent)
                autoais_log.append({
                    "question": item['question'],
                    "output": item['output'],
                    "claim": sent if "true" in autoais_model_name else sent[len("{} ".format(item["question"])):],
                    "passage": [joint_passage],
                    "model_type": "NLI",
                    "{}_output".format(autoais_model_name): joint_entail,
                })

            entail += joint_entail
            if len(ref) > 1:
                sent_mcite += 1

            # calculate the precision score if applicable
            if joint_entail and len(ref) > 1:
                sent_mcite_support += 1
                # Precision check: did the model cite any unnecessary documents?
                for psgs_id in ref:
                    # condition A
                    passage = _format_document(item['docs'][psgs_id]) 
                    nli_result = _run_nli_autoais(passage, target_sent)

                    # condition B
                    if not nli_result:
                        subset_exclude = copy.deepcopy(ref)
                        subset_exclude.remove(psgs_id)
                        passage = '\n'.join([_format_document(item['docs'][pid]) for pid in subset_exclude])
                        nli_result = _run_nli_autoais(passage, target_sent)
                        if nli_result: # psgs_id is not necessary
                            flag = 0
                            sent_mcite_overcite += 1 
                        else:
                            entail_prec += 1
                    else:
                        entail_prec += 1
            else:
                entail_prec += joint_entail 

        num_cits.append(total_citations)
        num_sents.append(len(sents))

        sent_total += len(sents)
        ais_scores.append(entail / len(sents))
        ais_scores_prec.append(entail_prec / total_citations if total_citations > 0 else 0) # len(sents))

    if sent_mcite > 0 and sent_mcite_support > 0:
        logger.info("Among all sentences, %.2f%% have multiple citations, among which %.2f%% are supported by the joint set, among which %.2f%% overcite." % (
            100 * sent_mcite / sent_total, 
            100 * sent_mcite_support / sent_mcite, 
            100 * sent_mcite_overcite / sent_mcite_support
        ))

    # with open("./eval_attrscore_xl_result.json", "w") as tgt_json:
    #     json.dump(autoais_log, tgt_json, indent=4)

    return {
        "citation_rec": 100 * np.mean(ais_scores),
        "citation_prec": 100 * np.mean(ais_scores_prec),
        "citation_rec_eq_w": 100 * sum([r * n for r, n in zip(ais_scores, num_sents)]) / sum(num_sents),
        "citation_prec_eq_w": (100 * sum([p * n for p, n in zip(ais_scores_prec, num_cits)]) / sum(num_cits)) if sum(num_cits) > 0 else 0,
    }


def compute_qampari_f1(data, cot=False, doc_rec=None):
    prec = []
    rec = []
    num_ans = []
    rec_top5 = []
    f1 = []
    f1_top5 = []

    doc_adj_prec = []
    doc_adj_rec = []
    doc_adj_rec_top5 = []
    doc_adj_f1 = []
    doc_adj_f1_top5 = []

    num_preds = []
    for i, item in enumerate(data):
        if cot:
            if ":" in item['output']:
                o = ':'.join(item['output'].split(":")[1:]) # try to separate the COT part and the answer list part.
            else:
                o = ""
        else:
            o = item['output']
        preds = [normalize_answer(x.strip()) for x in o.rstrip().rstrip(".").rstrip(",").split(",")]
        preds = [p for p in preds if len(p) > 0] # delete empty answers
        num_preds.append(len(preds))
        answers = [[normalize_answer(x) for x in ans] for ans in item['answers']]
        flat_answers = [ans_item for sublist in answers for ans_item in sublist]
        
        prec.append(sum([p in flat_answers for p in preds]) / len(preds) if len(preds) > 0 else 0)

        hits = [any([x in preds for x in a]) for a in answers]
        rec.append(sum(hits) / len(answers))
        num_ans.append(len(answers))

        rec_top5.append(min(5, sum(hits)) / min(5, len(answers)))
        if (prec[-1] + rec[-1]) == 0:
            f1.append(0)
        else:
            f1.append(2 * prec[-1] * rec[-1] / (prec[-1] + rec[-1]))
        if (prec[-1] + rec_top5[-1]) == 0:
            f1_top5.append(0) 
        else:
            f1_top5.append(2 * prec[-1] * rec_top5[-1] / (prec[-1] + rec_top5[-1]))

        if doc_rec is not None:
            doc_adj_flat_answers = [ans_item for idx, sublist in enumerate(answers) for ans_item in sublist if idx in doc_rec[i]["doc_rec_ans"]]
            doc_adj_prec.append(sum([p in doc_adj_flat_answers for p in preds]) / len(preds) if len(preds) > 0 else 0)
            
            doc_adj_hits = [hits[idx] for idx in doc_rec[i]["doc_rec_ans"]]
            doc_adj_r = sum(doc_adj_hits) / len(doc_rec[i]["doc_rec_ans"]) if len(doc_rec[i]["doc_rec_ans"]) > 0 else 1
            doc_adj_rec.append(doc_adj_r)
            doc_adj_r_top5 = (min(5, sum(doc_adj_hits)) / min(5, len(doc_rec[i]["doc_rec_ans"]))) if len(doc_rec[i]["doc_rec_ans"]) > 0 else 1
            doc_adj_rec_top5.append(doc_adj_r_top5)

            if (doc_adj_prec[-1] + doc_adj_rec[-1]) == 0:
                doc_adj_f1.append(0)
            else:
                doc_adj_f1.append(2 * doc_adj_prec[-1] * doc_adj_rec[-1] / (doc_adj_prec[-1] + doc_adj_rec[-1]))
            if (doc_adj_prec[-1] + doc_adj_rec_top5[-1]) == 0:
                doc_adj_f1_top5.append(0)
            else:
                doc_adj_f1_top5.append(2 * doc_adj_prec[-1] * doc_adj_rec_top5[-1] / (doc_adj_prec[-1] + doc_adj_rec_top5[-1]))

    qampari_prec_eq_w = sum([p * n for p, n in zip(prec, num_preds)]) / sum(num_preds)
    qampari_rec_eq_w = sum([r * n for r, n in zip(rec, num_ans)]) / sum(num_ans)
    qampari_rec_top5_eq_w = sum([r * min(n, 5) for r, n in zip(rec_top5, num_ans)]) / sum([min(n, 5) for n in num_ans])
    qampari_f1_eq_w = (2 * qampari_prec_eq_w * qampari_rec_eq_w) / (qampari_prec_eq_w + qampari_rec_eq_w)
    qampari_f1_top_5_eq_w = (2 * qampari_prec_eq_w * qampari_rec_top5_eq_w) / (qampari_prec_eq_w + qampari_rec_top5_eq_w)

    return {
        "num_preds": np.mean(num_preds),
        "qampari_prec": 100 * np.mean(prec),
        "qampari_rec": 100 * np.mean(rec),
        "qampari_rec_top5": 100 * np.mean(rec_top5),
        "qampari_f1": 100 * np.mean(f1),
        "qampari_f1_top5": 100 * np.mean(f1_top5),
        "qampari_prec_eq_w": qampari_prec_eq_w,
        "qampari_rec_eq_w": qampari_rec_eq_w,
        "qampari_rec_top5_eq_w": qampari_rec_top5_eq_w,
        "qampari_f1_eq_w": qampari_f1_eq_w,
        "qampari_f1_top_5_eq_w": qampari_f1_top_5_eq_w,
        "qampari_doc_adj_prec": 100 * np.mean(doc_adj_prec) if doc_rec is not None else 0,
        "qampari_doc_adj_rec": 100 * np.mean(doc_adj_rec) if doc_rec is not None else 0,
        "qampari_doc_adj_rec_top5": 100 * np.mean(doc_adj_rec_top5) if doc_rec is not None else 0,
        "qampari_doc_adj_f1": 100 * np.mean(doc_adj_f1) if doc_rec is not None else 0,
        "qampari_doc_adj_f1_top5": 100 * np.mean(doc_adj_f1_top5) if doc_rec is not None else 0,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=str, required=True, help="Output file. Should have field `question`, `output`, (ROUGE) `answer`, \
                        (accuracy) `qa_pairs`, (AIS) `docs`")
    parser.add_argument("--no_rouge", action="store_true", help="Do not evaluate ROUGE score")
    parser.add_argument("--qa", action="store_true", help="Use the QA model")
    parser.add_argument("--mauve", action="store_true", help="Use the mauve score model")
    parser.add_argument("--citations", action="store_true", help="Evaluation with citation")
    parser.add_argument("--at_most_citations", type=int, default=3, help="At most take this many documents (mostly for precision)")
    parser.add_argument("--claims_nli", action="store_true", help="Use claims for ELI5")
    parser.add_argument("--no_doc_rec_corr", action="store_true", help="Whether or not to compute adjusted correctness recall and precision with only answers that can be recalled by the passages")
    parser.add_argument("--nli_model_name_or_path", type=str, default="osunlp/attrscore-flan-t5-xl")
    parser.add_argument("--nli_model_type", type=str, default="bnb", choices=["bf16", "4bit", "bnb"])
    parser.add_argument("--beam_search_depth", type=int, default=0, help="The beam search depth to use for inference")

    # QAMPARI
    parser.add_argument("--cot", action="store_true", help="For QAMPARI, try to find colon and separate the COT and answer listing")

    args = parser.parse_args()

    global autoais_model_name, autoais_model_type
    autoais_model_name = args.nli_model_name_or_path
    autoais_model_type = args.nli_model_type

    with open(args.f) as f:
        data_with_config = json.load(f)
    data = data_with_config['data'] 

    dataset = data[0]["dataset"] if "dataset" in data[0] else (args.f.split("/")[-1]).split("-")[0]

    doc_rec = None
    if not args.no_doc_rec_corr:
        logger.warning("Computing document adjusted recall in addition to the original recall metric")
        # To get document adjusted recall
        doc_rec_json_path_map = {
            "asqa": "./tasks/qa_feedback/data/asqa_rlhf_test_shot_1_ndoc_5_ndoc_in_demo_5_no_inst_doc_rec_ans_true_xxl_seq_2048.json",
            "qampari": "./tasks/qa_feedback/data/qampari_rlhf_test_shot_1_ndoc_5_ndoc_in_demo_5_no_inst_doc_rec_ans_true_xxl_seq_2048.json",
            "eli5": "./tasks/qa_feedback/data/eli5_rlhf_test_1000_shot_1_ndoc_5_ndoc_in_demo_5_no_inst_doc_rec_ans_true_xxl_seq_2048.json",
            "expertqa": "./tasks/qa_feedback/data/expertqa_rlhf_test_shot_2_ndoc_5_ndoc_in_demo_5_default_inst_doc_rec_ans_true_xxl_seq_2048.json"
        }
        with open(doc_rec_json_path_map[dataset], "r") as doc_rec_json:
            doc_rec_map = json.load(doc_rec_json)
            doc_rec_map = {s["question"]: s for s in doc_rec_map}

            doc_rec = []

            for i in range(len(data)):
                s = doc_rec_map[data[i]["question"]]
                doc_rec.append({"doc_rec_ans": s["doc_rec_ans"]})
            assert len(data) == len(doc_rec)

    if "qampari" in args.f:
        args.no_rouge = True
        args.qa = False
        args.mauve = False
        args.decontext = False
        qampari = True
    else:
        qampari = False

    if args.beam_search_depth > 0 and "beam_search" in data[0]:
        use_earlier_beam_count = 0
        for i in range(len(data)):
            key = "beam_search_depth_{}".format(args.beam_search_depth)
            if key in data[i] and len(data[i][key]) > 0:
                data[i]["output"] = data[i][key][0]["pred_text"]
                use_earlier_beam_count = use_earlier_beam_count + 1
        logger.info("{} samples uses earlier beams at depth {}".format(use_earlier_beam_count, args.beam_search_depth))


    # Truncate by newline and remove on the fly search result
    logger.warning("We remove all the pre/appended space/newlines and we truncate the answer by the first newline.")
    logger.warning("We replace any on the fly search result to standard bracket citation format.")
    for i in range(len(data)):
        data[i]['output'] = data[i]['output'].strip().split("\n")[0]
        data[i]['output'] = data[i]['output'].replace("<|im_end|>", "")


    # Remove all citations for all non-AutoAIS evaluation
    normalized_data = copy.deepcopy(data)
    for i in range(len(normalized_data)):
        normalized_data[i]['output'] = remove_citations(normalized_data[i]['output'])

    result = {}
    result['length'] = compute_len(normalized_data)

    # The correctness on ASQA per disambiguated question D (1 if any answer choice of D is presented in output) and ambiguous question A (1 if every disambiguated question D in A is answered by output)
    result['str_em'], result['str_hit'], result["str_em_eq_w"], result["doc_adj_str_em"] = compute_str_em(normalized_data, doc_rec=doc_rec, dataset=dataset)
    if qampari:
        # The correctness on QAMPARI
        result.update(compute_qampari_f1(normalized_data, cot=args.cot, doc_rec=doc_rec))
    if not args.no_rouge:
        # The original correctness metric on ELI5
        result['rougeLsum'] = compute_rouge(normalized_data)
    if args.qa:
        result.update(compute_qa(normalized_data))
    if args.mauve:
        # The fluency on ASQA and ELI5, evaluated between question + answer and question + output
        result['mauve'] = compute_mauve(normalized_data)
    if args.citations:
        # The citation recall and citation precision
        result.update(compute_autoais(data, qampari=qampari, at_most_citations=args.at_most_citations))
    if args.claims_nli:
        # The correctness on ELI5
        result["claims_nli"], result["doc_adj_claims_nli"] = compute_claims(normalized_data, doc_rec=doc_rec)

    logger.info(result)
    json.dump(result, open("{}_{}_{}.json.score".format(args.f[:-len(".json")], args.nli_model_name_or_path.split("/")[-1], args.nli_model_type), "w"), indent=4)


if __name__ == "__main__":
    main()
