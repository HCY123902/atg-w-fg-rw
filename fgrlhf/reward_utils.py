from math import exp
import re
from shutil import ExecError
from nltk import sent_tokenize

import torch
import collections
import string

import numpy as np

from transformers import BitsAndBytesConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from tqdm import tqdm

import copy

import gc
import time

def get_max_memory():
    """Get the maximum memory available for the current GPU for loading models."""
    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    max_memory = f'{free_in_GB-6}GB'
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    return max_memory


# split long text into sentences
def split_text_to_sentences(long_text, spacy_nlp):
    doc = spacy_nlp(long_text)
    return [0] + [sent.end_char for sent in doc.sents if len(str(sent).strip()) > 0]
    

# split long text into subsentences
def split_text_to_subsentences(long_text, spacy_nlp):
    def get_sub_sentence_starts(tokens, min_subsent_words=5):

        def _is_tok_end_of_subsent(tok):
            if re.match('[,;!?]', tok[-1]) is not None:
                return True
            return False

        # assert len(tokens) > 0
        is_subsent_starts = [True]
        prev_tok = tokens[0]
        prev_subsent_start_idx = 0
        for i, tok in enumerate(tokens[1:]):
            tok_id = i + 1
            if _is_tok_end_of_subsent(prev_tok) and tok_id + min_subsent_words < len(tokens):
                if tok_id - prev_subsent_start_idx < min_subsent_words:
                    if prev_subsent_start_idx > 0:
                        is_subsent_starts += [True]
                        is_subsent_starts[prev_subsent_start_idx] = False
                        prev_subsent_start_idx = tok_id
                    else:
                        is_subsent_starts += [False]
                else:
                    is_subsent_starts += [True]
                    prev_subsent_start_idx = tok_id
            else:
                is_subsent_starts += [False]
            prev_tok = tok

        return is_subsent_starts


    def tokenize_with_indices(text):
        tokens = text.split()
        token_indices = []

        current_index = 0
        for token in tokens:
            start_index = text.find(token, current_index)
            token_indices.append((token, start_index))
            current_index = start_index + len(token)

        return token_indices
    
    doc = spacy_nlp(long_text)
    sentence_start_char_idxs= [0] + [sent.end_char for sent in doc.sents if len(str(sent).strip()) > 0]
    
    char_starts = []
    
    for sentence_idx, sentence_start_char_idx in enumerate(sentence_start_char_idxs[:-1]):
        
        sentence = long_text[sentence_start_char_idx: sentence_start_char_idxs[sentence_idx+1]]
        
        tokens_with_indices = tokenize_with_indices(sentence)
        
        tokens = [i[0] for i in tokens_with_indices]
        is_sub_starts = get_sub_sentence_starts(tokens, min_subsent_words=5)
        
        for token_with_idx, is_sub_start in zip(tokens_with_indices, is_sub_starts):
            if is_sub_start:
                char_starts.append(sentence_start_char_idx + token_with_idx[1])
    
    return char_starts + [len(long_text)]


def split_qampari_text(long_text):
    # TODO: Handle citations [i, j] in place of [i][j]

    # # If needed, retain only the first line. Do not strip in any case as it will affect the character postions
    # long_text = long_text.split("\n")[0]

    sents = long_text.split(",")

    # Remove the last empty sentence after splitting, since its sentence_end_char_idx will be the same as the second last sentence, and the token end_idx will overlap with each other
    if long_text.endswith(","):
        sents = sents[:-1]

    sentence_end_char_idxs = [0]
    refs = [[(0, -1)]]
    current_idx = 0
    for i, sent in enumerate(sents):
        if i == len(sents) - 1 and not long_text.endswith(","):
            # The last part does not have comma if the original text does not end with a comma
            next_idx = current_idx + len(sent)
        else:
            next_idx = current_idx + len("{},".format(sent))
        sentence_end_char_idxs.append(next_idx)
        
        ref = [(current_idx + m.end(), int(m.group()[:-1])-1) for m in re.finditer(r"\d+\]", long_text[current_idx:next_idx])] # In text citation id starts from 1
        refs.append(ref)

        current_idx = next_idx

    # Preprocessing adapted from ALCE. Remove end token.
    sents = [sent.replace("<|im_end|>", "") for sent in sents]

    # sents = [question + " " + long_text[sentence_end_char_idxs[i]:sentence_end_char_idxs[i+1]].rstrip().rstrip(".").rstrip(",").strip() for i in range(len(sentence_end_char_idxs)-1)]
    return sents, sentence_end_char_idxs, refs


def split_text(long_text):
    # # If needed, retain only the first line. Do not strip in any case as it will affect the character postions
    # long_text = long_text.split("\n")[0]

    sents = sent_tokenize(long_text)
    sentence_end_char_idxs = [0]
    refs = [[(0, -1)]]

    for sent in sents:
        # start_idx = long_text.find(sent)
        start_idx = long_text.find(sent, sentence_end_char_idxs[-1])
        ref = [(start_idx + m.end(), int(m.group()[:-1])-1) for m in re.finditer(r"\d+\]", sent)] # In text citation id starts from 1
        sentence_end_char_idxs.append(start_idx + len(sent))
        refs.append(ref)

    # Preprocessing adapted from ALCE. Remove end token.
    sents = [sent.replace("<|im_end|>", "") for sent in sents]

    # if "attrscore" in nli_model_ckpt:
    #     sents = ["{} {}".format(question, sent) for sent in sents]
    return sents, sentence_end_char_idxs, refs




def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def remove_citations(sent):
    # TODO: Handle citations [i, j] in place of [i][j]
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")


def exact_presence(n_answers, n_context):
    """Verify if any of the answers is present in the given context.
    Args:
        answers: list of short answers to look for in the context
        context: a paragraph to search for short answers
    Returns:
        true if any of the short answers is present in the context
    """


    for ans in n_answers:
        if ans in n_context:
            return True

    return False



# Combine compute_str_em and compute_qampari_f1 in eval.py
def compute_str_em(sent, answers, existing_hits, dataset):
    pred_prec = 0
    new_hits = []

    pred = normalize_answer(sent)
    if len(pred) == 0:
        return False, []
    answers = [[normalize_answer(x) for x in ans] for ans in answers]
    flat_answers = [item for sublist in answers for item in sublist]
    # Use strict string matching in qampari. Refer to compute_qampari_f1 in eval.py
    # Use substring matching in asqa, as pred here is a sentence
    if dataset == "qampari":
        pred_prec = pred in flat_answers
    else:
        pred_prec = any([item in pred for item in flat_answers])
    for i, a in enumerate(answers):
        if exact_presence(a, pred) and i not in existing_hits:
            new_hits.append(i)

    return pred_prec, new_hits



class RewardModelBestN():
    def __init__(self, autoais_model_name, autoais_model_type,
            cit_rec_pos_rw=0.2,
            cit_rec_neg_rw=-0.2,
            cit_prec_pos_rw=0.2,
            cit_prec_neg_rw=-0.2,
            corr_rec_pos_rw=0.2,
            corr_rec_neg_rw=-0.2,
            inference=False,
        ):
        self.autoais_model_name = autoais_model_name
        self.autoais_model_type = autoais_model_type
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.autoais_model = None
        self.autoais_tokenizer = None

        import logging
        self.logger = logging.getLogger(__name__)
        self.init_autoais()
        self.cit_rec_pos_rw = cit_rec_pos_rw
        self.cit_rec_neg_rw = cit_rec_neg_rw
        self.cit_prec_pos_rw = cit_prec_pos_rw
        self.cit_prec_neg_rw = cit_prec_neg_rw
        self.corr_rec_pos_rw = corr_rec_pos_rw
        self.corr_rec_neg_rw = corr_rec_neg_rw

        self.at_most_citations = 3
        self.cot = False

        self.max_seq_len = 2048

        self.inference = inference

        print("Inference is set to {}. Note that setting inference to True will remove the correctness recall reward.".format(self.inference))

    def init_autoais(self):
        if self.autoais_model is not None:
            self.logger.info("Deleteing previous NLI model and initializing it again")
            del self.autoais_model
            if self.autoais_tokenizer is not None:
                del self.autoais_tokenizer
            
            torch.cuda.empty_cache()
            gc.collect()

            time.sleep(30)

        self.logger.info("Initializing the NLI model and the correspoding tokenizer")
        if self.autoais_model_type == "bf16":
            self.autoais_model = AutoModelForSeq2SeqLM.from_pretrained(self.autoais_model_name, torch_dtype=torch.bfloat16, max_memory=get_max_memory(), device_map="auto", offload_folder="offload", offload_state_dict = True)
        elif self.autoais_model_type == "4bit":
            self.autoais_model = AutoModelForSeq2SeqLM.from_pretrained(self.autoais_model_name, load_in_4bit=True, device_map=0)
        elif self.autoais_model_type == "bnb":
            self.autoais_model = AutoModelForSeq2SeqLM.from_pretrained(self.autoais_model_name, quantization_config=self.bnb_config)
        else:
            raise Exception("autoais_model_type {} is not valid".format(self.autoais_model_type))
        self.autoais_tokenizer = AutoTokenizer.from_pretrained(self.autoais_model_name, use_fast=False)

    def get_reward(self, data, generated_texts, num_returned_sequence, dataset):
        assert len(data) == len(generated_texts)

        exclude_idx = []
        for i in range(len(data)):
            data[i]["sampled_outputs"] = generated_texts[i]
            
            if num_returned_sequence != len(generated_texts[i]):
                exclude_idx.append(i)
                self.logger.warning("Sample {} has {} sequences. Will either repeat the last generated text or truncate the list of text to make it have {} sequences".format(i, len(generated_texts[i]), num_returned_sequence))
                for _ in range(max(num_returned_sequence - len(generated_texts[i]), 0)):
                    data[i]["sampled_outputs"].append(generated_texts[i][-1])
                data[i]["sampled_outputs"] = data[i]["sampled_outputs"][:num_returned_sequence]
            assert num_returned_sequence == len(generated_texts[i]), "{}: {}".format(i, generated_texts[i])
            data[i]["sampled_rewards"] = [0.] * len(generated_texts[i])
            data[i]["gpt_output"] = data[i]["output"]
            del data[i]["output"]

        self.logger.info("We remove all the pre/appended space/newlines and we truncate the answer by the first newline.")
        self.logger.info("We replace any on the fly search result to standard bracket citation format.")
        for r in range(num_returned_sequence):
            
            for i in range(len(data)):
                data[i]['output'] = data[i]["sampled_outputs"][r]
                data[i]['output'] = data[i]['output'].strip().split("\n")[0]
                data[i]['output'] = data[i]['output'].replace("<|im_end|>", "")

            # Remove all citations for all non-AutoAIS evaluation
            normalized_data = copy.deepcopy(data)
            for i in range(len(normalized_data)):
                normalized_data[i]['output'] = remove_citations(normalized_data[i]['output'])

            if dataset == "combined":
                total_count = 0
                for name in ["asqa", "qampari", "eli5"]:
                    idx_map = [i for i in range(len(data)) if data[i]["dataset"] == name]
                    curr_data = [data[i] for i in idx_map]
                    curr_normalized_data = [normalized_data[i] for i in idx_map]
                    
                    self.get_single_reward(curr_data, curr_normalized_data, name, r)
                    total_count = total_count + len(idx_map)
                assert total_count == len(data)
            else:
                self.get_single_reward(data, normalized_data, dataset, r)

        for i in range(len(data)):
            output = max([(data[i]["sampled_rewards"][r], data[i]["sampled_outputs"][r]) for r in range(num_returned_sequence)], key=lambda x:x[0])
            data[i]["output"] = [output[1]]
            
    def get_single_reward(self, data, normalized_data, dataset, r):
        cit_rec_rw, cit_prec_rw = self.compute_autoais_best_n(data, qampari=dataset == "qampari", at_most_citations=self.at_most_citations)

        if dataset == "asqa":
            corr_rec_rw = self.compute_str_em_best_n(normalized_data, doc_rec=None, dataset=dataset)
        elif dataset == "qampari":
            corr_rec_rw = self.compute_qampari_f1_best_n(normalized_data, cot=self.cot, doc_rec=None)
        elif dataset == "eli5":
            corr_rec_rw = self.compute_claims_best_n(normalized_data, doc_rec=None)
        else:
            raise Exception("{} is not valid".format(dataset))

        assert len(cit_rec_rw) == len(data) and len(cit_prec_rw) == len(data) and len(corr_rec_rw) == len(data)
        
        # Modify the value in place
        for i in range(len(data)):
            if self.inference:
                data[i]["sampled_rewards"][r] = cit_rec_rw[i] + cit_prec_rw[i]
            else:
                data[i]["sampled_rewards"][r] = cit_rec_rw[i] + cit_prec_rw[i] + corr_rec_rw[i]
            del data[i]["output"]

        for inspect_idx in [0, 1, len(data)-1]:
            self.logger.info("Sample {} Question: {}".format(inspect_idx, data[inspect_idx]["question"]))
            self.logger.info("Sample {} Generated Text: {}".format(inspect_idx, data[inspect_idx]["sampled_outputs"][r]))
            self.logger.info("Sample {} Reward: {}".format(inspect_idx, data[inspect_idx]["sampled_rewards"][r]))
            self.logger.info("Sample {} Citation Recall: {}; Citation Precision: {} | Correctness Recall: {}".format(inspect_idx, cit_rec_rw[inspect_idx], cit_prec_rw[inspect_idx], corr_rec_rw[inspect_idx]))

        return cit_rec_rw, cit_prec_rw, corr_rec_rw


    def exact_presence_best_n(self, short_answers, context):
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



    def compute_str_em_best_n(self, data, doc_rec=None, dataset=None):
        """Compute STR-EM metric (only for ASQA)
        Args:
            data: requires field `qa_pairs/short_answers` and `output`
        Returns:
            STR-EM and STR-EM-HIT ()
        """

        if dataset != "asqa":
            return 0, 0, 0, 0

        corr_rec_rw = []


        for i, item in enumerate(data):
            loc_acc = []
            for qa_pair in item['qa_pairs']:
                loc_acc.append(self.exact_presence_best_n(qa_pair['short_answers'], item["output"]))
            reward_count = sum(loc_acc)
            penalize_count = len(loc_acc) - reward_count

            curr_corr_rec_rw = reward_count * self.corr_rec_pos_rw + penalize_count *self.corr_rec_neg_rw

            corr_rec_rw.append(curr_corr_rec_rw)
            
        return corr_rec_rw


    def _run_nli_autoais_best_n(self, passage, claim):
        """
        Run inference for assessing AIS between a premise and hypothesis.
        Adapted from https://github.com/google-research-datasets/Attributed-QA/blob/main/evaluation.py
        """

        if "true" in self.autoais_model_name:
            input_text = "premise: {} hypothesis: {}".format(passage, claim)
        elif "attrscore" in self.autoais_model_name:
            input_text = "As an Attribution Validator, your task is to verify whether a given reference can support the given claim. A claim can be either a plain sentence or a question followed by its answer. Specifically, your response should clearly indicate the relationship: Attributable, Contradictory or Extrapolatory. A contradictory error occurs when you can infer that the answer contradicts the fact presented in the context, while an extrapolatory error means that you cannot infer the correctness of the answer based on the information provided in the context. \n\nClaim: {} \n Reference: {}".format(claim, passage)
        else:
            raise Exception("Model {} is not defined".format(self.autoais_model_name))

        # May need to consider max_input_len
        if "true" in self.autoais_model_name:
            input_ids = self.autoais_tokenizer(input_text, return_tensors="pt", max_length=self.max_seq_len, truncation=True).input_ids.to(self.autoais_model.device)
        elif "attrscore" in self.autoais_model_name:
            input_ids = self.autoais_tokenizer(input_text, return_tensors="pt").input_ids.to(self.autoais_model.device)
        else:
            raise Exception("Model {} is not defined".format(self.autoais_model_name))
        with torch.inference_mode():
            outputs = self.autoais_model.generate(input_ids, max_new_tokens=10)
        result = self.autoais_tokenizer.decode(outputs[0], skip_special_tokens=True)


        if "true" in self.autoais_model_name:
            inference = 1 if result == "1" else 0
        elif "attrscore" in self.autoais_model_name:
            att_keys = [w.casefold() for w in ["Attributable"]]
            # Combine Extrapolatory and Contraditory into 1 single category
            inference = 1 if result.casefold() in att_keys else 0
        else:
            raise Exception("Model {} is not defined".format(self.autoais_model_name))
        return inference


    def compute_claims_best_n(self, data, doc_rec=None):
        self.logger.info("Computing claims...")

        corr_rec_rw = []


        for i in tqdm(range(len(data))):
            item = data[i]
            normalized_output = remove_citations(item['output'])
            claims = item["claims"]

            entails = []

            for claim in claims:
                if "attrscore" in self.autoais_model_name:
                    claim = "{} {}".format(item["question"], claim)
                entails.append(self._run_nli_autoais_best_n(normalized_output, claim))

            reward_count = sum(entails)
            penalize_count = len(claims) - reward_count

            curr_corr_rec_rw = reward_count * self.corr_rec_pos_rw + penalize_count *self.corr_rec_neg_rw

            corr_rec_rw.append(curr_corr_rec_rw)

        return corr_rec_rw


    def compute_autoais_best_n(self, data,
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

        self.logger.info(f"Running AutoAIS...")

        def _format_document(doc):
            """Format document for AutoAIS."""

            if "sent" in doc:
                # QA-extracted docs
                return "Title: %s\n%s" % (doc['title'], doc['sent'])
            else:
                return "Title: %s\n%s" % (doc['title'], doc['text'])

        cit_rec_rw = []
        cit_prec_rw = []

        num_sents = []

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
                if "true" in self.autoais_model_name:
                    sents = sent_tokenize(item['output'])
                elif "attrscore" in self.autoais_model_name:
                    sents = ["{} {}".format(item["question"], sent) for sent in sent_tokenize(item['output'])]
                else:
                    raise Exception("Model {} is not defined".format(self.autoais_model_name))

            if len(sents) == 0:
                self.logger.info("Response to question {} is empty. Will set the citation recall reward and citation precision reward to 0 for this response.".format(item['question']))
                cit_rec_rw.append(0)
                cit_prec_rw.append(0)
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
                self.logger.info(f"For `{sent}`, find citations {ref}")
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
                    joint_entail = self._run_nli_autoais_best_n(joint_passage, target_sent)
                    autoais_log.append({
                        "question": item['question'],
                        "output": item['output'],
                        "claim": sent if "true" in self.autoais_model_name else sent[len("{} ".format(item["question"])):],
                        "passage": [joint_passage],
                        "model_type": "NLI",
                        "{}_output".format(self.autoais_model_name): joint_entail,
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
                        nli_result = self._run_nli_autoais_best_n(passage, target_sent)

                        # condition B
                        if not nli_result:
                            subset_exclude = copy.deepcopy(ref)
                            subset_exclude.remove(psgs_id)
                            passage = '\n'.join([_format_document(item['docs'][pid]) for pid in subset_exclude])
                            nli_result = self._run_nli_autoais_best_n(passage, target_sent)
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

            cit_rec_reward_count = entail
            cit_rec_penalize_count = len(sents) - entail
            curr_cit_rec_rw = cit_rec_reward_count * self.cit_rec_pos_rw + cit_rec_penalize_count * self.cit_rec_neg_rw

            cit_rec_rw.append(curr_cit_rec_rw)

            cit_prec_reward_count = entail_prec
            cit_prec_penalize_count = total_citations - entail_prec
            curr_cit_prec_ew = cit_prec_reward_count * self.cit_prec_pos_rw + cit_prec_penalize_count * self.cit_prec_neg_rw

            cit_prec_rw.append(curr_cit_prec_ew)

        if sent_mcite > 0 and sent_mcite_support > 0:
            self.logger.info("Among all sentences, %.2f%% have multiple citations, among which %.2f%% are supported by the joint set, among which %.2f%% overcite." % (
                100 * sent_mcite / sent_total, 
                100 * sent_mcite_support / sent_mcite, 
                100 * sent_mcite_overcite / sent_mcite_support
            ))
        
        return cit_rec_rw, cit_prec_rw


    def compute_qampari_f1_best_n(self, data, cot=False, doc_rec=None):
        corr_rec_rw = []

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
            answers = [[normalize_answer(x) for x in ans] for ans in item['answers']]

            hits = [any([x in preds for x in a]) for a in answers]

            reward_count = sum(hits)
            penalize_count = max(min(5, len(answers)) - reward_count, 0)

            curr_corr_rec_rw = reward_count * self.corr_rec_pos_rw + penalize_count *self.corr_rec_neg_rw

            corr_rec_rw.append(curr_corr_rec_rw)


        return corr_rec_rw



class RewardModelBeamSearch():
    def __init__(self, autoais_model_name, autoais_model_type,
            cit_rec_pos_rw=0.2,
            cit_rec_neg_rw=-0.2,
            cit_prec_pos_rw=0.2,
            cit_prec_neg_rw=-0.2,
            corr_rec_pos_rw=0.2,
            corr_rec_neg_rw=-0.2,
            device=1,
            inference=False,
            no_citation_reward=False,
            no_correctness_reward=False,
            init_autoais=True,
        ):
        self.autoais_model_name = autoais_model_name
        self.autoais_model_type = autoais_model_type
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.autoais_model = None
        self.autoais_tokenizer = None

        import logging
        self.logger = logging.getLogger(__name__)

        if init_autoais:
            self.init_autoais(device)
        else:
            self.logger.info("The NLI model is not initalized. Make sure that the dataset is either ASQA or QAMPARI and the citation reward is exluded")

        self.cit_rec_pos_rw = cit_rec_pos_rw
        self.cit_rec_neg_rw = cit_rec_neg_rw
        self.cit_prec_pos_rw = cit_prec_pos_rw
        self.cit_prec_neg_rw = cit_prec_neg_rw
        self.corr_rec_pos_rw = corr_rec_pos_rw
        self.corr_rec_neg_rw = corr_rec_neg_rw

        self.at_most_citations = 3
        self.cot = False

        self.max_seq_len = 2048

        self.inference = inference

        print("Inference is set to {}. Note that setting inference to True will remove the correctness recall reward.".format(self.inference))

        assert not no_citation_reward or not no_correctness_reward
        self.no_citation_reward = no_citation_reward
        self.no_correctness_reward = no_correctness_reward

        print("no_citation_reward: {} | no_correctness_reward: {}".format(self.no_citation_reward, self.no_correctness_reward))

    def init_autoais(self, device=1):
        if self.autoais_model is not None:
            self.logger.info("Deleteing previous NLI model and initializing it again")
            del self.autoais_model
            if self.autoais_tokenizer is not None:
                del self.autoais_tokenizer
            
            torch.cuda.empty_cache()
            gc.collect()

            time.sleep(30)

        self.logger.info("Initializing the NLI model and the correspoding tokenizer")
        if self.autoais_model_type == "bf16":
            self.autoais_model = AutoModelForSeq2SeqLM.from_pretrained(self.autoais_model_name, torch_dtype=torch.bfloat16, device_map=device, offload_folder="offload", offload_state_dict = True)
        elif self.autoais_model_type == "4bit":
            self.autoais_model = AutoModelForSeq2SeqLM.from_pretrained(self.autoais_model_name, load_in_4bit=True, device_map=device)
        elif self.autoais_model_type == "bnb":
            self.autoais_model = AutoModelForSeq2SeqLM.from_pretrained(self.autoais_model_name, quantization_config=self.bnb_config)
        else:
            raise Exception("autoais_model_type {} is not valid".format(self.autoais_model_type))
        self.autoais_tokenizer = AutoTokenizer.from_pretrained(self.autoais_model_name, use_fast=False)

    def get_reward(self, item, seq, pred_text, existing_hits):

        # self.logger.info("We remove all the pre/appended space/newlines and we truncate the answer by the first newline.")
        # self.logger.info("We replace any on the fly search result to standard bracket citation format.")
            
        item['output'] = pred_text.replace("<|im_end|>", "")

        # Remove all citations for all non-AutoAIS evaluation
        normalized_item = copy.deepcopy(item)
        normalized_item['output'] = remove_citations(normalized_item['output'])

        r = self.get_single_item_reward(item, normalized_item, existing_hits)

        # print("get_reward: {}".format(r))

        return r


    def get_single_item_reward(self, item, normalized_item, existing_hits):
        dataset = item["dataset"]
        if self.no_citation_reward:
            cit_rec_rw_change = 0
            cit_prec_rw_change = 0
            entail = 0
            num_sents = 0
            entail_prec = 0
            total_citations = 0
        else:
            cit_rec_rw_change, cit_prec_rw_change, entail, num_sents, entail_prec, total_citations = self.compute_autoais_beam_search(item, qampari=dataset == "qampari", at_most_citations=self.at_most_citations)

        if self.inference or self.no_correctness_reward:
            corr_rec_rw_change = 0
            new_hits = []
        else:
            if dataset == "asqa":
                corr_rec_rw_change, new_hits = self.compute_str_em_beam_search(normalized_item, existing_hits=existing_hits)
            elif dataset == "qampari":
                corr_rec_rw_change, new_hits = self.compute_qampari_f1_beam_search(normalized_item, cot=self.cot, existing_hits=existing_hits)
            elif dataset == "eli5":
                corr_rec_rw_change, new_hits = self.compute_claims_beam_search(normalized_item, existing_hits=existing_hits)
            else:
                raise Exception("{} is not valid".format(dataset))
        
        # The correctness recall reward increase for 1 hit will be equal to self.corr_rec_pos_rw-self.corr_rec_neg_rw
        # If there is no new hit, then the correctness recall reward increase is 0.
        # Setting the initial correctness recall reward to num_ans * self.corr_rec_neg_rw, the eventual reward value will be the same
        
        if self.inference or self.no_correctness_reward:
            rw_change = cit_rec_rw_change + cit_prec_rw_change
        elif self.no_citation_reward:
            rw_change = corr_rec_rw_change
        else:
            rw_change = cit_rec_rw_change + cit_prec_rw_change + corr_rec_rw_change
        del item["output"]

        return {
            "rw_change": rw_change,
            "cit_rec_rw_change": cit_rec_rw_change,
            "cit_prec_rw_change": cit_prec_rw_change,
            "corr_rec_rw_change": corr_rec_rw_change,
            "entail": entail,
            "num_sents": num_sents,
            "entail_prec": entail_prec,
            "total_citations": total_citations,
            "new_hits": new_hits,
        }


    def exact_presence_beam_search(self, short_answers, context):
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



    def compute_str_em_beam_search(self, item, existing_hits=[]):
        """Compute STR-EM metric (only for ASQA)
        Args:
            data: requires field `qa_pairs/short_answers` and `output`
        Returns:
            STR-EM and STR-EM-HIT ()
        """

        if item["dataset"] != "asqa":
            raise Exception("compute_str_em_best_n is only for ASQA")

        loc_acc = []
        new_hits = []
        for h, qa_pair in enumerate(item['qa_pairs']):
            if h in existing_hits:
                continue
            if self.exact_presence_beam_search(qa_pair['short_answers'], item["output"]):
                new_hits.append(h)

        
        corr_rec_rw_change = len(new_hits) * (self.corr_rec_pos_rw-self.corr_rec_neg_rw)
        return corr_rec_rw_change, new_hits


    def _run_nli_autoais_beam_search(self, passage, claim):
        """
        Run inference for assessing AIS between a premise and hypothesis.
        Adapted from https://github.com/google-research-datasets/Attributed-QA/blob/main/evaluation.py
        """

        if "true" in self.autoais_model_name:
            input_text = "premise: {} hypothesis: {}".format(passage, claim)
        elif "attrscore" in self.autoais_model_name:
            input_text = "As an Attribution Validator, your task is to verify whether a given reference can support the given claim. A claim can be either a plain sentence or a question followed by its answer. Specifically, your response should clearly indicate the relationship: Attributable, Contradictory or Extrapolatory. A contradictory error occurs when you can infer that the answer contradicts the fact presented in the context, while an extrapolatory error means that you cannot infer the correctness of the answer based on the information provided in the context. \n\nClaim: {} \n Reference: {}".format(claim, passage)
        else:
            raise Exception("Model {} is not defined".format(self.autoais_model_name))

        # May need to consider max_input_len
        if "true" in self.autoais_model_name:
            input_ids = self.autoais_tokenizer(input_text, return_tensors="pt", max_length=self.max_seq_len, truncation=True).input_ids.to(self.autoais_model.device)
        elif "attrscore" in self.autoais_model_name:
            input_ids = self.autoais_tokenizer(input_text, return_tensors="pt").input_ids.to(self.autoais_model.device)
        else:
            raise Exception("Model {} is not defined".format(self.autoais_model_name))
        with torch.inference_mode():
            outputs = self.autoais_model.generate(input_ids, max_new_tokens=10)
        result = self.autoais_tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "true" in self.autoais_model_name:
            inference = 1 if result == "1" else 0
        elif "attrscore" in self.autoais_model_name:
            att_keys = [w.casefold() for w in ["Attributable"]]
            # Combine Extrapolatory and Contraditory into 1 single category
            inference = 1 if result.casefold() in att_keys else 0
        else:
            raise Exception("Model {} is not defined".format(self.autoais_model_name))
        return inference


    def compute_claims_beam_search(self, item, existing_hits=[]):
        # self.logger.info("Computing claims...")

        normalized_output = remove_citations(item['output'])
        claims = item["claims"]

        new_hits = []

        for h, claim in enumerate(claims):
            if h in existing_hits:
                continue
            if "attrscore" in self.autoais_model_name:
                claim = "{} {}".format(item["question"], claim)
            if self._run_nli_autoais_beam_search(normalized_output, claim):
                new_hits.append(h)


        corr_rec_rw_change = len(new_hits) * (self.corr_rec_pos_rw-self.corr_rec_neg_rw)
        return corr_rec_rw_change, new_hits


    def compute_autoais_beam_search(self, item,
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

        # self.logger.info(f"Running AutoAIS...")

        def _format_document(doc):
            """Format document for AutoAIS."""

            if "sent" in doc:
                # QA-extracted docs
                return "Title: %s\n%s" % (doc['title'], doc['sent'])
            else:
                return "Title: %s\n%s" % (doc['title'], doc['text'])

        # Get sentences by using NLTK
        if qampari:
            sents = [item['question'] + " " + x.strip() for x in item['output'].rstrip().rstrip(".").rstrip(",").split(",")]
        else:
            if "true" in self.autoais_model_name:
                sents = sent_tokenize(item['output'])
            elif "attrscore" in self.autoais_model_name:
                sents = ["{} {}".format(item["question"], sent) for sent in sent_tokenize(item['output'])]
            else:
                raise Exception("Model {} is not defined".format(self.autoais_model_name))

        if len(sents) == 0:
            self.logger.info("Response to question {} is empty. Will set the citation recall reward and citation precision reward to 0 for this response.".format(item['question']))
            return 0, 0, 0, 0, 0, 0

        target_sents = [remove_citations(sent).strip() for sent in sents]

        entail = 0
        entail_prec = 0
        total_citations = 0
        for sent_id, sent in enumerate(sents):
            target_sent = target_sents[sent_id] # Citation removed and (if opted for) decontextualized
            joint_entail = -1 # Undecided

            # Find references
            ref = [int(r[1:])-1 for r in re.findall(r"\[\d+", sent)] # In text citation id starts from 1
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
                joint_entail = self._run_nli_autoais_beam_search(joint_passage, target_sent)

            entail += joint_entail
            # if len(ref) > 1:
            #     sent_mcite += 1

            # calculate the precision score if applicable
            if joint_entail and len(ref) > 1:
                # sent_mcite_support += 1
                # Precision check: did the model cite any unnecessary documents?
                for psgs_id in ref:
                    # condition A
                    passage = _format_document(item['docs'][psgs_id]) 
                    nli_result = self._run_nli_autoais_beam_search(passage, target_sent)

                    # condition B
                    if not nli_result:
                        subset_exclude = copy.deepcopy(ref)
                        subset_exclude.remove(psgs_id)
                        passage = '\n'.join([_format_document(item['docs'][pid]) for pid in subset_exclude])
                        nli_result = self._run_nli_autoais_beam_search(passage, target_sent)
                        if nli_result: # psgs_id is not necessary
                            flag = 0
                            # sent_mcite_overcite += 1 
                        else:
                            entail_prec += 1
                    else:
                        entail_prec += 1
            else:
                entail_prec += joint_entail 


        cit_rec_reward_count = entail
        cit_rec_penalize_count = len(sents) - entail
        curr_cit_rec_rw_change = cit_rec_reward_count * self.cit_rec_pos_rw + cit_rec_penalize_count * self.cit_rec_neg_rw


        cit_prec_reward_count = entail_prec
        cit_prec_penalize_count = total_citations - entail_prec
        curr_cit_prec_ew_change = cit_prec_reward_count * self.cit_prec_pos_rw + cit_prec_penalize_count * self.cit_prec_neg_rw

        
        return curr_cit_rec_rw_change, curr_cit_prec_ew_change, entail, len(sents), entail_prec, total_citations


    def compute_qampari_f1_beam_search(self, item, cot=False, existing_hits=[]):

        if cot:
            if ":" in item['output']:
                o = ':'.join(item['output'].split(":")[1:]) # try to separate the COT part and the answer list part.
            else:
                o = ""
        else:
            o = item['output']
        preds = [normalize_answer(x.strip()) for x in o.rstrip().rstrip(".").rstrip(",").split(",")]
        preds = [p for p in preds if len(p) > 0] # delete empty answers
        answers = [[normalize_answer(x) for x in ans] for ans in item['answers']]


        new_hits = []

        for h, a in enumerate(answers):
            if h in existing_hits:
                continue
            if any([x in preds for x in a]):
                new_hits.append(h)


        if len(existing_hits) < 5:
            if len(existing_hits) + len(new_hits) <= 5:
                corr_rec_rw_change = len(new_hits) * (self.corr_rec_pos_rw-self.corr_rec_neg_rw)
            else:
                corr_rec_rw_change = (5 - len(existing_hits)) * (self.corr_rec_pos_rw-self.corr_rec_neg_rw) + (len(existing_hits) + len(new_hits) - 5) * self.corr_rec_pos_rw
        else:
            corr_rec_rw_change = len(new_hits) * self.corr_rec_pos_rw

        return corr_rec_rw_change, new_hits 
