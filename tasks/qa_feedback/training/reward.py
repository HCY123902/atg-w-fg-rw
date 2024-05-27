from cgitb import text
import json
import os
from socket import NI_NUMERICHOST
from typing import Optional, List, Iterable, Dict, Any, Tuple

from dateparser import data
from fgrlhf import reward_utils
import torch, spacy
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, BitsAndBytesConfig
import abc
import numpy as np
import logging
import re

from fgrlhf.reward import BasicReward
from fgrlhf.reward_utils import *
from fgrlhf.evaluators import get_rouge_scores, get_single_rouge_score
from fgrlhf.utils import get_bnb_config, DEFAULT_DEVICE_RANK

import mauve
import string
import copy

logging.basicConfig(level=logging.ERROR)

class NLIModel:
    def __init__(self, ckpt, log_info, accelerator, device_rank=DEFAULT_DEVICE_RANK):
        log_info("Loading AutoAIS model...")
        # Try to avoid offloading with QLoRA
        # May need to offload to different directories for different processed
        self.ckpt = ckpt

        compute_type = getattr(torch, "bfloat16")
        bnb_config = get_bnb_config(compute_type)

        if accelerator is not None:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(ckpt, quantization_config=bnb_config)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(ckpt, device_map={"":device_rank}, quantization_config=bnb_config)
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt, use_fast=False)
        self.accelerator = accelerator

        self.max_seq_len = 2048

    def run_nli_autoais(self, passage, claim):
        """
        Run inference for assessing AIS between a premise and hypothesis.
        Adapted from https://github.com/google-research-datasets/Attributed-QA/blob/main/evaluation.py
        """
        
        if "true" in self.ckpt:
            input_text = "premise: {} hypothesis: {}".format(passage, claim)
        elif "attrscore" in self.ckpt:
            input_text = "As an Attribution Validator, your task is to verify whether a given reference can support the given claim. A claim can be either a plain sentence or a question followed by its answer. Specifically, your response should clearly indicate the relationship: Attributable, Contradictory or Extrapolatory. A contradictory error occurs when you can infer that the answer contradicts the fact presented in the context, while an extrapolatory error means that you cannot infer the correctness of the answer based on the information provided in the context. \n\nClaim: {} \n Reference: {}".format(claim, passage)
        else:
            raise Exception("Model {} is not defined".format(self.ckpt))

        # May need to consider max_input_len
        if "true" in self.ckpt:
            input_ids = self.tokenizer(input_text, return_tensors="pt", max_length=self.max_seq_len, truncation=True).input_ids.to(self.model.device)
        elif "attrscore" in self.ckpt:
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.model.device)
        else:
            raise Exception("Model {} is not defined".format(self.ckpt))
        with torch.inference_mode():
            if self.accelerator is not None:
                unwrapped_model = self.accelerator.unwrap_model(self.model)
            else:
                unwrapped_model = self.model
            outputs = unwrapped_model.generate(input_ids, max_new_tokens=10)
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # print("Result: {}\n".format(result))

        if "true" in self.ckpt:
            inference = 1 if result == "1" else 0
        elif "attrscore" in self.ckpt:
            att_keys = [w.casefold() for w in ["Attributable"]]
            # Combine Extrapolatory and Contraditory into 1 single category
            inference = 1 if result.casefold() in att_keys else 0
        else:
            raise Exception("Model {} is not defined".format(self.ckpt))
        return inference

    # Do not need to call normalize_answer for sent and claims here
    def compute_claims(self, sent, claims, question, existing_hits):
        # logger.info("Computing claims...")
        pred_prec = 0
        new_hits = []

        normalized_output = remove_citations(sent)
        entail = 0
        for i, claim in enumerate(claims):
            if "attrscore" in self.ckpt:
                claim = "{} {}".format(question, claim)
            entail = self.run_nli_autoais(normalized_output, claim)
            if entail:
                pred_prec = 1
                if i not in existing_hits:
                    new_hits.append(i)

        return pred_prec, new_hits


# Not currently used
class SequenceFluencyReward:
    def __init__(self,
                 tokenizer,
                #  model_ckpt,
                 mean = 0.0,
                 std = 1.0,
                 bias = 0.0,
                 scale = 1.0,
                 ):
        
        # use mean and std to normalize the reward
        # use bias and scale to rescale the reward
        # prepare policy tokenizer
        self.policy_tokenizer = tokenizer
        
        self.mean = mean
        self.std = std
        
        self.bias = bias
        self.scale = scale
        
    def get_reward(self, 
                   prompts_input_ids: torch.tensor, 
                   prompts_attention_mask: torch.tensor, 
                   generated_input_ids: torch.tensor, # (B, output_len)
                   generated_attention_mask: torch.tensor, # (B, output_len)
                   normalized_generated_texts: List[str],
                   metadata=None,
                   ):

        sequence_level_reward = []

        for batch_idx, (meta, gen_text) in enumerate(zip(metadata, normalized_generated_texts)):

            # Exclude Rouge at the moment
            if meta["dataset"] != "":
            # if meta["dataset"] == "qampari":
                # TODO: Check the evental reward value
                sequence_level_reward.append(self.mean)
                continue

            rouge_score = get_single_rouge_score(gen_text, meta["human_answer"]) 

            sequence_level_reward.append(rouge_score)
        

        # align with generated texts, make it fine-grained
        fine_grained_reward = [
            [0.] * (l-1) + [((r-self.mean)/self.std)*self.scale + self.bias]
            for r, l in zip(sequence_level_reward, torch.sum(generated_attention_mask, dim=1).tolist())
        ]
        
        return fine_grained_reward



class SequenceLevelCorrectnessReward:
    def __init__(self,
                tokenizer,
                correctness_positive_reward,
                correctness_negative_reward,
                # sep = "</s>",
                cot=False,
                nli_model=None,
                asqa_multiplier=None,
                ):
        
        # prepare policy tokenizer
        self.policy_tokenizer = tokenizer
        
        # rewards
        self.correctness_positive_reward = correctness_positive_reward
        self.correctness_negative_reward = correctness_negative_reward

        self.cot = cot
        self.nli_model = nli_model

        self.asqa_multiplier = asqa_multiplier

    def process_one_generation(self, long_text, meta):
        if meta["dataset"] == "qampari":
            # Consideration: Do not use split_text_to_subsentences as it does not work well when citations are present
            sents, sentence_end_char_idxs, refs = split_qampari_text(long_text)
        else:
            # Consideration: Do not use split_text_to_sentences as it does not work well when citations are present
            sents, sentence_end_char_idxs, refs = split_text(long_text)

        if meta["dataset"] == "qampari":
            # Check how pred is first processed with remove_citations in normalized_data in eval.py and then stripped with compute_qampari_f1; Do not include question for each answer
            sents = [remove_citations(long_text[sentence_end_char_idxs[i]:sentence_end_char_idxs[i+1]]).rstrip().rstrip(".").rstrip(",").strip() for i in range(len(sentence_end_char_idxs)-1)]
        else:
            # ASQA: Only need to remove citations, check normalized_data and compute_str_em in eval.py
            # ELI5: Only need to remove citations, check normalized_data and compute_claims in eval.py; Question is prepended to claim, no output
            sents = [remove_citations(sent) for sent in sents]
        return sents

        
    def get_reward(self, 
            prompts_input_ids: torch.tensor, 
            prompts_attention_mask: torch.tensor, 
            generated_input_ids: torch.tensor, # (B, output_len)
            generated_attention_mask: torch.tensor, # (B, output_len)
            generated_texts: List[str],
            normalized_generated_texts: List[str],
            metadata=None,
        ):
        
        sequence_level_reward = []
        n_total_hits = []
        n_answers = []
        n_sentences_with_new_hits = []
        n_sentences = []

        for batch_idx, (meta, gen_text, normalized_gen_text) in enumerate(zip(metadata, generated_texts, normalized_generated_texts)):

            sents = self.process_one_generation(gen_text, meta)

            dataset = meta["dataset"]
            answers = [qa_pair["short_answers"] for qa_pair in meta["qa_pairs"]] if dataset == "asqa" else (meta["answers"] if dataset == "qampari" else meta["claims"])
            
            n_answers.append(len(answers))
            n_sent_with_new_hits = 0
            n_sentences.append(len(sents))

            if dataset == "eli5":
                sent_prec, new_hits = self.nli_model.compute_claims(normalized_gen_text, answers, meta["question"], [])

            else:
                # Ignore QAMPARI sent_prec. It is not used. It is neither correct as it attempts to match the entire generated text, which contains comma, with each of the gold answers exactly
                sent_prec, new_hits = compute_str_em(normalized_gen_text, answers, [], dataset)

            # # Option 1: New sequence level reward
            # Each QAMPARI sample has at least 5 answers, which means max(min(5, len(answers)) - len(new_hits), 0) is equal to max(5 - len(new_hits), 0)
            penalize_count = (len(answers) - len(new_hits)) if dataset != "qampari" else max(5 - len(new_hits), 0)
            correctness_recall_reward = len(new_hits) * self.correctness_positive_reward + penalize_count * self.correctness_negative_reward
            
            if dataset == "asqa" and self.asqa_multiplier is not None:
                correctness_recall_reward = correctness_recall_reward * self.asqa_multiplier

            # # Option 2: New sequence level reward with doc_rec_ans
            # num_doc_rec_ans = len(meta['doc_rec_ans'])
            # correctness_recall_reward = len(new_hits) * self.correctness_positive_reward + max(num_doc_rec_ans - len(new_hits), 0) * self.correctness_negative_reward

            # # Scale it to correspond to correctness_recall
            # correctness_recall_reward = (correctness_recall_reward / len(answers)) if dataset != "qampari else (correctness_recall_reward / max(5, len(answers)))

            sequence_level_reward.append(correctness_recall_reward)

            n_sentences_with_new_hits.append(n_sent_with_new_hits)
            n_total_hits.append(len(new_hits))

        # align with generated texts, make it fine-grained
        fine_grained_reward = [
            [0.] * (l-1) + [r]
            for r, l in zip(sequence_level_reward, torch.sum(generated_attention_mask, dim=1).tolist())
        ]
        
        return {
            "correctness_rewards": fine_grained_reward,
            "n_total_hits": n_total_hits,
            "n_answers": n_answers,
            "n_sentences_with_new_hits": n_sentences_with_new_hits,
            "n_sentences": n_sentences, 
        }


# Not currently used
class CorrectnessReward:
    def __init__(self,
                tokenizer,
                correctness_positive_reward,
                correctness_negative_reward,
                # sep = "</s>",
                cot=False,
                nli_model=None,
                asqa_multiplier=None,
    ):
        
        # prepare policy tokenizer
        self.policy_tokenizer = tokenizer
        
        # rewards
        self.correctness_positive_reward = correctness_positive_reward
        self.correctness_negative_reward = correctness_negative_reward

        self.cot = cot
        self.nli_model = nli_model

        self.asqa_multiplier = asqa_multiplier

    def find_sep_position(self, input_ids):
        return torch.nonzero(input_ids == self.sep_id, as_tuple=False).squeeze().tolist()
    
    def process_one_generation(self, long_text, policy_text_len, meta):
        
        if meta["dataset"] == "qampari":
            # Consideration: Do not use split_text_to_subsentences as it does not work well when citations are present
            sents, sentence_end_char_idxs, refs = split_qampari_text(long_text)
        else:
            # Consideration: Do not use split_text_to_sentences as it does not work well when citations are present
            sents, sentence_end_char_idxs, refs = split_text(long_text)
        
        # Initialize an empty list to store the end token indices of each sentence
        sentence_end_indices = []
    
        for sent_idx in range(len(sents)):
            tokens = self.policy_tokenizer.tokenize(long_text[:sentence_end_char_idxs[sent_idx+1]])
            token_count = len(tokens)
            # tokenizer.tokenize does not add the </s> token; Using token_count - 1 as the index to assign reward value to the last token of the sentence
            sentence_end_indices.append(token_count - 1)
        

        # in case overlength    
        sentence_end_indices = [min(item, policy_text_len-1) for item in sentence_end_indices]

        if meta["dataset"] == "qampari":
            # Check how pred is first processed with remove_citations in normalized_data in eval.py and then stripped with compute_qampari_f1; Do not include question for each answer
            sents = [remove_citations(long_text[sentence_end_char_idxs[i]:sentence_end_char_idxs[i+1]]).rstrip().rstrip(".").rstrip(",").strip() for i in range(len(sentence_end_char_idxs)-1)]
        else:
            # ASQA: Only need to remove citations, check normalized_data and compute_str_em in eval.py
            # ELI5: Only need to remove citations, check normalized_data and compute_claims in eval.py; Question is prepended to claim, no output
            sents = [remove_citations(sent) for sent in sents]
    
        assert len(sentence_end_indices) == len(sents)

        return sents, sentence_end_indices
    
    
    def get_reward(self, 
            prompts_input_ids: torch.tensor, 
            prompts_attention_mask: torch.tensor, 
            generated_input_ids: torch.tensor, # (B, output_len)
            generated_attention_mask: torch.tensor, # (B, output_len)
            generated_texts: List[str],
            # normalized_generated_texts: List[str],
            metadata=None
        ):
        
        # get the length of generated outputs
        policy_inputs_lens = torch.sum(generated_attention_mask, dim=1).tolist()

        correctness_recall_rewards = []
        correctness_precision_rewards = []
        n_total_hits = []
        n_answers = []
        n_sentences_with_new_hits = []
        n_sentences = []

        for batch_idx, (meta, gen_text) in enumerate(zip(metadata, generated_texts)):

            sents, sentence_end_indices = self.process_one_generation(gen_text, 
                                                    policy_inputs_lens[batch_idx], meta)

            correctness_recall_reward = [0]*policy_inputs_lens[batch_idx]
            correctness_precision_reward = [0]*policy_inputs_lens[batch_idx]

            dataset = meta["dataset"]
            answers = [qa_pair["short_answers"] for qa_pair in meta["qa_pairs"]] if dataset == "asqa" else (meta["answers"] if dataset == "qampari" else meta["claims"])
            
            n_answers.append(len(answers))
            n_sent_with_new_hits = 0
            n_sentences.append(len(sents))

            if dataset == "qampari":
                sentence = ", ".join(sents)
                if self.cot:
                    if ":" in sentence:
                        o = ':'.join(sentence.split(":")[1:]) # try to separate the COT part and the answer list part.
                    else:
                        o = ""
                else:
                    o = sentence
                sents = o.split(", ")
            
            existing_hits = []
            for sent_id, sent in enumerate(sents):
                if dataset == "eli5":
                    sent_prec, new_hits = self.nli_model.compute_claims(sent, answers, meta["question"], existing_hits)
                else:
                    sent_prec, new_hits = compute_str_em(sent, answers, existing_hits, dataset)
                existing_hits.extend(new_hits)
                end_token_idx = sentence_end_indices[sent_id]
                correctness_recall_reward[end_token_idx] = len(new_hits) * self.correctness_positive_reward if len(new_hits) > 0 else self.correctness_negative_reward
                if sent_prec:
                    correctness_precision_reward[end_token_idx] = self.correctness_positive_reward
                    n_sent_with_new_hits = n_sent_with_new_hits + 1
                else:
                    correctness_precision_reward[end_token_idx] = self.correctness_negative_reward
            
            if dataset == "asqa" and self.asqa_multiplier is not None:
                correctness_recall_reward = [r * self.asqa_multiplier for r in correctness_recall_reward]
                correctness_precision_reward = [r * self.asqa_multiplier for r in correctness_precision_reward]

            correctness_recall_rewards.append(correctness_recall_reward)
            correctness_precision_rewards.append(correctness_precision_reward)
            n_sentences_with_new_hits.append(n_sent_with_new_hits)

            assert len(existing_hits) == len(set(existing_hits))
            n_total_hits.append(len(existing_hits))

        # Do not use correctness precision for now as it may not reflect the holisticity of the answer. There can be an answer that has multiple sentences hitting the same gold answer.
        return {
            "correctness_rewards": correctness_recall_rewards,
            "n_total_hits": n_total_hits,
            "n_answers": n_answers,
            "n_sentences_with_new_hits": n_sentences_with_new_hits,
            "n_sentences": n_sentences, 
        }




class CitationRecallAndPrecisionReward:
    def __init__(self,
                tokenizer,
                # model_ckpt,
                citation_recall_positive_reward = 1.0,
                citation_recall_negative_reward = -1.0,
                citation_precision_positive_reward = 1.0,
                citation_precision_negative_reward = -1.0,
                at_most_citations=None,
                # sep = "</s>",
                nli_model=None,
                asqa_multiplier=None,
        ):
        
        # prepare policy tokenizer
        self.policy_tokenizer = tokenizer
        
        # # rewards
        self.citation_recall_positive_reward = citation_recall_positive_reward
        self.citation_recall_negative_reward = citation_recall_negative_reward
        self.citation_precision_positive_reward = citation_precision_positive_reward
        self.citation_precision_negative_reward = citation_precision_negative_reward
        self.at_most_citations = at_most_citations

        self.nli_model = nli_model

        self.asqa_multiplier = asqa_multiplier

    def find_sep_position(self, input_ids):
        return torch.nonzero(input_ids == self.sep_id, as_tuple=False).squeeze().tolist()
    
    def process_one_generation(self, long_text, policy_text_len, meta):
        
        if meta["dataset"] == "qampari":
            # Consideration: Do not use split_text_to_subsentences as it does not work well when citations are present
            sents, sentence_end_char_idxs, refs = split_qampari_text(long_text)
        else:
            # Consideration: Do not use split_text_to_sentences as it does not work well when citations are present
            sents, sentence_end_char_idxs, refs = split_text(long_text)
        
        # Initialize an empty list to store the end token indices of each sentence
        sentence_end_indices = []
        citation_end_indices = []
    
        for sent_idx in range(len(sents)):
            tokens = self.policy_tokenizer.tokenize(long_text[:sentence_end_char_idxs[sent_idx+1]])
            token_count = len(tokens)
            # tokenizer.tokenize does not add the </s> token; Using token_count - 1 as the index to assign reward value to the last token of the sentence
            sentence_end_indices.append(token_count - 1)

        for ref in refs[1:]:
            citation_end_idx = []
            for char_idx, _ in ref:
                tokens = self.policy_tokenizer.tokenize(long_text[:char_idx])
                token_count = len(tokens)
                citation_end_idx.append(token_count - 1)
            citation_end_indices.append(citation_end_idx)
        
        referneces = [[psg_id for _, psg_id in ref] for ref in refs[1:]]

        # in case overlength    
        sentence_end_indices = [min(item, policy_text_len-1) for item in sentence_end_indices]
        citation_end_indices = [[min(item, policy_text_len-1) for item in citation_end_idx] for citation_end_idx in citation_end_indices]
    
        # Check eval.py compute_autoais
        if meta["dataset"] == "qampari":
            sents = [meta["question"] + " " + long_text[sentence_end_char_idxs[i]:sentence_end_char_idxs[i+1]].rstrip().rstrip(".").rstrip(",").strip() for i in range(len(sentence_end_char_idxs)-1)]
        else:
            if "attrscore" in self.nli_model.ckpt:
                sents = ["{} {}".format(meta["question"], sent) for sent in sents]

        assert len(sentence_end_indices) == len(citation_end_indices) and len(sents) == len(citation_end_indices) and len(referneces) == len(sents)
        return sents, sentence_end_indices, citation_end_indices, referneces
    
    def _format_document(self, doc):
        """Format document for AutoAIS."""

        if "sent" in doc:
            # QA-extracted docs
            return "Title: %s\n%s" % (doc['title'], doc['sent'])
        else:
            return "Title: %s\n%s" % (doc['title'], doc['text'])
    
    def get_reward(self, 
                   prompts_input_ids: torch.tensor, 
                   prompts_attention_mask: torch.tensor, 
                   generated_input_ids: torch.tensor, # (B, output_len)
                   generated_attention_mask: torch.tensor, # (B, output_len)
                   generated_texts: List[str],
                #    generated_texts_for_reward: List[str],
                   metadata=None
                   ):
        
        # get the length of generated outputs
        policy_inputs_lens = torch.sum(generated_attention_mask, dim=1).tolist()

        citation_recall_rewards = []
        citation_precision_rewards = []
        n_sentences_entailed_by_citations = []
        n_citations_correct = []
        n_citations = []

        for batch_idx, (meta, gen_text) in enumerate(zip(metadata, generated_texts)):
            sents, sentence_end_indices, citation_end_indices, references = self.process_one_generation(gen_text, 
                                                    policy_inputs_lens[batch_idx], meta)

            target_sents = [remove_citations(sent).strip() for sent in sents]

            

            citation_recall_reward = [0]*policy_inputs_lens[batch_idx]
            citation_precision_reward = [0]*policy_inputs_lens[batch_idx]
            n_sentence_entailed_by_cit = 0
            n_citation_correct = 0
            n_citation = 0

            for sent_id, sent in enumerate(sents):
                target_sent = target_sents[sent_id] # Citation removed and (if opted for) decontextualized
                joint_entail = -1 # Undecided

                # Find references
                ref = references[sent_id]

                if len(ref) == 0:
                    # No citations
                    joint_entail = 0
                elif any([ref_id >= len(meta['docs']) for ref_id in ref]):
                    # Citations out of range
                    joint_entail = 0
                else:
                    if self.at_most_citations is not None:
                        ref = ref[:self.at_most_citations]
                    n_citation += len(ref)

                    joint_passage = '\n'.join([self._format_document(meta['docs'][psgs_id]) for psgs_id in ref])
                
                # If not directly rejected by citation format error, calculate the recall score
                if joint_entail == -1: 
                    joint_entail = self.nli_model.run_nli_autoais(joint_passage, target_sent)

                sent_end_token_idx = sentence_end_indices[sent_id]
                citation_recall_reward[sent_end_token_idx] = self.citation_recall_positive_reward if joint_entail else self.citation_recall_negative_reward
                n_sentence_entailed_by_cit = n_sentence_entailed_by_cit + joint_entail
                

                if joint_entail and len(ref) > 1:
                    for idx, psgs_id in enumerate(ref):
                        # condition A
                        passage = self._format_document(meta['docs'][psgs_id]) 
                        nli_result = self.nli_model.run_nli_autoais(passage, target_sent)

                        # condition B
                        if not nli_result:
                            subset_exclude = copy.deepcopy(ref)
                            subset_exclude.remove(psgs_id)
                            passage = '\n'.join([self._format_document(meta['docs'][pid]) for pid in subset_exclude])
                            nli_result = self.nli_model.run_nli_autoais(passage, target_sent)
                            if nli_result: # psgs_id is not necessary

                                end_token_idx = citation_end_indices[sent_id][idx]
                                citation_precision_reward[end_token_idx] = self.citation_precision_negative_reward
                                
                            else:
                                # psg_id implicitly helps other references to entail the claim
                                end_token_idx = citation_end_indices[sent_id][idx]
                                citation_precision_reward[end_token_idx] = self.citation_precision_positive_reward
                                n_citation_correct = n_citation_correct + 1
                        else:
                            # psg_id explicitly entails the claim
                            end_token_idx = citation_end_indices[sent_id][idx]
                            citation_precision_reward[end_token_idx] = self.citation_precision_positive_reward
                            n_citation_correct = n_citation_correct + 1
                elif not joint_entail:
                    for idx, psgs_id in enumerate(ref):
                        end_token_idx = citation_end_indices[sent_id][idx]
                        citation_precision_reward[end_token_idx] = self.citation_precision_negative_reward
                else:
                    # The only possible case here is that there is only 1 reference and it entails the claim
                    for idx, psgs_id in enumerate(ref):
                        end_token_idx = citation_end_indices[sent_id][idx]
                        citation_precision_reward[end_token_idx] = self.citation_precision_positive_reward
                    n_citation_correct = n_citation_correct + 1

            dataset = meta["dataset"]
            
            if dataset == "asqa" and self.asqa_multiplier is not None:
                citation_recall_reward = [r * self.asqa_multiplier for r in citation_recall_reward]
                citation_precision_reward = [r * self.asqa_multiplier for r in citation_precision_reward]

            citation_recall_rewards.append(citation_recall_reward)
            citation_precision_rewards.append(citation_precision_reward)
            n_sentences_entailed_by_citations.append(n_sentence_entailed_by_cit)
            n_citations_correct.append(n_citation_correct)
            n_citations.append(n_citation)

            
        return {
            "citation_recall_rewards": citation_recall_rewards,
            "citation_precision_rewards": citation_precision_rewards,
            "n_sentences_entailed_by_citations": n_sentences_entailed_by_citations,
            "n_citations_correct": n_citations_correct,
            "n_citations": n_citations
        }



class FineGrainedRewardCitation(BasicReward):
    
    def __init__(self,
                 tokenizer,
                #  non_factual_model_ckpt,
                #  factual_model_ckpt,
                #  completeness_model_ckpt,
                 kl_coef,
                 citation_recall_positive_reward = 1.0,
                 citation_recall_negative_reward = -1.0,
                 citation_precision_positive_reward = 1.0,
                 citation_precision_negative_reward = -1.0,
                 at_most_citations = 3,
                 correctness_positive_reward = 1.0, # Per new hit, mutiplied by number of new hit in a sentence
                 correctness_negative_reward = -1.0, # If no hit is found in a sentence
                 fluency_reward_mean = 0.0,
                 fluency_reward_std = 1.0,
                 fluency_reward_bias = 0.0,
                 fluency_reward_scale = 1.0,
                #  sep = "</s>",
                 nli_model=None,
                 holistic=False,
                 citation_recall=True,
                 citation_precision=True,
                 correctness_recall=True,
                 log_info=None,
                ):
        
        super().__init__(kl_coef)
        
        self.fluency_reward_bias = fluency_reward_bias
        self.fluency_reward_scale = fluency_reward_scale
        
        self.citation_reward = CitationRecallAndPrecisionReward(tokenizer,
            # non_factual_model_ckpt,
            citation_recall_positive_reward=citation_recall_positive_reward,
            citation_recall_negative_reward=citation_recall_negative_reward,
            citation_precision_positive_reward=citation_precision_positive_reward,
            citation_precision_negative_reward=citation_precision_negative_reward,
            at_most_citations=at_most_citations,
            # sep = sep,
            nli_model=nli_model,
            # nli_tokenizer=nli_tokenizer,    
        )
        

        self.correctness_reward = SequenceLevelCorrectnessReward(tokenizer,
            # factual_model_ckpt,
            correctness_positive_reward=correctness_positive_reward,
            correctness_negative_reward=correctness_negative_reward,
            # sep = sep,
            nli_model=nli_model,
            # nli_tokenizer=nli_tokenizer,  
        )

        
        self.fluency_reward = SequenceFluencyReward(tokenizer,
            # completeness_model_ckpt,
            mean=fluency_reward_mean,
            std=fluency_reward_std,
            bias=fluency_reward_bias,
            scale=fluency_reward_scale)
        
        self.nlp = spacy.load("en_core_web_sm")

        self.holistic = holistic
        self.mask = [int(citation_recall), int(citation_precision), int(correctness_recall)]
        self.ablation = sum(self.mask) < 3
        self.log_info = log_info
        self.log_info("Holistic: {} | Reward model mask: {} | Ablation: {}".format(self.holistic, self.mask, self.ablation))
        assert self.holistic != self.ablation or (not self.holistic and not self.ablation)
        assert sum(self.mask) > 0 and sum(self.mask) <= 3

    def get_finegrained_reward(self, prompts_input_ids, prompts_attention_mask, 
                            generated_input_ids, generated_attention_mask, 
                            generated_texts, normalized_generated_text, metadata):
        
        fine_grained_rewards = []
        n_sentences = []
        
        citation_rewards = self.citation_reward.get_reward(prompts_input_ids, prompts_attention_mask, 
                                                            generated_input_ids, generated_attention_mask, 
                                                            generated_texts, metadata=metadata)
            
        citation_recall_rewards = citation_rewards['citation_recall_rewards'] # Sentence level
        citation_precision_rewards = citation_rewards['citation_precision_rewards'] # Citation level
        n_sentences_entailed_by_citations = citation_rewards["n_sentences_entailed_by_citations"]
        n_citations_correct = citation_rewards['n_citations_correct']
        n_citations = citation_rewards["n_citations"]
        
        

        correctness = self.correctness_reward.get_reward(prompts_input_ids, prompts_attention_mask, 
                                                    generated_input_ids, generated_attention_mask, 
                                                    generated_texts, normalized_generated_text, metadata=metadata)

        n_sentences = correctness['n_sentences']
        correctness_rewards = correctness['correctness_rewards'] # Sentence level
        n_sentences_with_new_hits = correctness['n_sentences_with_new_hits']
        n_total_hits = correctness['n_total_hits']
        n_answers = correctness['n_answers']

        
        # combine the rewards
        for text_idx, generated_text in enumerate(generated_texts):

            if self.ablation:
                fine_grained_reward = [a*self.mask[0]+b*self.mask[1]+c*self.mask[2] for a,b,c in zip(citation_recall_rewards[text_idx], 
                                                         citation_precision_rewards[text_idx],
                                                         correctness_rewards[text_idx])]
            else:
                fine_grained_reward = [a+b+c for a,b,c in zip(citation_recall_rewards[text_idx], 
                                                            citation_precision_rewards[text_idx],
                                                            correctness_rewards[text_idx])]
            
            fine_grained_rewards.append(fine_grained_reward)
            
        return {
                "rewards": fine_grained_rewards, 
                # "n_sub_sentences": n_sub_sentences,
                "n_citations": n_citations,
                "n_sentences": n_sentences,
                "citation_recall_rewards": citation_recall_rewards,
                "citation_precision_rewards": citation_precision_rewards,
                "n_sentences_entailed_by_citations": n_sentences_entailed_by_citations,
                "n_citations_correct": n_citations_correct,
                "correctness_rewards": correctness_rewards,
                "n_sentences_with_new_hits": n_sentences_with_new_hits,
                "n_total_hits": n_total_hits,
                "n_answers": n_answers,
                # "fluency_rewards": fluency_rewards
            }
        


    def get_reward(self, 
                   prompts_input_ids: torch.tensor, 
                   prompts_attention_mask: torch.tensor, 
                   generated_input_ids: torch.tensor, # (B, output_len)
                   generated_attention_mask: torch.tensor, # (B, output_len)
                   generated_texts: List[str],
                   normalized_generated_text: List[str],
                   metadata=None, 
                   print_reward=False,
                   ):
        
        rewards_output = self.get_finegrained_reward(prompts_input_ids, prompts_attention_mask, 
                            generated_input_ids, generated_attention_mask, 
                            generated_texts, normalized_generated_text, metadata)
        
        if self.holistic:
            rewards_output['rewards'] = [[0.]* (len(fine_grained_reward) - 1) + [sum(fine_grained_reward)] for fine_grained_reward in rewards_output['rewards']]

        if print_reward:
            for i in range(len(rewards_output['rewards'])):
                self.log_info("Sampled generation {} citation_recall_reward: {}".format(i, rewards_output["citation_recall_rewards"][i]))
                self.log_info("Sampled generation {} citation_precision_reward: {}".format(i, rewards_output["citation_precision_rewards"][i]))
                self.log_info("Sampled generation {} correctness_reward: {}".format(i, rewards_output["correctness_rewards"][i]))

        return {'rewards/raw': rewards_output['rewards']}
            
        
    def eval_metrics(self, 
                prompts_input_ids: torch.tensor, 
                prompts_attention_mask: torch.tensor, 
                generated_input_ids: torch.tensor, # (B, output_len)
                generated_attention_mask: torch.tensor, # (B, output_len)
                generated_texts: List[str],
                normalized_generated_text: List[str],
                metadata=None, 
                ):
        
        output = {}
        
        finegrained_rewards_output = self.get_finegrained_reward(prompts_input_ids, prompts_attention_mask, 
                            generated_input_ids, generated_attention_mask, 
                            generated_texts, normalized_generated_text, metadata)
        
        n_citations = finegrained_rewards_output["n_citations"]
        n_sentences = finegrained_rewards_output['n_sentences']
        n_answers = finegrained_rewards_output['n_answers']
        
        citation_recalls = []
        citation_precisions = []
        correctness_precisions = []
        correctness_recalls = []
        correctness_recalls_top5 = []
        
        for text_idx, generated_text in enumerate(generated_texts):
            # citation recall reward
            n_sentence = n_sentences[text_idx]
            n_sentence_with_correct_citations = finegrained_rewards_output['n_sentences_entailed_by_citations'][text_idx]
            citation_recalls.append(n_sentence_with_correct_citations / n_sentence if n_sentence > 0 else 0)

            # citation precision reward
            n_citation = n_citations[text_idx]
            n_citation_correct = finegrained_rewards_output['n_citations_correct'][text_idx]
            citation_precisions.append(n_citation_correct/ n_citation if n_citation > 0 else 0)
            
            # correctness reward
            n_sentence_with_new_hit = finegrained_rewards_output['n_sentences_with_new_hits'][text_idx]
            n_total_hit = finegrained_rewards_output['n_total_hits'][text_idx]
            correctness_precisions.append(n_sentence_with_new_hit / n_sentence if n_sentence > 0 else 0)

            n_answer = n_answers[text_idx]
            correctness_recalls.append(n_total_hit / n_answer)
            correctness_recalls_top5.append(min(5, n_total_hit)/min(5, n_answer))

        
        # compute rouge scores
        rouge_scores = get_rouge_scores(generated_texts, [m['human_answer'] for m in metadata])
        
        # lens of generations
        generation_lens = torch.sum(generated_attention_mask, dim=-1).tolist()
        
        output.update({
            "eval/rouge": rouge_scores,
            "eval/rewards": [np.sum(sublist) for sublist in finegrained_rewards_output['rewards']],
            "eval/citation_recalls": citation_recalls,
            "eval/citation_precisions": citation_precisions,
            "eval/correctness_precisions": correctness_precisions,
            "eval/correctness_recalls": correctness_recalls,
            "eval/correctness_recalls_top5": correctness_recalls_top5,
            # "eval/fluency_rewards": fluency_rewards,
            # "eval/n_sub_sentences": n_sub_sentences,
            "eval/n_sentences": n_sentences,
            "eval/n_citations": n_citations,
            "eval/n_answers": n_answers,
            "eval/lengths": generation_lens
        })
        
        return output
    
    
    def aggregate_metrics(self, wandb_table, value_columns):
        # how to average over the metrics in wandb table for reporting
        stats = {}
        for k in value_columns:
            stats[k] = np.mean([row[wandb_table.columns.index(k)] for row in wandb_table.data])
        
        # citation_recalls, citation_precisions, and correctness_recalls are weighted by the number of sentences, citations, and answers respectively
        
        stats['eval/citation_recalls_eq_w'] = (np.sum([row[wandb_table.columns.index('eval/citation_recalls')] 
                                                  * row[wandb_table.columns.index('eval/n_sentences')] 
                                                  for row in wandb_table.data]) 
                                          / np.sum([row[wandb_table.columns.index('eval/n_sentences')] 
                                                    for row in wandb_table.data]))

        stats['eval/citation_precisions_eq_w'] = (np.sum([row[wandb_table.columns.index('eval/citation_precisions')] 
                                                  * row[wandb_table.columns.index('eval/n_citations')] 
                                                  for row in wandb_table.data]) 
                                          / np.sum([row[wandb_table.columns.index('eval/n_citations')] 
                                                    for row in wandb_table.data]))
        
        stats['eval/correctness_recalls_eq_w'] = (np.sum([row[wandb_table.columns.index('eval/correctness_recalls')]
                                                   * row[wandb_table.columns.index('eval/n_answers')]
                                                   for row in wandb_table.data])
                                           / np.sum([row[wandb_table.columns.index('eval/n_answers')]
                                                     for row in wandb_table.data]))

        stats['eval/correctness_precisions_eq_w'] = (np.sum([row[wandb_table.columns.index('eval/correctness_precisions')]
                                                    * row[wandb_table.columns.index('eval/n_sentences')]
                                                    for row in wandb_table.data])
                                            / np.sum([row[wandb_table.columns.index('eval/n_sentences')]
                                                        for row in wandb_table.data]))
        
        return stats
