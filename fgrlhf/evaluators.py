from rouge_score import rouge_scorer
import nltk

scorer = rouge_scorer.RougeScorer(['rougeLsum'], use_stemmer=True)

def postprocess_text(preds, list_of_labels):

    # if 2 > 1:
    #     raise Exception("preds: {} | list_of_labels: {}".format(preds, list_of_labels))
    
    # rougeLSum expects newline after each sentence
    preds = ['\n'.join(nltk.sent_tokenize(pred.strip())) for pred in preds]
    list_of_labels = [['\n'.join(nltk.sent_tokenize(label.strip())) for label in labels] 
                      for labels in list_of_labels]
    return preds, list_of_labels


def get_rouge_scores(preds, list_of_labels):

    # Post-process text
    preds, list_of_labels = postprocess_text(preds, list_of_labels)

    # Score all predictions
    all_scores = []
    for pred, labels in zip(preds, list_of_labels):
        # We calculate scores for each label, and take the max
        label_scores = [scorer.score(pred, label) for label in labels]
        if len(labels) > 0:
            max_score = max(label_scores, key=lambda x: x['rougeLsum'].fmeasure)
        else:
            # QAMPARI samples do not have human answer lables for evaluation
            max_score = {'qampari_default': 0}
        all_scores.append(max_score)
    
    all_scores = [(round(v['rougeLsum'].fmeasure * 100, 4) if 'rougeLsum' in v else v['qampari_default']) for v in all_scores]
    return all_scores

def get_single_rouge_score(gen_text, labels):
    
    gen_text, labels = postprocess_text([gen_text], [labels])
    gen_text = gen_text[0]
    labels = labels[0]
    label_scores = [scorer.score(gen_text, label) for label in labels]
    max_score = max(label_scores, key=lambda x: x['rougeLsum'].fmeasure)
    return max_score['rougeLsum'].fmeasure
