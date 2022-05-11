import os
import json
from tqdm import tqdm


def load_aokvqa(aokvqa_dir, split, version='v1p0'):
    assert split in ['train', 'val', 'test', 'test_w_ans']
    dataset = json.load(open(
        os.path.join(aokvqa_dir, f"aokvqa_{version}_{split}.json")
    ))
    return dataset


def get_coco_path(split, image_id, coco_dir):
    return os.path.join(coco_dir, f"{split}2017", f"{image_id:012}.jpg")


def eval_aokvqa(dataset, preds, multiple_choice=False, ensure_valid_choice=True):

    if isinstance(dataset, list):
        dataset = { dataset[i]['question_id'] : dataset[i] for i in range(len(dataset)) }

    if multiple_choice:
        dataset = {k:v for k,v in dataset.items() if v['difficult_direct_answer'] is False}

    dataset_qids = set(dataset.keys())
    preds_qids = set(preds.keys())
    assert dataset_qids.issubset(preds_qids)

    # dataset = q_id (str) : dataset element (dict)
    # preds = q_id (str) : prediction (str)

    acc = []

    for q in dataset.keys():
        assert q in preds
        pred = preds[q]
        choices = dataset[q]['choices']
        direct_answers = dataset[q]['direct_answers']

        ## Multiple Choice setting
        if multiple_choice:
            if ensure_valid_choice:
                assert pred in choices, 'Prediction must be a valid choice'
            correct_choice_idx = dataset[q]['correct_choice_idx']
            acc.append( float(pred == choices[correct_choice_idx]) )
        ## Direct Answer setting
        else:
            num_match = sum([pred == da for da in direct_answers])
            vqa_acc = min(1.0, num_match / 3.0)
            acc.append(vqa_acc)

    acc = sum(acc) / len(acc) * 100

    return acc


def map_to_choices(dataset, predictions, device='cpu'):
    if isinstance(dataset, list):
        dataset = { dataset[i]['question_id'] : dataset[i] for i in range(len(dataset)) }

    if all([p in dataset[q]['choices'] for q, p in predictions.items()]):
        return predictions

    try:
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.util import cos_sim
    except ModuleNotFoundError:
        print( 'Error: Please `pip install sentence-transformers` before calling the `map_to_choices` function.'
               'Returning unmodified predictions instead.' )
        return predictions

    model = SentenceTransformer('sentence-transformers/average_word_embeddings_glove.6B.300d')
    model.to(device)
    for q in tqdm(predictions.keys()):
        choices = dataset[q]['choices']
        if predictions[q] not in choices:
            choice_embeddings = model.encode([predictions[q]] + choices, convert_to_tensor=True)
            a_idx = cos_sim(choice_embeddings[0], choice_embeddings[1:]).argmax().item()
            predictions[q] = choices[a_idx]

    return predictions
