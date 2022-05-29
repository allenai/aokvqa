import argparse
import pathlib
import json
import glob

from load_aokvqa import load_aokvqa


def eval_aokvqa(dataset, preds, multiple_choice=False, strict=True):

    if isinstance(dataset, list):
        dataset = { dataset[i]['question_id'] : dataset[i] for i in range(len(dataset)) }

    if multiple_choice:
        dataset = {k:v for k,v in dataset.items() if v['difficult_direct_answer'] is False}

    if strict:
        dataset_qids = set(dataset.keys())
        preds_qids = set(preds.keys())
        assert dataset_qids.issubset(preds_qids)

    # dataset = q_id (str) : dataset element (dict)
    # preds = q_id (str) : prediction (str)

    acc = []

    for q in dataset.keys():
        if q not in preds.keys():
            acc.append(0.0)
            continue

        pred = preds[q]
        choices = dataset[q]['choices']
        direct_answers = dataset[q]['direct_answers']

        ## Multiple Choice setting
        if multiple_choice:
            if strict:
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--aokvqa-dir', type=pathlib.Path, required=True, dest='aokvqa_dir')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test_w_ans'], required=True)
    parser.add_argument('--preds', type=str, required=True, dest='prediction_files')
    parser.add_argument('--multiple-choice', action='store_true', dest='multiple_choice')
    args = parser.parse_args()

    dataset = load_aokvqa(args.aokvqa_dir, args.split)

    for prediction_file in glob.glob(args.prediction_files):
        predictions = json.load(open(prediction_file, 'r'))

        acc = eval_aokvqa(
            dataset,
            predictions,
            multiple_choice=args.multiple_choice,
            ensure_valid_choice=False
        )

        print(prediction_file, acc)
