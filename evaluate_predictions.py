import argparse
import pathlib
import json
import glob

from aokvqa_utils import load_aokvqa, eval_aokvqa, map_to_choices


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
