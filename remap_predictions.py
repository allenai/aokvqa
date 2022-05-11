import argparse
import pathlib
import json

from aokvqa_utils import load_aokvqa, map_to_choices


parser = argparse.ArgumentParser()
parser.add_argument('--aokvqa-dir', type=pathlib.Path, required=True, dest='aokvqa_dir')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], required=True)
parser.add_argument('--pred', type=argparse.FileType('r'), required=True, dest='prediction_file')
parser.add_argument('--out', type=argparse.FileType('w'), required=True, dest='output_file')
args = parser.parse_args()


dataset = load_aokvqa(args.aokvqa_dir, args.split)
predictions = json.load(args.prediction_file)
predictions = map_to_choices(dataset, predictions)

json.dump(predictions, args.output_file)
