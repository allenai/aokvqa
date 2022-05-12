# A-OKVQA

Official repository for [A-OKVQA: A Benchmark for Visual Question Answering using World Knowledge]()

### Abstract

The Visual Question Answering (VQA) task aspires to provide a meaningful testbed for the development of AI models that can jointly reason over visual and natural language inputs. Despite a proliferation of VQA datasets, this goal is hindered by a set of common limitations. These include a reliance on relatively simplistic questions that are repetitive in both concepts and linguistic structure, little world knowledge needed outside of the paired image, and limited reasoning required to arrive at the correct answer. We introduce A-OKVQA, a crowdsourced dataset composed of a diverse set of about 25K questions requiring a broad base of commonsense and world knowledge to answer. In contrast to the existing knowledge-based VQA datasets, the questions generally cannot be answered by simply querying a knowledge base, and instead require some form of commonsense reasoning about the scene depicted in the image.  We demonstrate the potential of this new dataset through a detailed analysis of its contents and baseline performance measurements over a variety of state-of-the-art vision–language models.

![A-OKVQA Figure 1](./readme_files/teaser.png)

## Getting started

```bash
git clone --single-branch --recurse-submodules git@github.com:allenai/aokvqa.git

cd aokvqa
export PYTHONPATH=.

conda env create --name aokvqa
conda activate aokvqa
pip uninstall sentencepiece  # Conflicts with pytorch-lightning
```

### Downloading the dataset

```bash
export AOKVQA_DIR=./datasets/aokvqa/
mkdir -p ${AOKVQA_DIR}

curl https://prior-datasets.s3.us-east-2.amazonaws.com/aokvqa/aokvqa_v1p0.zip
unzip aokvqa_v1p0.zip -d ${AOKVQA_DIR}; rm aokvqa_v1p0.zip
```

You'll also want to download the following images and annotations for COCO 2017:

```bash
export COCO_DIR=./datasets/coco/
mkdir -p ${COCO_DIR}

for split in train val test; do
    wget "http://images.cocodataset.org/zips/${split}2017.zip"
    unzip "${split}2017.zip" -d ${COCO_DIR}; rm "${split}2017.zip"
done

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip -d ${COCO_DIR}; rm annotations_trainval2017.zip
```

Loading our dataset is easy! Just grab our [aokvqa_utils.py](https://github.com/allenai/aokvqa/blob/main/aokvqa_utils.py) file and refer to the following code.

```python
import os
import aokvqa_utils

aokvqa_dir = os.getenv('AOKVQA_DIR')

train_dataset = aokvqa_utils.load_aokvqa(aokvqa_dir, 'train')
```

```python
dataset_example = train_dataset[0]

print(dataset_example['question_id'])
# 22MexNkBPpdZGX6sxbxVBH

coco_dir = os.getenv('COCO_DIR')
image_path = aokvqa_utils.get_coco_path('train', dataset_example['image_id'], coco_dir)
print(image_path)
# ./datasets/coco/train2017/000000299207.jpg

print(dataset_example['question'])
print(dataset_example['choices'])
# What is the man by the bags awaiting?
# ['skateboarder', 'train', 'delivery', 'cab']

correct_choice = dataset_example['choices'][ dataset_example['correct_choice_idx'] ]
# Corrrect: cab

print(dataset_example['rationales'][0])
# A train would not be on the street, he would not have luggage waiting for a delivery, and the skateboarder is there and not paying attention to him so a cab is the only possible answer.
```

## Evaluation

Please prepare a `predictions.json` file for each evaluation split with the format: `{ question_id (str) : prediction (str) }`. Be sure that this file is complete (i.e. includes a prediction for every question in the evaluation set).

```python
import os
import json
import aokvqa_utils

aokvqa_dir = os.getenv('AOKVQA_DIR')
split = 'val'
multiple_choice = False
predictions_file = './path/to/predictions.json'

eval_dataset = aokvqa_utils.load_aokvqa(aokvqa_dir, split)
predictions = json.load(open(predictions_file, 'r'))

acc = aokvqa_utils.eval_aokvqa(eval_dataset, predictions, multiple_choice=multiple_choice)
print(acc)
```

To compute metrics over a batch of predictions files (e.g. `./predictions/{model-name}_val-da.json`), you can instead run `python evaluate_predictions.py --aokvqa-dir ${AOKVQA_DIR} --split val --preds "./predictions/*_val-da.json"`. Add the `--multiple-choice` flag to run MC evaluation over (e.g. `*_val-mc.json`) files that have instead been generated for the multiple-choice setting.

### Leaderboard

You can submit predictions from your model to our leaderboard! Simply produce predictions files for each of the settings (Val DA, Val MC, Test DA, Test MC) and submit here. Remember that you aren't allowed to give choices to your model when predicting direct answers.

## Codebase

We provide all code and pretrained models necessary to replicate our experiments for Large-Scale Pretrained Models (sec. 5.2) and Rationale Generation (sec. 5.3).

### Preparing data

```bash
export FEATURES_DIR=./features/
mkdir -p ${FEATURES_DIR}
```

You can compute CLIP features for our vocabulary and dataset. These are most commonly used by our other experiments.

```bash
python data_scripts/encode_vocab_clip.py --vocab ${AOKVQA_DIR}/large_vocab_train.csv --model-type ViT-B/32 --out ${FEATURES_DIR}/clip-ViT-B-32_large_vocab.pt

for split in train val test; do
    python data_scripts/extract_clip_features.py --aokvqa-dir ${AOKVQA_DIR} --coco-dir ${COCO_DIR} --split ${split} --model-type ViT-B/32 --out ${FEATURES_DIR}/clip-ViT-B-32_${split}.pt
done
```

If you want to train our ClipCap models with the transformer mapping network (instead of an MLP, like we do), you'll also need to run `extract_clip_features.py` with `--model-type RN50x4`.

Our ResNet and BERT classification experiments require these respective features instead of CLIP. To generate these, please run the following commands for each `${split}`:
```bash
python data_scripts/extract_resnet_features.py --aokvqa-dir ${AOKVQA_DIR} --coco-dir ${COCO_DIR} --split ${split} --out ${FEATURES_DIR}/resnet_${split}.pt

python data_scripts/extract_bert_features.py --aokvqa-dir ${AOKVQA_DIR} --split ${split} --out ${FEATURES_DIR}/bert_${split}.pt
```

### Models and Predictions

```bash
export LOG_DIR=./logs/
export PREDS_DIR=./predictions/
mkdir -p ${LOG_DIR} ${PREDS_DIR}
```

#### Heuristics

Simply run [random_unweighted.py](https://github.com/allenai/aokvqa/blob/main/heuristics/random_unweighted.py), [random_weighted.py](https://github.com/allenai/aokvqa/blob/main/heuristics/random_weighted.py), or [most_common_answer.py](https://github.com/allenai/aokvqa/blob/main/heuristics/most_common_answer.py) with the following example command.

```bash
python heuristics/random_unweighted.py --aokvqa-dir ${AOKVQA_DIR} --split val --out predictions/random-unweighted_val-da.json
```

You can add the `--mc` flag to generate predictions for the multiple choice setting (also for `--out ..._val-mc.json`).

#### Transfer Learning Experiments

We use the following training/prediction scripts for the classifier, zero-shot, and contrastive experiments in Table 3.

```bash
## Training: build your own command
python transfer_experiments/train.py --aokvqa-dir ${AOKVQA_DIR} --vocab ${AOKVQA_DIR}/large_vocab_train.csv --log-dir ${LOG_DIR}

# Backbone
--backbone clip --clip-model-type ViT-B/32 --train-features ${FEATURES_DIR}/clip-ViT-B-32_train.pt --val-features ${FEATURES_DIR}/clip-ViT-B-32_val.pt
--inputs question # OR --inputs image  # OR --inputs question image
# OR
--backbone resnet --train-features ${FEATURES_DIR}/resnet_train.pt --val-features ${FEATURES_DIR}/resnet_val.pt --inputs image
# OR
--backbone bert --train-features ${FEATURES_DIR}/bert_train.pt --val-features ${FEATURES_DIR}/bert_val.pt --inputs question

# Objective
--objective classifier
# OR
--objective contrastive --vocab-features ${FEATURE_DIR}/clip-ViT-B-32_large_vocab.pt
```

```bash
## Predictions
python transfer_experiments/predict.py --aokvqa-dir ${AOKVQA_DIR}

# Split
--split val # or test
# Adjust according to backbone and split:
--features ${FEATURE_DIR}/clip-ViT-B-32_val.pt

# Model
--ckpt path/to/model.ckpt
# OR
--zero-shot --clip-model-type ViT-B/32
--inputs question  # OR --inputs image  # OR --inputs question image

# For multiple choice mode add: --mc

parser.add_argument('--vocab', type=argparse.FileType('r'))
parser.add_argument('--vocab-features', type=pathlib.Path, dest='vocab_features')
#
```

#### ClipCap

We have modified the [ClipCap](https://github.com/rmokady/CLIP_prefix_caption) codebase for our task of VQA. In particular, we have forked the original repo via [our ClipCap branch](https://github.com/allenai/aokvqa/tree/ClipCap) and [made additional changes](https://github.com/allenai/aokvqa/compare/1ad805a...ClipCap).

Be sure that you cloned this repo with the command above: i.e. with `--recurse-submodules`.
Also, you'll need to grab the pretrained model (ClipCap with MLP mapping network and finetuned GPT-2, trained on COCO):
```bash
export PT_MODEL_DIR=./pretrained_models/
mkdir -p ${PT_MODEL_DIR}

gdown 1IdaBtMSvtyzF0ByVaBHtvM0JYSXRExRX -O ${PT_MODEL_DIR}/clipcap_coco_weights.pt
```

Then, you can finetune and predict on our dataset with:

```bash
python ClipCap/train.py --log-dir ${LOG_DIR}/clipcap --aokvqa-dir ${AOKVQA_DIR} --train-features ${FEATURES_DIR}/clip-ViT-B-32_train.pt --val-features ${FEATURES_DIR}/clip-ViT-B-32_val.pt --pretrained-model ${PT_MODEL_DIR}/clipcap_coco_weights.pt --generation-target answer --mapping mlp --finetune-gpt
```

We add `--prompt-with-choices` for training a model in the multiple-choice setting.

```bash
python ClipCap predict.py --log-dir ${LOG_DIR}/clipcap --epoch 9 --aokvqa-dir ${AOKVQA_DIR} --split val --eval-features ${FEATURES_DIR}/clip-ViT-B-32_val.pt --out ${PREDS_DIR}/clipcap_val-da.json
```

Please add the `--map-to-choices` flag and `--out ...-mc.json` for producing predictions in the multiple-choice setting.

If you would like to instead train the transformer mapping network, grab that pretrained model with `gdown 1GYPToCqFREwi285wPLhuVExlz7DDUDfJ -O clipcap_transformer_weights.pt` (replacing `--pretrained-model` appropriately) and set `--mapping transformer`.

#### Generating Rationales

To generate rationales, we repeat the above ClipCap training and predictions, with some modifications.

For training, we instead pass `--generation-target rationale` and always exclude `--prompt-with-choices`. We only train one model between the DA and MC settings.

For prediction, we add the `--beam-search` flag and output to `--out ${LOG_DIR}/gpt3-inputs/clipcap-rationales_val.json`.

#### Querying GPT-3

To follow our experiments which use GPT-3, you. Please retrieve your [organization](https://beta.openai.com/account/org-settings) and [API](https://beta.openai.com/account/api-keys) keys and set them in your environment variables.

```bash
export OPENAI_ORG=....
export OPENAI_API_KEY=...
```

Create ground-truth rationale files.

Then you can simply run:
```bash
python gpt3/query_gpt3.py --aokvqa-dir ${AOKVQA_DIR} --split val --out ${PREDS_DIR}/gpt3_val-da.json
python remap_predictions.py --aokvqa-dir ${AOKVQA_DIR} --split val --pred ${PREDS_DIR}/gpt3_val-da.json --out ${PREDS_DIR}/gpt3_val-mc.json
```

You can also prompt GPT-3 with "Context: ..." by providing files with `{ question_id (str) : context (str) }` for the training and evaluation splits. For example, we can prompt with ground-truth train and ClipCap generated val rationales with `--train-context ${LOG_DIR}/gpt3-inputs/rationales_train.json --context ${LOG_DIR}/gpt3-inputs/clipcap-rationales_val.json`.
