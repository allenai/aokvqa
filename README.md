# A-OKVQA

Official repository for [A-OKVQA: A Benchmark for Visual Question Answering using World Knowledge]()

### Abstract

The Visual Question Answering (VQA) task aspires to provide a meaningful testbed for the development of AI models that can jointly reason over visual and natural language inputs. Despite a proliferation of VQA datasets, this goal is hindered by a set of common limitations. These include a reliance on relatively simplistic questions that are repetitive in both concepts and linguistic structure, little world knowledge needed outside of the paired image, and limited reasoning required to arrive at the correct answer. We introduce A-OKVQA, a crowdsourced dataset composed of a diverse set of about 25K questions requiring a broad base of commonsense and world knowledge to answer. In contrast to the existing knowledge-based VQA datasets, the questions generally cannot be answered by simply querying a knowledge base, and instead require some form of commonsense reasoning about the scene depicted in the image.  We demonstrate the potential of this new dataset through a detailed analysis of its contents and baseline performance measurements over a variety of state-of-the-art vision–language models.

![A-OKVQA Figure 1](./readme_files/teaser.png)

## Downloading the dataset

<!-- wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip

unzip val2017.zip
rm val2017.zip -->

Loading the dataset is easy!

```python
import aokvqa_utils

aokvqa_dir = './datasets/aokvqa'
coco_dir = './datasets/coco'

train_dataset = aokvqa_utils.load_aokvqa(aokvqa_dir, 'train')
dataset_example = train_dataset[0]

print(dataset_example['question_id'])
print(dataset_example['question'])
# 22MexNkBPpdZGX6sxbxVBH
# What is the man by the bags awaiting?

image_path = aokvqa_utils.get_coco_path('train', dataset_example['image_id'], coco_dir)
print(image_path)
# ./datasets/coco/train2017/000000299207.jpg

choices = dataset_example['choices']
correct_choice = choices[dataset_example['correct_choice_idx']]
print(choices)
print(f"Correct: {correct_choice}")
# ['skateboarder', 'train', 'delivery', 'cab']
# Corrrect: cab

print(dataset_example['rationales'][0])
# A train would not be on the street, he would not have luggage waiting for a delivery, and the skateboarder is there and not paying attention to him so a cab is the only possible answer.
```

### Evaluation

```python
import json
import aokvqa_utils

aokvqa_dir = './datasets/aokvqa'
split = 'val'
multiple_choice = False
predictions_file = './path/to/predictions.json'

eval_dataset = aokvqa_utils.load_aokvqa(aokvqa_dir, split)
predictions = json.load(open(predictions_file, 'r'))

acc = aokvqa_utils.eval_aokvqa(eval_dataset, predictions, multiple_choice=multiple_choice)
```

## Codebase

We provide all code and pretrained models necessary to replicate our experiments for Large-Scale Pretrained Models (sec. 5.2) and Rationale Generation (sec. 5.3).

```bash
git clone --single-branch --recurse-submodules git@github.com:allenai/aokvqa.git

cd aokvqa
export PYTHONPATH=.

conda env create --name aokvqa
conda activate aokvqa
pip uninstall sentencepiece  # Conflicts with pytorch-lightning
```

### Preparing data

### Heuristics & Transfer Learning Experiments

### ClipCap & Generating Rationales

### Querying GPT-3
