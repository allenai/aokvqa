
## ClipCap

We have modified the [ClipCap](https://github.com/rmokady/CLIP_prefix_caption) codebase for our task of VQA. In particular, we have forked the original repo via [our ClipCap branch](https://github.com/allenai/aokvqa/tree/ClipCap) and [made additional changes](https://github.com/allenai/aokvqa/compare/1ad805a...ClipCap). This is already part of the codebase you cloned, assuming you included `--recurse-submodules` as directed in the [main branch README](https://github.com/allenai/aokvqa/blob/main/README.md).

<details> <summary><b>Downloading pretrained models</b></summary>

```bash
# We use this model: MLP mapping network and finetuned GPT-2 (pretrained on COCO)
gdown 1IdaBtMSvtyzF0ByVaBHtvM0JYSXRExRX -O ${PT_MODEL_DIR}/clipcap_coco_weights.pt
```

</details>

```bash
# Finetuning on our dataset
python ClipCap/train.py --log-dir ${LOG_DIR}/clipcap --aokvqa-dir ${AOKVQA_DIR} --train-features ${FEATURES_DIR}/clip-ViT-B-32_train.pt --val-features ${FEATURES_DIR}/clip-ViT-B-32_val.pt --pretrained-model ${PT_MODEL_DIR}/clipcap_coco_weights.pt --generation-target answer --mapping mlp --finetune-gpt

# Predicting (e.g. for epoch 3)
python ClipCap/predict.py --log-dir ${LOG_DIR}/clipcap --epoch 3 --aokvqa-dir ${AOKVQA_DIR} --split val --eval-features ${FEATURES_DIR}/clip-ViT-B-32_val.pt --out ${PREDS_DIR}/clipcap_val-da.json
```

For the multiple-choice setting, adjust the following arguments:
```bash
# ClipCap/train.py: --log-dir ${LOG_DIR}/clipcap-mc --prompt-with-choices
# ClipCap/predict.py: --log-dir ${LOG_DIR}/clipcap-mc --map-to-choices --out ${PREDS_DIR}/clipcap_val-mc.json
```

<details> <summary><b>For training with a Transformer mapping network</b></summary>

```bash
# Grab the Transformer ClipCap weights (pretrained on COCO)
gdown 1GYPToCqFREwi285wPLhuVExlz7DDUDfJ -O ${PT_MODEL_DIR}/clipcap_transformer_weights.pt

# ClipCap/train.py: --train-features ${FEATURES_DIR}/clip-RN50x4_train.pt --pretrained-model ${PT_MODEL_DIR}/clipcap_transformer_weights.pt --mapping transformer
# ClipCap/predict.py: --eval-features ${FEATURES_DIR}/clip-RN50x4_val.pt
```

</details>

## Generating Captions & Rationales

To generate rationales, we repeat the [above](#clipcap) ClipCap training and predictions, with some modifications. We only train one model (even between DA and MC settings).

```bash
mkdir -p ${LOG_DIR}/gpt3-inputs

# ClipCap/train.py: --log-dir ${LOG_DIR}/clipcap-rationale --generation-target rationale
# Be sure to exclude --prompt-with-choices

# ClipCap/predict.py: --log-dir ${LOG_DIR}/clipcap-rationale --beam-search --out ${LOG_DIR}/gpt3-inputs/clipcap-rationales_val.json
# Be sure to exclude --map-to-choices
```

<details> <summary><b>Prompting GPT-3 with rationales</b></summary>

First see [Querying GPT-3](https://github.com/allenai/aokvqa/blob/main/gpt3/README.md).

We should generate ground-truth rationale files:
```bash
for split in train val; do
    python gpt3/rationale_inputs.py --aokvqa-dir ${AOKVQA_DIR} --split ${split} --out logs/gpt3-inputs/rationales_${split}.json
done
```

You can prompt GPT-3 as described in the link, but with the following modified arguments:

```bash
# For prompting with ground-truth rationales:

# gpt3/query_gpt3.py: --train-context ${LOG_DIR}/gpt3-inputs/rationales_train.json --context ${LOG_DIR}/gpt3-inputs/rationales_val.json --out ${PREDS_DIR}/gpt3-rationales_val-da.json
# remap_predictions.py: --pred ${PREDS_DIR}/gpt3-rationales_val-da.json --out ${PREDS_DIR}/gpt3-rationales_val-mc.json

# For prompting with generated rationales:

# gpt3/query_gpt3.py: --train-context ${LOG_DIR}/gpt3-inputs/rationales_train.json --context ${LOG_DIR}/gpt3-inputs/clipcap-rationales_val.json --out ${PREDS_DIR}/gpt3-clipcap-rationales_val-da.json
# remap_predictions.py: --pred ${PREDS_DIR}/gpt3-clipcap-rationales_val-da.json --out ${PREDS_DIR}/gpt3-clipcap-rationales_val-mc.json
```

</details>

<details> <summary><b>Generating and prompting with captions</b></summary>

Please read everything else above first.

We can generate COCO captions with the original ClipCap weights.

```bash
python ClipCap/predict_clipcap.py --ckpt ${PT_MODEL_DIR}/clipcap_coco_weights.pt --mapping mlp --aokvqa-dir ${AOKVQA_DIR} --split val --eval-features ${FEATURES_DIR}/clip-ViT-B-32_val.pt --beam-search --out logs/gpt3-inputs/clipcap-captions_val.json
```

We should also generate ground-truth captions (for train and val).

```bash
for split in train val; do
    python gpt3/caption_inputs.py --aokvqa-dir ${AOKVQA_DIR} --coco-dir ${COCO_DIR} --split ${split} --out ${LOG_DIR}/gpt3-inputs/captions_${split}.json
done
```

Query GPT-3 with original arguments and the following modifications, and produce predictions.

```bash
# For prompting with ground-truth captions:

# gpt3/query_gpt3.py: --train-context ${LOG_DIR}/gpt3-inputs/captions_train.json --context ${LOG_DIR}/gpt3-inputs/captions_val.json --out ${PREDS_DIR}/gpt3-captions_val-da.json
# remap_predictions.py: --pred ${PREDS_DIR}/gpt3-captions_val-da.json --out ${PREDS_DIR}/gpt3-captions_val-mc.json

# For prompting with generated captions:

# gpt3/query_gpt3.py: --train-context ${LOG_DIR}/gpt3-inputs/captions_train.json --context ${LOG_DIR}/gpt3-inputs/clipcap-captions_val.json --out ${PREDS_DIR}/gpt3-clipcap-captions_val-da.json
# remap_predictions.py: --pred ${PREDS_DIR}/gpt3-clipcap-captions_val-da.json --out ${PREDS_DIR}/gpt3-clipcap-captions_val-mc.json
```

</details>
