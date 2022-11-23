import os
import pathlib
import argparse

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, \
    FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image

from load_aokvqa import load_aokvqa, get_coco_path


class AokvqaImageDataset(Dataset):
    def __init__(self, split, aokvqa_dir=None, coco_dir=None, transforms=None, version='v1p0'):
        self.split = split
        self.aokvqa_dir = aokvqa_dir if aokvqa_dir else os.getenv('AOKVQA_DIR')
        self.coco_dir = coco_dir if coco_dir else os.getenv('COCO_DIR')

        self.version = version
        self.transforms = transforms

        self.dataset = load_aokvqa(self.aokvqa_dir, self.split, self.version)
        self.image_ids = [sample['image_id'] for sample in self.dataset]
        self.image_paths = [
            get_coco_path(
                self.split, image_id, self.coco_dir
            ) for image_id in self.image_ids
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = read_image(self.image_paths[idx])

        if self.transforms:
            for transform in self.transforms:
                img = transform(img)

        return img, self.image_paths[idx]



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--aokvqa-dir', type=pathlib.Path, required=True, dest='aokvqa_dir')
    parser.add_argument('--coco-dir', type=pathlib.Path, required=True, dest='coco_dir')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], required=True)
    parser.add_argument('--model', type=str, choices=['faster_rcnn', 'fcos', 'retina_net', 'ssd', 'ssdlite'], required=True)
    parser.add_argument('--batch-size', type=int, default=1, dest='batch_size')
    parser.add_argument('--out-dir', type=pathlib.Path, required=True, dest='out_dir')

    args = parser.parse_args()

    if args.model == 'faster_rcnn':
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
    else:
        raise NotImplementedError(f"Model: {args.model} not implemented yet.")

    model.eval()

    transforms = [weights.transforms()]

    image_set = AokvqaImageDataset(args.split, transforms=transforms)
    image_dataloader = DataLoader(image_set, batch_size=args.batch_size, shuffle=False)

    for batch, (img, img_path) in enumerate(tqdm(image_dataloader)):
        prediction = model([img])[0]
        labels = [weights.meta["categories"][i] for i in prediction["labels"]]
        box = draw_bounding_boxes(img,
                                  boxes=prediction["boxes"],
                                  labels=labels,
                                  colors="red",
                                  width=4,
                                  font_size=30)
        img_bb = to_pil_image(box.detach())
        new_path = outdir.joinpath(img_path.split('/')[-1])
        img_bb.save(new_path)

        break