import argparse
import os
import json

import torch
from torchvision.ops import box_convert

def main(args):
    results = os.path.join(args.output_dir, "results.pth")
    results = torch.load(results)
    
    submit_results = []
    for image_id, output in results.items():
        for label, box, score in zip(output["labels"], output["boxes"], output["scores"]):
            xywh_box = box_convert(box, "xyxy", "xywh")

            submit_results.append({
                "image_id": image_id,
                "category_id": label,
                "bbox": xywh_box.squeeze().tolist(),
                "score": score,
                "segmentation": []
            })

    with open(args.output_txt_dir, "w") as f:
        f.write(json.dumps(submit_results))
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', "-o", type=str, required=True)
    parser.add_argument('--output-txt-dir', type=str, required=True)
    args = parser.parse_args()

    main(args)