import os
from typing import List
import PIL.Image
import supervision as sv
from ultralytics import YOLO
import argparse
import PIL
import pandas as pd


class YoloEvaluator:
    def __init__(self, output_dir: str):
        self.model = YOLO('yolov9e.pt')
        self.bounding_box_annotator = sv.BoundingBoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.output_dir = output_dir

    def evaluate_example(self, image_path: str, class_name: str, expected_count: int):
        pil_image = PIL.Image.open(image_path)

        # Run model
        result = self.model(pil_image)[0]
        detections = sv.Detections.from_ultralytics(result)
        
        # Analyze results
        detections_df = pd.DataFrame([[x for x in detections.xyxy], [*detections.data['class_name']]]).T
        detections_df = detections_df.rename({0: 'box', 1: 'class_name'}, axis=1)
        detections_df['correct_object'] = detections_df['class_name'].apply(lambda detected_class_name: detected_class_name == class_name)
        output = detections_df['correct_object'].sum()        
        
        # Create annotated image with bounding boxes and labels
        annotated_frame = self._create_annotated_image(pil_image, detections)
        
        return {
            "output": output,            
            "is_success": output == expected_count,
            "annotated_frame": annotated_frame,
            "image_path": image_path,
            "class_name": class_name,
            "expected_count": expected_count
        }

    def _create_annotated_image(self, pil_image: PIL.Image, detections):
        annotated_frame = pil_image.copy()
        annotated_frame = self.bounding_box_annotator.annotate(
            scene=annotated_frame,
            detections=detections
        )
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame,
            detections=detections
        )
        
        return annotated_frame


def extract_image_paths(images_dir: str) -> List[str]:
    return [f'{images_dir}/{img_name}' for img_name in os.listdir(images_dir) if img_name.endswith('.png')]


def analyze_image_name(image_path: str) -> int:
    parts = os.path.basename(image_path).split('__')
    assert len(parts) >= 2, f"Expected image name to have at least 2 parts separated by '__', but received {len(parts)} parts"
    try:
        expected_count = int(parts[0])
    except:
        raise ValueError(f"Expected first part of the image name to be an integer, but received {parts[0]}")
    class_name = parts[1]
    
    return expected_count, class_name


def save_results(results: List[dict], output_dir: str):
    pd.DataFrame(results).to_csv(f"{output_dir}/results.csv", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, default='examples')
    parser.add_argument('--output_dir', type=str, default='output')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    images_paths = extract_image_paths(args.images_dir)
    evaluator = YoloEvaluator(args.output_dir)
    results = []
    for image_path in images_paths:
        # Extract image details
        expected_count, class_name = analyze_image_name(image_path)
        
        # Evaluate
        result = evaluator.evaluate_example(image_path, class_name, expected_count)

        # Save annotated frame
        result.pop('annotated_frame').save(f"{args.output_dir}/{os.path.basename(image_path)}")

        # Save result
        results.append(result)
            
    save_results(results, args.output_dir)
