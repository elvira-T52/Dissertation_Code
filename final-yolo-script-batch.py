import argparse
import numpy as np
import os
import csv
import json
import logging
import time
from pathlib import Path
import gc
import torch
from typing import Dict, List, Tuple, Optional
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Added logging configuration here
#logging.basicConfig(
    #filename=os.path.join(os.getcwd(), 'yolo_detection_chair.log'),
    #level=logging.INFO,
    #format='%(asctime)s - %(levelname)s - %(message)s',
    #force=True  # This ensures the config isn't overwritten
#)

def get_yolo_classes(model: YOLO) -> Dict[int, str]:
    """Get class names and indices from YOLO model."""
    return model.names

def build_class_help_text(model: YOLO) -> str:
    """Build help text including all YOLO class codes."""
    classes = model.names
    help_text = "Target class code for detection. Available codes:\n"
    for idx in range(0, len(classes), 3):
        row_items = []
        for col in range(3):
            if idx + col < len(classes):
                row_items.append(f"{idx + col}: {classes[idx + col]}")
        help_text += "  " + "  |  ".join(row_items) + "\n"
    return help_text

class COCOEvaluator:
    def __init__(self, annotation_file: str):
        """Initialize COCO evaluator with ground truth annotations."""
        ##try:
        if not os.path.exists(annotation_file):
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
            
        with open(annotation_file, 'r') as f:
            try:
                self.annotations = json.load(f)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON format in annotation file")
            
        self.coco_gt = COCO(annotation_file)

        self.model_to_coco_map = {
                75: 86,  # Map model class "vase" (75) to COCO class "vase" (86)
                0: 1,    # Model class "person" to COCO class "person"
                2: 3,    # Model class "car" to COCO class "car"
                56: 62   # Model class "chair" to COCO class "chair"
            }


            #logging.info("COCO annotations loaded successfully")
        #except Exception as e:
            #logging.error(f"Error initializing COCO evaluator: {str(e)}")
            #raise

    def calculate_map(self, image_id: str, predictions: List[Dict]) -> Optional[float]:
        """Calculate mAP for a single image using COCO evaluation."""
        try:
            # Create COCO prediction format
            coco_predictions = []
            for pred in predictions:
                mapped_category_id = self.model_to_coco_map.get(pred['category_id'], pred['category_id'])

                #logging.info(f"Mapping model class ID {pred['category_id']} to COCO class ID {mapped_category_id}")

                coco_pred = {
                    'image_id': int(image_id),
                    'category_id': mapped_category_id,
                    'bbox': pred['bbox'],  # [x, y, width, height]
                    'score': float(pred['score'])
                }
                coco_predictions.append(coco_pred)

            if not coco_predictions:
                return None

            # Create COCO prediction object
            coco_dt = self.coco_gt.loadRes(coco_predictions)
            
            # Initialize COCOeval object
            coco_eval = COCOeval(self.coco_gt, coco_dt, 'bbox')
            coco_eval.params.imgIds = [image_id]
            
            # Run evaluation
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            # Return AP@0.50:0.95
            return float(coco_eval.stats[0])
        except Exception as e:
            #logging.error(f"Error calculating mAP for image {image_id}: {str(e)}")
            return None

class YOLODetector:
    def __init__(self, yolo_dir: str, target_class: str, conf_threshold: float = 0.01):
        """Initialize YOLOv8 detector with model and target class."""
        model_path = os.path.join(yolo_dir, 'yolov8n.pt')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model not found at: {model_path}")

        self.model = YOLO(model_path) #.to('cuda:0')
        self.conf_threshold = conf_threshold
        ##logging.info(f"Initial batch config: {self.model.args}")
        self.model.overrides['batch'] = 192  # Force batch size
        self.model.overrides['workers'] = 8
        self.model.overrides['device'] = 0 
        ##logging.info(f"Updated batch config: {self.model.args}")
        
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.to('cuda:0').half()
            self.model.args.update({'batch': 192})
            ##logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        #if torch.cuda.is_available():
            #logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            #self.model.to('cuda:0').half()  # Move model to GPU
        #else:
            #logging.warning("No GPU detected, running on CPU!")
            #self.model.to('cpu')  # Fallback to CPU
        #logging.info(f"Model device: {next(self.model.parameters()).device}")
        #logging.info(f"Confidence threshold set to {self.conf_threshold}")

        #logging.info(f"Model loaded from: {model_path}")
        #logging.info(f"Available classes: {self.model.names}")
        #logging.info(f"Model task type: {self.model.task}")
        #logging.info(f"Model architecture: {self.model.model}")

        self.target_class_id = int(target_class)
        
        if self.target_class_id not in self.model.names:
            raise ValueError(f"Invalid class ID: {self.target_class_id}")
            
        self.target_class = self.model.names[self.target_class_id]
        #logging.info(f"Successfully initialized detector for class {self.target_class_id} ({self.target_class})")

    def detect_objects(self, image_path: str) -> Tuple[Optional[float], Optional[List[Dict]]]:
        try:
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                #logging.info(f"Current GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                results = self.model.predict(source=image_path, conf=self.conf_threshold, device=0, batch=192, workers=8, verbose=False)
                ##results = self.model(image_path, conf=self.conf_threshold, batch=192, verbose=False)
                end.record()
                torch.cuda.synchronize()
                #logging.info(f"Batch inference time: {start.elapsed_time(end)}ms")
                
                predictions = []
                total_confidence_area = 0
                total_area = 0
                target_confidences = []

                ##logging.info(f"Processing image: {image_path}")
                ##logging.info(f"Looking for class {self.target_class_id} ({self.target_class})")

                for result in results:
                    boxes = result.boxes
                    #logging.info(f"Found {len(boxes)} total detections")
                    for box in boxes:
                        class_id = int(box.cls[0])
                        #logging.info(f"Found object of class {class_id}")

                        if class_id == self.target_class_id:
                            confidence = float(box.conf[0])
                            target_confidences.append(confidence)
                            #logging.info(f"Found target class {class_id} with confidence {confidence}")
                            bbox = box.xywh[0].tolist()
                            bbox_area = bbox[2] * bbox[3]
                            total_confidence_area += confidence * bbox_area
                            total_area += bbox_area
                                    
                            bbox[0] = bbox[0] - bbox[2] / 2
                            bbox[1] = bbox[1] - bbox[3] / 2

                            #logging.info(f"Converted bbox: {bbox}")

                            predictions.append({
                                'category_id': self.target_class_id,
                                'bbox': bbox,
                                'score': confidence
                            })

                #if target_confidences:
                    #logging.info(f"Target class confidence scores: {target_confidences}")
                    #logging.info(f"Number of target class detections: {len(target_confidences)}")
                    #logging.info(f"Average target class confidence: {sum(target_confidences)/len(target_confidences)}")
                #else:
                    #logging.info("No target class detections found")
            
                weighted_confidence = total_confidence_area / total_area if total_area > 0 else 0.0
                torch.cuda.synchronize()
                return weighted_confidence, predictions
                
        except Exception as e:
            #logging.error(f"Error processing image {image_path}: {str(e)}")
            return None, None
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    def detect_objects_batch(self, image_paths: List[str]) -> List[Tuple[Optional[float], Optional[List[Dict]]]]:
            """
            Perform batched inference on multiple images.
            Returns a list of (weighted_confidence, predictions) tuples, one per image.
            """
            results_list = []

            # Use autocast for half precision
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                #logging.info(f"Current GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
                
                # Start timing
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

                # PASS THE ENTIRE LIST OF IMAGES AT ONCE
                results = self.model.predict(
                    source=image_paths,
                    conf=self.conf_threshold,
                    device=0,
                    batch=192,
                    workers=8,
                    verbose=False
                )

                end.record()
                torch.cuda.synchronize()
                #logging.info(f"Batched inference time: {start.elapsed_time(end)}ms for {len(image_paths)} images")

            # Now 'results' is a list of 'Results' objects, one per image in the same order
            for image_path, result in zip(image_paths, results):
                predictions = []
                total_confidence_area = 0
                total_area = 0
                target_confidences = []

                #logging.info(f"Processing image: {image_path}")
                ##logging.info(f"Looking for class {self.target_class_id} ({self.target_class})")
                boxes = result.boxes
               ## logging.info(f"Found {len(boxes)} total detections")

                for box in boxes:
                    class_id = int(box.cls[0])
                    #logging.info(f"Found object of class {class_id}")
                    
                    if class_id == self.target_class_id:
                        confidence = float(box.conf[0])
                        target_confidences.append(confidence)
                        ##logging.info(f"Found target class {class_id} with confidence {confidence}")
                        bbox = box.xywh[0].tolist()
                        bbox_area = bbox[2] * bbox[3]
                        total_confidence_area += confidence * bbox_area
                        total_area += bbox_area

                        # Convert from xywh center to top-left
                        bbox[0] = bbox[0] - bbox[2] / 2
                        bbox[1] = bbox[1] - bbox[3] / 2

                        ##logging.info(f"Converted bbox: {bbox}")

                        predictions.append({
                            'category_id': self.target_class_id,
                            'bbox': bbox,
                            'score': confidence
                        })

                #if target_confidences:
                    ##logging.info(f"Target class confidence scores: {target_confidences}")
                    ##logging.info(f"Number of target class detections: {len(target_confidences)}")
                    ##logging.info(f"Average target class confidence: {sum(target_confidences)/len(target_confidences)}")
                #else:
                    ##logging.info("No target class detections found")

                weighted_confidence = total_confidence_area / total_area if total_area > 0 else 0.0
                results_list.append((weighted_confidence, predictions))

            torch.cuda.synchronize()
            return results_list

class TransformationAnalyzer:
    def __init__(self, class_dir: str, yolo_detector: YOLODetector, coco_evaluator: Optional[COCOEvaluator] = None):
        """Initialize transformation analyzer."""
        self.class_dir = Path(class_dir)
        if not self.class_dir.is_dir():
            raise NotADirectoryError(f"Class directory not found: {class_dir}")
        
        self.yolo_detector = yolo_detector
        self.coco_evaluator = coco_evaluator
        self.enable_map = coco_evaluator is not None
        self.error_summary = {
            'unprocessed_folders': [],
            'missing_files': [],
            'unreadable_images': [],
            'map_calculation_errors': [] if self.enable_map else None
        }

    def process_transformation_folder(self, transform_folder: Path) -> None:
        """Process a single transformation folder."""
        try:
            csv_files = list(transform_folder.glob('*.csv'))
            if not csv_files:
                raise FileNotFoundError(f"No transforms CSV found in {transform_folder}")
            
            csv_file = csv_files[0]
            
            with open(csv_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                fieldnames = reader.fieldnames

            if 'confidence_score' not in fieldnames:
                fieldnames.append('confidence_score')
            if self.enable_map and 'mAP' not in fieldnames:
                fieldnames.append('mAP')

            BATCH_SIZE = 192
            for i in range(0, len(rows), BATCH_SIZE):
                batch = rows[i:i + BATCH_SIZE]

                # 1) Collect the image paths in a list
                image_paths = [row['output_path'] for row in batch]

                # 2) Call the batched inference
                batched_results = self.yolo_detector.detect_objects_batch(image_paths)
                # Now `batched_results` is a list of (confidence, predictions) for each image in `image_paths`

                # 3) Match each result to the corresponding row
                for row, (confidence, predictions) in zip(batch, batched_results):
                    if not os.path.exists(row['output_path']):
                        self.error_summary['missing_files'].append(row['output_path'])
                        row['confidence_score'] = 'File Missing'
                        if self.enable_map:
                            row['mAP'] = 'Error'
                        continue

                    row['confidence_score'] = str(confidence) if confidence is not None else 0.0

                    # If mAP is enabled, compute it if predictions exist
                    if self.enable_map and predictions:
                        image_name = row['image_name']
                        image_id = int(image_name.split('.')[0])

                        ##logging.info(f"Extracted image ID: {image_id} from image_name: {image_name}")

                        map_score = self.coco_evaluator.calculate_map(image_id, predictions)
                        row['mAP'] = str(map_score) if map_score is not None else 'Error'
                    elif self.enable_map:
                        row['mAP'] = 'No Predictions'

                # 4) Write the CSV back
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
        except Exception as e:
            #logging.error(f"Error processing folder {transform_folder}: {str(e)}")
            self.error_summary['unprocessed_folders'].append(str(transform_folder))

    def process_all(self) -> None:
        """Process all transformation folders in the class directory."""
        try:
            # Get all immediate subdirectories (transformation folders)
            transform_dirs = [d for d in self.class_dir.iterdir() if d.is_dir()]
            #logging.info(f"Found directories to process: {[str(d) for d in transform_dirs]}")
            
            if not transform_dirs:
                raise ValueError(f"No transformation directories found in {self.class_dir}")
            
            for transform_dir in transform_dirs:
                #logging.info(f"Processing transformation folder: {transform_dir}")
                if not any(transform_dir.rglob('*.jpg')) and not any(transform_dir.rglob('*.png')):
                    #logging.warning(f"No images found in {transform_dir}")
                    continue
                    
                self.process_transformation_folder(transform_dir)
                
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            self._generate_summary_report()
            
        except Exception as e:
            #logging.error(f"Error in main processing loop: {str(e)}")
            raise

    def _generate_summary_report(self) -> None:
        """Generate and save summary report."""
        report_path = self.class_dir / 'detection_summary_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("YOLOv8 Detection Analysis Summary Report\n")
            f.write("======================================\n\n")
            
            # Add mAP status
            f.write(f"mAP Calculation: {'Enabled' if self.enable_map else 'Disabled'}\n\n")
            
            f.write("1. Unprocessed Folders:\n")
            for folder in self.error_summary['unprocessed_folders']:
                f.write(f"   - {folder}\n")
            
            f.write("\n2. Missing Files:\n")
            for file in self.error_summary['missing_files']:
                f.write(f"   - {file}\n")
            
            f.write("\n3. Unreadable Images:\n")
            for image in self.error_summary['unreadable_images']:
                f.write(f"   - {image}\n")
            
            if self.enable_map:
                f.write("\n4. mAP Calculation Errors:\n")
                for error in self.error_summary['map_calculation_errors']:
                    f.write(f"   - {error}\n")

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='YOLOv8 Detection Analysis Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    # Basic arguments
    parser.add_argument('--list_classes', action='store_true',
                        help='List all available YOLO classes and their indices')
    parser.add_argument('--class_dir', type=str,
                        help='Directory for the specific class to process')
    parser.add_argument('--yolo_dir', type=str, required=True,
                        help='Directory containing YOLO model file')
    parser.add_argument('--enable_map', action='store_true',
                        help='Enable mAP calculation (requires annotation file)')
    parser.add_argument('--annotation_file', type=str,
                        help='Path to COCO-style annotations.json file (required if --enable_map is set)')
    parser.add_argument('--conf_threshold', type=float, default=0.01,
                    help='Confidence threshold for YOLO detection (default: 0.01)')
    
    # Add `--target_class` without relying on the model initially
    parser.add_argument('--target_class', type=str, required=True,
                        help='Target class code for YOLO detection (use --list_classes to see available options)')
    
    # Parse arguments
    args = parser.parse_args()

    # Handle `--list_classes` early
    if args.list_classes:
        model = YOLO(os.path.join(args.yolo_dir, 'yolov8n.pt'))
        print("\nAvailable YOLO classes:")
        for idx, name in model.names.items():
            print(f"{idx}: {name}")
        exit(0)

    # Validate `--class_dir`
    if not args.class_dir:
        parser.error("--class_dir is required when not listing classes")
    
    # Validate `--target_class` using the YOLO model
    model = YOLO(os.path.join(args.yolo_dir, 'yolov8n.pt'))
    if not args.target_class.isdigit() or int(args.target_class) not in model.names:
        parser.error(f"--target_class must be a valid class ID. Use --list_classes to see available options.")

    # Validate mAP requirements
    if args.enable_map and not args.annotation_file:
        parser.error("--annotation_file is required when --enable_map is set")
    
    return args

def main():
    """Main function."""
    try:
        start_time = time.time()
        args = parse_args()
        
        # Print the paths to verify they're correct
        print(f"YOLO directory: {args.yolo_dir}")
        print(f"Class directory: {args.class_dir}")
        print(f"Annotation file: {args.annotation_file}")
        
        # Check if the required files/directories exist
        print(f"YOLO model exists: {os.path.exists(os.path.join(args.yolo_dir, 'yolov8n.pt'))}")
        print(f"Class directory exists: {os.path.exists(args.class_dir)}")
        print(f"Annotation file exists: {os.path.exists(args.annotation_file)}")
        
        # Initialize COCO evaluator only if mAP is enabled
        coco_evaluator = None
        if args.enable_map:
            print("Initializing COCO evaluator...")
            coco_evaluator = COCOEvaluator(args.annotation_file)
        else:
            print("mAP calculation disabled")
        
        # Initialize YOLO detector
        print("Initializing YOLO detector...")
        detector = YOLODetector(args.yolo_dir, args.target_class, args.conf_threshold)
        
        # Initialize and run transformation analyzer
        print("Starting analysis...")
        analyzer = TransformationAnalyzer(args.class_dir, detector, coco_evaluator)
        analyzer.process_all()
        
        print("Processing completed successfully")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Total runtime: {elapsed_time:.2f} seconds")
        
    except Exception as e:
        print(f"Fatal error in main: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
#def main():
#    """Main function."""
#    try:
#        start_time = time.time()
#        args = parse_args()
#        
#        # Initialize COCO evaluator only if mAP is enabled
#        coco_evaluator = None
#        if args.enable_map:
            #logging.info("mAP calculation enabled - initializing COCO evaluator")
#            coco_evaluator = COCOEvaluator(args.annotation_file)
#        else:
#            print("mAP calculation disabled")
            #logging.info("mAP calculation disabled")
        
        # Initialize YOLO detector
#        detector = YOLODetector(args.yolo_dir, args.target_class, args.conf_threshold)
        
        # Initialize and run transformation analyzer
#        analyzer = TransformationAnalyzer(args.class_dir, detector, coco_evaluator)
#        analyzer.process_all()
        
        #logging.info("Processing completed successfully")

#        end_time = time.time()
#        elapsed_time = end_time - start_time
        #logging.info(f"Total runtime: {elapsed_time:.2f} seconds")
        
#    except Exception as e:
        #logging.error(f"Fatal error in main: {str(e)}")
#        raise

if __name__ == '__main__':
    main()