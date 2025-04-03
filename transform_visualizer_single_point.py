import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from pathlib import Path
import logging
import time

log_dir = 'logs'
Path(log_dir).mkdir(exist_ok=True)

# Set up logging to both console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        # Console handler
        logging.StreamHandler(),
        # File handler - creates one log file per run with timestamp
        logging.FileHandler(os.path.join(log_dir, f'transform_visualizer_{time.strftime("%Y%m%d_%H%M%S")}.log'))
    ]
)
class TransformVisualizer:
    def __init__(self, metric_type='confidence', threshold=None):
        """Initialize visualizer with chosen metric and threshold.
        
        Args:
            metric_type (str): Either 'confidence' or 'map'
            threshold (float): Threshold value for coloring points
        """
        self.metric_type = metric_type
        self.threshold = threshold or (0.85 if metric_type == 'confidence' else 0.5)

    def read_data(self, csv_path):
        """Read and validate the CSV data.
        
        Args:
            csv_path (str): Path to CSV file
            
        Returns:
            pd.DataFrame: Processed dataframe
        """
        logging.info(f"Reading data from {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            
            # Check for necessary columns
            required_cols = []
            
            # Transform columns - check for different naming patterns
            transform_cols = [col for col in df.columns if col.startswith('transform')]
            if not transform_cols:
                raise ValueError("No transform columns found in the CSV")
            required_cols.extend(transform_cols)
            
            # Range columns
            range_cols = [col for col in df.columns if col.startswith('range')]
            if not range_cols:
                raise ValueError("No range columns found in the CSV")
            required_cols.extend(range_cols)
            
            # Applied value columns
            value_cols = [col for col in df.columns if col.startswith('applied_value')]
            if not value_cols:
                raise ValueError("No applied_value columns found in the CSV")
            required_cols.extend(value_cols)
            
            # Metric column
            metric_col = 'confidence_score' if self.metric_type == 'confidence' else 'mAP'
            if metric_col not in df.columns:
                raise ValueError(f"Required column {metric_col} not found in CSV")
            required_cols.append(metric_col)
            
            logging.info(f"Found required columns: {', '.join(required_cols)}")
            logging.info(f"Total records in CSV: {len(df)}")
            
            return df
            
        except Exception as e:
            logging.error(f"Error reading CSV: {str(e)}")
            raise

    def process_data(self, df):
        """Process data and group by transform combinations and ranges.
        Instead of calculating means, keep individual data points per range."""
        logging.info("Processing data for single point visualization")
        
        # Identify transform columns and range columns
        transform_cols = [col for col in df.columns if col.startswith('transform')]
        range_cols = [col for col in df.columns if col.startswith('range')]
        value_cols = [col for col in df.columns if col.startswith('applied_value')]
        
        # Get metric column
        metric_col = 'confidence_score' if self.metric_type == 'confidence' else 'mAP'
        
        # Initialize results dictionary
        grouped_results = {}
        
        # Determine how many valid transform columns we have (with non-NA values)
        valid_transform_cols = [col for col in transform_cols if df[col].notna().any()]
        transform_count = len(valid_transform_cols)
        
        logging.info(f"Detected {transform_count} valid transform columns")
        
        if transform_count == 1:
            # SINGLE TRANSFORM CASE
            t1_col = valid_transform_cols[0]
            for t1_val in df[t1_col].unique():
                if pd.isna(t1_val):
                    continue
                
                # Filter data for this transform
                t1_data = df[df[t1_col] == t1_val].copy()
                
                # Process each range
                r1_col = [col for col in range_cols if df[col].notna().any()][0]
                v1_col = [col for col in value_cols if df[col].notna().any()][0]
                
                for r1_val in t1_data[r1_col].unique():
                    if pd.isna(r1_val):
                        continue
                        
                    # Filter data for this range
                    range_data = t1_data[t1_data[r1_col] == r1_val].copy()
                    
                    # Instead of calculating means, store individual points
                    if len(range_data) > 0:
                        # Calculate threshold pass/fail for coloring
                        for idx, row in range_data.iterrows():
                            metric_value = row[metric_col]
                            passed = metric_value > self.threshold
                            
                            # Store results for each individual point
                            transform_key = str(t1_val)
                            if transform_key not in grouped_results:
                                grouped_results[transform_key] = {}
                            
                            # Use a unique key for each row that includes the applied value
                            point_key = f"{r1_val}_{row[v1_col]:.4f}"
                            grouped_results[transform_key][point_key] = {
                                'transform1': t1_val,
                                'transform2': None,
                                'range1': r1_val,
                                'range2': None,
                                'applied_value': row[v1_col],
                                self.metric_type: metric_value,
                                'pass': passed,
                                'image_name': row.get('image_name', f"img_{idx}")
                            }
        elif transform_count >= 2:
            # TWO TRANSFORM CASE - only handle first two transforms if more are present
            t1_col = valid_transform_cols[0]
            t2_col = valid_transform_cols[1]
            
            for t1_val in df[t1_col].unique():
                if pd.isna(t1_val):
                    continue
                    
                for t2_val in df[t2_col].unique():
                    if pd.isna(t2_val):
                        continue
                        
                    # Filter data for this transform combination
                    t_data = df[(df[t1_col] == t1_val) & (df[t2_col] == t2_val)].copy()
                    
                    if len(t_data) == 0:
                        continue
                    
                    # Get corresponding range columns
                    valid_range_cols = [col for col in range_cols if df[col].notna().any()]
                    r1_col = valid_range_cols[0]
                    r2_col = valid_range_cols[1] if len(valid_range_cols) > 1 else None
                    
                    valid_value_cols = [col for col in value_cols if df[col].notna().any()]
                    v1_col = valid_value_cols[0]
                    v2_col = valid_value_cols[1] if len(valid_value_cols) > 1 else None
                    
                    # Check if we really have 2D data
                    if r2_col is None or v2_col is None:
                        logging.warning("Second transform detected but missing range or value columns")
                        continue
                    
                    # Process each range combination with individual points
                    for idx, row in t_data.iterrows():
                        r1_val = row[r1_col]
                        r2_val = row[r2_col]
                        
                        if pd.isna(r1_val) or pd.isna(r2_val):
                            continue
                                
                        # Get metric and check if passed threshold
                        metric_value = row[metric_col]
                        passed = metric_value > self.threshold
                        
                        # Store individual point
                        transform_key = f"{t1_val}_{t2_val}"
                        if transform_key not in grouped_results:
                            grouped_results[transform_key] = {}
                        
                        # Use a unique key that includes both applied values
                        point_key = f"{r1_val}_{r2_val}_{row[v1_col]:.4f}_{row[v2_col]:.4f}"
                        grouped_results[transform_key][point_key] = {
                            'transform1': t1_val,
                            'transform2': t2_val,
                            'range1': r1_val,
                            'range2': r2_val,
                            'applied_value1': row[v1_col],
                            'applied_value2': row[v2_col],
                            self.metric_type: metric_value,
                            'pass': passed,
                            'image_name': row.get('image_name', f"img_{idx}")
                        }
        
        logging.info(f"Processed {len(grouped_results)} transform combinations with individual points")
        return grouped_results

    def _parse_range(self, range_str):
        """Parse range string to get min and max values."""
        try:
            min_val, max_val = map(float, range_str.strip('()').split(','))
            return min_val, max_val
        except Exception:
            logging.warning(f"Could not parse range string: {range_str}")
            return 0, 0
    
    def create_visualizations(self, grouped_results, output_dir, class_name=''):
        """Create visualizations for transform combinations with individual data points.
        
        Args:
            grouped_results (dict): Processed data
            output_dir (str): Directory to save visualizations
            class_name (str): Optional class name to append to output files
        """
        logging.info("Creating visualizations for individual points")
        
        # Make sure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Create visualizations for each transform combination
        for transform_key, points in grouped_results.items():
            # Skip if no data
            if not points:
                continue
                
            # Get transform values
            first_point = next(iter(points.values()))
            transform1 = first_point.get('transform1')
            transform2 = first_point.get('transform2')
            
            # Choose plot type based on number of transforms
            if transform2 is None:
                # 1D plot for single transform
                self._create_1d_plot(transform_key, points, output_dir, class_name)
            else:
                # 2D plot for transform pairs
                self._create_2d_plot(transform_key, points, output_dir, class_name)
    
    def _create_1d_plot(self, transform_key, points, output_dir, class_name=''):
        """Create 1D plot for single transform with individual points."""
        logging.info(f"Creating 1D plot for {transform_key} with individual points")
        
        # Get transform name if possible
        transform_names = {
            'gaussian_blur': 'Gaussian Blur',
            'rotation': 'Rotation',
            'brightness': 'Brightness',
            'translation': 'Translation',
            'elastic_distortion': 'Elastic Distortion'
        }
        display_name = transform_names.get(transform_key, transform_key)
        
        plt.figure(figsize=(15, 6))
        
        # If we have custom ranges, draw them first
        if hasattr(self, 'custom_range_sets') and len(self.custom_range_sets) >= 1:
            # custom_range_sets[0] => transform #1's ranges
            x_ranges = self.custom_range_sets[0]
            for x_min, x_max in x_ranges:
                plt.axvspan(x_min, x_max, 
                          color='lightgray', 
                          alpha=0.7,
                          edgecolor='Red',
                          linewidth=4)
        
        # Group points by range
        range_groups = {}
        for point_key, data in points.items():
            range_val = data['range1']
            if range_val not in range_groups:
                range_groups[range_val] = []
            range_groups[range_val].append(data)
        
        # Extract and organize data points
        for range_val, data_points in range_groups.items():
            # Parse range for visualization
            min_val, max_val = self._parse_range(range_val)
            
            # Plot points for this range
            for i, point in enumerate(data_points):
                applied_val = point['applied_value']
                metric_val = point[self.metric_type]
                passed = point['pass']
                
                # Plot point at the applied value with metric on y-axis
                color = 'blue' if passed else 'red'
                plt.scatter(applied_val, metric_val, color=color, s=30, alpha=0.7)
                
                # Optional: Add transparent vertical line to show range bounds
                if i == 0:  # Only add once per range
                    plt.axvspan(min_val, max_val, color='gray', alpha=0.1)
                    
                    # Add range min/max indicators
                    plt.axvline(x=min_val, color='gray', linestyle='--', alpha=0.5)
                    plt.axvline(x=max_val, color='gray', linestyle='--', alpha=0.5)
        
        # Add threshold line
        plt.axhline(y=self.threshold, color='green', linestyle='-', alpha=0.5, 
                   label=f'Threshold ({self.threshold})')
        
        # Add legend and formatting
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='blue', linestyle='None',
                      label=f'{self.metric_type} > {self.threshold}'),
            plt.Line2D([0], [0], marker='o', color='red', linestyle='None',
                      label=f'{self.metric_type} ≤ {self.threshold}'),
            plt.Line2D([0], [0], color='green', linestyle='-',
                      label=f'Threshold ({self.threshold})')
        ]
        plt.legend(handles=legend_elements, loc='best')
        
        plt.grid(True, alpha=0.3)
        plt.xlabel(f'{display_name} Parameter Value')
        plt.ylabel(f'{self.metric_type.capitalize()} Score')
        
        title = f'{display_name} - Individual {self.metric_type.capitalize()} Scores'
        if class_name:
            title += f' ({class_name})'
        plt.title(title)
        
        plt.tight_layout()
        
        # Save the figure
        file_suffix = f"_{class_name}" if class_name else ""
        output_file = os.path.join(output_dir, f"{transform_key}_{self.metric_type}_individual{file_suffix}.png")
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()
        
        logging.info(f"Saved 1D plot to {output_file}")
    
    def _create_2d_plot(self, transform_key, points, output_dir, class_name=''):
        """Create 2D plot for transform pairs with individual points."""
        logging.info(f"Creating 2D plot for {transform_key} with individual points")
        
        plt.figure(figsize=(12, 10))

        if hasattr(self, 'custom_range_sets') and len(self.custom_range_sets) >= 2:
            # custom_range_sets[0] => e.g. transform #1
            # custom_range_sets[1] => e.g. transform #2
            x_ranges = self.custom_range_sets[0]
            y_ranges = self.custom_range_sets[1]
            # We'll draw all possible rectangles in the product
            from itertools import product
            for (x_min, x_max), (y_min, y_max) in product(x_ranges, y_ranges):
                plt.gca().add_patch(plt.Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    fill=True,
                    facecolor='lightgray',
                    alpha=0.7,
                    edgecolor='Red',
                    linewidth=4
                ))

        
        # Get transform names directly from data
        first_point = next(iter(points.values()))
        transform1 = first_point.get('transform1')
        transform2 = first_point.get('transform2')
        
        # Get human-readable transform names if possible
        transform_names = {
            'gaussian_blur': 'Gaussian Blur',
            'rotation': 'Rotation',
            'brightness': 'Brightness',
            'translation': 'Translation',
            'elastic_distortion': 'Elastic Distortion'
        }
        display_name1 = transform_names.get(transform1, transform1)
        display_name2 = transform_names.get(transform2, transform2)
        
        # Group points by range combination
        range_groups = {}
        for point_key, data in points.items():
            range_key = f"{data['range1']}_{data['range2']}"
            if range_key not in range_groups:
                range_groups[range_key] = []
            range_groups[range_key].append(data)
        
        # Draw rectangles for ranges first (so they're behind points)
        for range_key, data_points in range_groups.items():
            if not data_points:
                continue
                
            # Get first point to determine range
            first_data = data_points[0]
            min1, max1 = self._parse_range(first_data['range1'])
            min2, max2 = self._parse_range(first_data['range2'])
            
            # Draw rectangle for the range
            plt.gca().add_patch(plt.Rectangle((min1, min2), 
                                            max1 - min1, 
                                            max2 - min2, 
                                            fill=True, 
                                            edgecolor='Black', 
                                            facecolor='lightgray',
                                            alpha=0.5,
                                            linewidth=2))
        
        # Now plot all individual points
        for point_key, data in points.items():
            # Extract values
            x = data['applied_value1']
            y = data['applied_value2']
            metric_val = data[self.metric_type]
            passed = data['pass']
            
            # Choose color based on passing threshold
            color = 'blue' if passed else 'red'
            
            # Plot the point
            # Optional: Use metric value for size/color intensity 
            plt.scatter(x, y, c=color, s=50, alpha=0.7)
                
            # Optional: Add small annotation with actual metric value
            # plt.annotate(f"{metric_val:.2f}", (x, y), fontsize=8, 
            #             xytext=(3, 3), textcoords='offset points')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='blue', linestyle='None',
                      label=f'{self.metric_type} > {self.threshold}'),
            plt.Line2D([0], [0], marker='o', color='red', linestyle='None',
                      label=f'{self.metric_type} ≤ {self.threshold}')
        ]
        plt.legend(handles=legend_elements)
        
        plt.grid(True, alpha=0.3)
        plt.xlabel(f"{display_name1} Parameter Value")
        plt.ylabel(f"{display_name2} Parameter Value")
        
        title = f"{display_name1} vs {display_name2} - Individual {self.metric_type.capitalize()} Scores"
        if class_name:
            title += f" ({class_name})"
        plt.title(title)
        
        # Save the figure
        file_suffix = f"_{class_name}" if class_name else ""
        output_file = os.path.join(output_dir, f"{transform_key}_{self.metric_type}_individual{file_suffix}.png")
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()
        
        logging.info(f"Saved 2D plot to {output_file}")

    def generate_test_cases(self, grouped_results, output_dir, total_test_count, class_name=''):
        """Generate test cases CSV from the visualization data."""
        logging.info("Generating test cases CSV for individual points")
        
        # Make sure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get transform name mapping
        transform_names = {
            'gaussian_blur': 'Gaussian Blur',
            'rotation': 'Rotation',
            'brightness': 'Brightness',
            'translation': 'Translation',
            'elastic_distortion': 'Elastic Distortion'
        }
        
        # Generate separate files for each transform combination
        for transform_key, points in grouped_results.items():
            # Prepare test case data for this transform combination
            test_cases = []
            passing_points = []
            
            logging.info(f"\n=== PROCESSING {transform_key} ===")
            
            for point_id, (point_key, data) in enumerate(points.items(), 1):
                # Get transform names if possible
                transform1_name = transform_names.get(data['transform1'], data['transform1'])
                transform2_val = data.get('transform2')
                transform2_name = transform_names.get(transform2_val, transform2_val) if transform2_val else 'N/A'
                
                # Create test case
                test_case = {
                    "test_id": f"TC_{transform_key}_{point_id}",
                    "transform1": data['transform1'],
                    "transform1_name": transform1_name,
                    "range1": data['range1'],
                }
                
                # Add applied_value
                if transform2_val:
                    test_case["applied_value1"] = data['applied_value1']
                    test_case["transform2"] = transform2_val
                    test_case["transform2_name"] = transform2_name
                    test_case["range2"] = data['range2']
                    test_case["applied_value2"] = data['applied_value2']
                else:
                    test_case["applied_value"] = data['applied_value']
                    test_case["transform2"] = "N/A"
                    test_case["transform2_name"] = "N/A"
                    test_case["range2"] = "N/A"
                
                # Add metric and pass/fail
                test_case[self.metric_type] = data[self.metric_type]
                test_case["pass"] = data['pass']
                test_case["image_name"] = data.get('image_name', f"img_{point_id}")
                test_case["total_test_count"] = total_test_count
                
                if class_name:
                    test_case["class_name"] = class_name
                
                test_cases.append(test_case)
                
                # Add to passing points if applicable
                if data['pass']:
                    # Log passing point
                    if transform2_val:
                        log_msg = f"PASS: {transform1_name} {data['applied_value1']:.2f} + {transform2_name} {data['applied_value2']:.2f} -> {data[self.metric_type]:.3f}"
                    else:
                        log_msg = f"PASS: {transform1_name} {data['applied_value']:.2f} -> {data[self.metric_type]:.3f}"
                    
                    logging.info(log_msg)
                    
                    # Add to passing points list
                    passing_point = {
                        "transform_combination": transform_key,
                        "transform1": data['transform1'],
                        "transform1_name": transform1_name,
                        "range1": data['range1']
                    }
                    
                    # Add applied_value
                    if transform2_val:
                        passing_point["applied_value1"] = data['applied_value1']
                        passing_point["transform2"] = transform2_val
                        passing_point["transform2_name"] = transform2_name
                        passing_point["range2"] = data['range2']
                        passing_point["applied_value2"] = data['applied_value2']
                    else:
                        passing_point["applied_value"] = data['applied_value']
                        passing_point["transform2"] = "N/A"
                        passing_point["transform2_name"] = "N/A"
                        passing_point["range2"] = "N/A"
                    
                    passing_point[self.metric_type] = data[self.metric_type]
                    passing_point["image_name"] = data.get('image_name', f"img_{point_id}")
                    
                    if class_name:
                        passing_point["class_name"] = class_name
                        
                    passing_points.append(passing_point)
            
            # Create DataFrames and save as CSVs for this transform combination
            if test_cases:
                test_case_df = pd.DataFrame(test_cases)
                
                # Add transform key and class name to filename if provided
                file_suffix = f"_{class_name}" if class_name else ""
                output_file = os.path.join(output_dir, f"individual_test_cases_{transform_key}_{self.metric_type}{file_suffix}.csv")
                test_case_df.to_csv(output_file, index=False)
                
                logging.info(f"Generated {len(test_cases)} individual test cases for {transform_key}")
                logging.info(f"Test cases saved to {output_file}")
            
            if passing_points:
                passing_df = pd.DataFrame(passing_points)
                
                # Add transform key and class name to filename if provided
                file_suffix = f"_{class_name}" if class_name else ""
                passing_file = os.path.join(output_dir, f"individual_passing_points_{transform_key}_{self.metric_type}{file_suffix}.csv")
                passing_df.to_csv(passing_file, index=False)
                
                logging.info(f"Generated summary of {len(passing_points)} passing individual points for {transform_key}")
                logging.info(f"Passing points saved to {passing_file}")
            
        # Return all test cases (though we're not using this return value currently)
        return test_cases

# near the top or bottom of transform_visualizer_single_point.py
def parse_range_file(range_file_path):
    """
    Reads a file with lines containing comma-separated (min,max) pairs.
    Returns a list of lists: each line is a list of (min, max) tuples.
    """
    import os
    import logging

    if not os.path.isfile(range_file_path):
        logging.error(f"Range file not found: {range_file_path}")
        return []

    with open(range_file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    all_range_sets = []
    for line in lines:
        pair_strings = line.split(',')
        # reassemble each pair of tokens into "(-30.0,-24.0)" style strings
        reassembled = []
        for i in range(0, len(pair_strings), 2):
            joined_pair = pair_strings[i] + ',' + pair_strings[i+1]
            reassembled.append(joined_pair)

        range_list = []
        for p in reassembled:
            p = p.strip()
            p = p.strip('()')  # remove parentheses like ()
            parts = p.split(',')
            if len(parts) == 2:
                try:
                    mn = float(parts[0])
                    mx = float(parts[1])
                    range_list.append((mn, mx))
                except ValueError:
                    logging.warning(f"Could not parse pair: {p}")
        all_range_sets.append(range_list)

    return all_range_sets

def parse_args():
    parser = argparse.ArgumentParser(description='2D Transform Visualization and Test Case Generator for Individual Points')
    
    parser.add_argument('csv_file', help='Path to transforms CSV file')

    # Metric selection (mutually exclusive)
    metric_group = parser.add_mutually_exclusive_group()
    metric_group.add_argument('--confidence', action='store_true', default=True,
                              help='Plot confidence scores (default)')
    metric_group.add_argument('--map', action='store_true',
                              help='Plot mAP values instead of confidence')

    # Threshold
    parser.add_argument('--threshold', type=float,
                        help='Threshold value (default: 0.85 for confidence, 0.5 for mAP)')

    parser.add_argument('--output-dir', default='test_results',
                        help='Output directory for visualizations and test cases')

    parser.add_argument('--class-name', type=str, default='',
                        help='Class name to append to output files')

    # NEW ARG FOR OPTIONAL RANGES
    parser.add_argument('--range-file', type=str, default='',
                        help='Optional path to a file containing custom ranges to build a grid system')

    return parser.parse_args()



def main():
    try:
        args = parse_args()
        
        # Determine which metric to use
        metric = 'map' if args.map else 'confidence'
        threshold = args.threshold
        if threshold is None:
            threshold = 0.5 if metric == 'map' else 0.85
        
        logging.info(f"Starting individual point visualization process for {metric} with threshold {threshold}")
        
        # OPTIONAL: Parse a range file if specified
        custom_range_sets = []
        if args.range_file:
            logging.info(f"Reading custom ranges from {args.range_file}")
            custom_range_sets = parse_range_file(args.range_file)
            logging.info(f"Found {len(custom_range_sets)} lines of ranges in the file")
        
        # create output dir
        os.makedirs(args.output_dir, exist_ok=True)

        # Initialize our visualizer
        visualizer = TransformVisualizer(metric_type=metric, threshold=threshold)

        # read the CSV as usual
        df = visualizer.read_data(args.csv_file)
        total_test_count = len(df)

        # Now, if you have custom ranges, you can incorporate them in your plotting or processing
        # For example, if there are 2 lines in the file, that might correspond to transform #1 and transform #2
        # And you can compute the product of those subranges to build "grid lines" or something similar.

        # 1) Process data from CSV
        grouped_results = visualizer.process_data(df)

        # 2) If you want to replace the range information from the CSV with your own "grid system,"
        #    you could pass `custom_range_sets` down to the create_visualizations function
        #    (and update that function to use your ranges instead).
        #    Or you can do an additional step to show the ranges from the file as bold boundaries.
        # For a quick demo, let's store them in the visualizer in a new attribute:
        visualizer.custom_range_sets = custom_range_sets

        # create visualizations
        visualizer.create_visualizations(grouped_results, args.output_dir, args.class_name)

        # generate test cases
        visualizer.generate_test_cases(grouped_results, args.output_dir, total_test_count, args.class_name)
        
        logging.info("Process completed successfully")
        
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise



if __name__ == '__main__':
    main()
