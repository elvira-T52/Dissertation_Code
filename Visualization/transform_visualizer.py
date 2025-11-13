import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from pathlib import Path
import logging
import time

"""
Hello, this script is to graph ECP for Case studies 2-3. Currently, there might be a few bugs, hence the logging. 

The Tranform visualizer grabs the data from the full test files, also on the GitHub, and then aggregates them based on the ranges. The mean is taken and this is used as the test.
I don't need all the information that I'm using from the test case; however, I do use a lot of it. I use the test case ID to help me quickly verify which requirements are satisfied.

If you have any questions or concerns, feel free to email me at elvirat@my.erau.edu. 
P.S. Anything I comment with TRE are notes for myself or anyone planning to re-implment this code.
"""




log_dir = 'logs'
Path(log_dir).mkdir(exist_ok=True)

# Remove later. TRE
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        #one log file per run with timestamp
        logging.FileHandler(os.path.join(log_dir, f'transform_visualizer_{time.strftime("%Y%m%d_%H%M%S")}.log'))
    ]
)
class TransformVisualizer:
    def __init__(self, metric_type='confidence', threshold=None): 
        self.metric_type = metric_type
        self.threshold = threshold or (0.85 if metric_type == 'confidence' else 0.5) ##Change this in case metrics need a better default, 0.85 might be *TOO* hard. TRE

    def read_data(self, csv_path):
        logging.info(f"Reading data from {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            
            # Check for necessary columns
            required_cols = []
            
            # Transform columns - check for different naming patterns
            transform_cols = [col for col in df.columns if col.startswith('transform')]
            if not transform_cols:
                raise ValueError("No transform columns found in the CSV")##Error handling TRE
            required_cols.extend(transform_cols)
            
            # Range columns
            range_cols = [col for col in df.columns if col.startswith('range')]
            if not range_cols:
                raise ValueError("No range columns found in the CSV")##EH TRE
            required_cols.extend(range_cols)
            
            # Applied value columns
            value_cols = [col for col in df.columns if col.startswith('applied_value')]
            if not value_cols:
                raise ValueError("No applied_value columns found in the CSV") ##EH TRE
            required_cols.extend(value_cols)
            
            # Metric column
            metric_col = 'confidence_score' if self.metric_type == 'confidence' else 'mAP'
            if metric_col not in df.columns:
                raise ValueError(f"Required column {metric_col} not found in CSV") ##EH TRE
            required_cols.append(metric_col)
            
            logging.info(f"Found required columns: {', '.join(required_cols)}")
            logging.info(f"Total records in CSV: {len(df)}")
            
            return df
            
        except Exception as e:
            logging.error(f"Error reading CSV: {str(e)}")
            raise

    def process_data(self, df):
        logging.info("Processing data and calculating metrics")
        
        # Identify transform columns and range columns
        transform_cols = [col for col in df.columns if col.startswith('transform')]
        range_cols = [col for col in df.columns if col.startswith('range')]
        value_cols = [col for col in df.columns if col.startswith('applied_value')]
        
        # Get metric column
        metric_col = 'confidence_score' if self.metric_type == 'confidence' else 'mAP'
        
        # Initialize results dictionary
        grouped_results = {} ##Not sure if I processed this exactly right, I could have used a Pd.dataframe.
        
        # Determine how many valid transform columns we have (with non-NA values)
        valid_transform_cols = [col for col in transform_cols if df[col].notna().any()]
        transform_count = len(valid_transform_cols)
        
        logging.info(f"Detected {transform_count} valid transform columns")
        
        if transform_count == 1:
            # SINGLE TRANSFORM CASE ##########THIS IS FOR 1D ECP
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
                    
                    # Calculate metrics
                    if len(range_data) > 0:
                        mean_metric = range_data[metric_col].mean()
                        passed = mean_metric > self.threshold
                        
                        # Store results
                        transform_key = str(t1_val)
                        if transform_key not in grouped_results:
                            grouped_results[transform_key] = {}
                        
                        range_key = str(r1_val)
                        grouped_results[transform_key][range_key] = {
                            'transform1': t1_val,
                            'transform2': None,
                            'range1': r1_val,
                            'range2': None,
                            'mean_' + self.metric_type: mean_metric,
                            'pass': passed,
                            'sample_size': len(range_data),
                            'min_value': range_data[v1_col].min(),
                            'max_value': range_data[v1_col].max(),
                            'mean_value': range_data[v1_col].mean()
                        }
        elif transform_count >= 2:
            # TWO TRANSFORM CASE - DETECT IF THERE IS MORE THAN 1 transform. t1-t1 = transforms1 & transform 2 in columns of the full test cases --TE.
            t1_col = valid_transform_cols[0]
            t2_col = valid_transform_cols[1]
            
            for t1_val in df[t1_col].unique():
                if pd.isna(t1_val):
                    continue
                    
                for t2_val in df[t2_col].unique():
                    if pd.isna(t2_val):
                        continue
                        
                    # Filter data for this transform combination -- Repeat from above
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
                    
                    # Error check in case we don't have a column of data.
                    if r2_col is None or v2_col is None:
                        logging.warning("Second transform detected but missing range or value columns")
                        continue
                    
                    # Process each range combination
                    for r1_val in t_data[r1_col].unique():
                        if pd.isna(r1_val):
                            continue
                            
                        for r2_val in t_data[r2_col].unique():
                            if pd.isna(r2_val):
                                continue
                                
                            # Filter data for this range combination using pandas
                            range_data = t_data[(t_data[r1_col] == r1_val) & 
                                            (t_data[r2_col] == r2_val)].copy()
                            
                            # Calculate metrics which either is mAP or Confidence
                            if len(range_data) > 0:
                                mean_metric = range_data[metric_col].mean()
                                passed = mean_metric > self.threshold
                                
                                # Store results for output for Test Cases.
                                transform_key = f"{t1_val}_{t2_val}"
                                if transform_key not in grouped_results:
                                    grouped_results[transform_key] = {}
                                
                                range_key = f"{r1_val}_{r2_val}" ##GROSS BUT GOTTA KEEP THE RANGES IN A DIC FOR THESE VALUES. THIS IS USED FOR THE TEST CASES TO VALIDATE FOR REQUIREMENTS. TRE
                                grouped_results[transform_key][range_key] = {
                                    'transform1': t1_val,
                                    'transform2': t2_val,
                                    'range1': r1_val,
                                    'range2': r2_val,
                                    'mean_' + self.metric_type: mean_metric,
                                    'pass': passed,
                                    'sample_size': len(range_data),
                                    'min_value1': range_data[v1_col].min(),
                                    'max_value1': range_data[v1_col].max(),
                                    'mean_value1': range_data[v1_col].mean(),
                                    'min_value2': range_data[v2_col].min(),
                                    'max_value2': range_data[v2_col].max(),
                                    'mean_value2': range_data[v2_col].mean()
                                }
        
        logging.info(f"Processed {len(grouped_results)} transform combinations")
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
        logging.info("Creating visualizations")
        
        # Make sure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Create visualizations for each transform combination
        for transform_key, ranges in grouped_results.items():
            # Skip if no data
            if not ranges:
                continue
                
            # Get transform values
            first_range = next(iter(ranges.values()))
            transform1 = first_range.get('transform1')
            transform2 = first_range.get('transform2')
            
            # Choose plot type based on number of transforms
            if transform2 is None:
                # 1D plot for single transform
                self._create_1d_plot(transform_key, ranges, output_dir, class_name)
            else:
                # 2D plot for transform pairs
                self._create_2d_plot(transform_key, ranges, output_dir, class_name)
    
    def _create_1d_plot(self, transform_key, ranges, output_dir, class_name=''): ##PLOTTING HERE
        logging.info(f"Creating 1D plot for {transform_key}")
        
        # Print the keys to verify what we have in the data ###REMOVE TRE
        ##for range_key, stats in ranges.items():
            ##logging.info(f"Range: {range_key}")
            ##logging.info(f"Stats keys: {list(stats.keys())}")
           ##if 'mean_value' in stats:
                ##logging.info(f"Mean value (parameter): {stats['mean_value']}")
            ##metric_key = f"mean_{self.metric_type}"
            ##if metric_key in stats:
                ##logging.info(f"Mean {self.metric_type}: {stats[metric_key]}")
        
        # Get transform name if possible
        transform_names = {
            'gaussian_blur': 'Gaussian Blur',
            'rotation': 'Rotation',
            'brightness': 'Brightness',
            'translation': 'Translation',
            'elastic_distortion': 'Elastic Distortion'
        }
        display_name = transform_names.get(transform_key, transform_key)
        
        plt.figure(figsize=(15, 3))
        plt.axhline(y=0, color='black', linewidth=2)
        # Extract data points
        points = []
        for range_key, stats in ranges.items():
            # Parse range
            min_val, max_val = self._parse_range(stats['range1'])
            
            # Get the parameter value for plotting
            if 'mean_value' in stats:
                param_value = stats['mean_value']
                logging.info(f"Using mean parameter value: {param_value}")
            else:
                param_value = (min_val + max_val) / 2
                logging.info(f"Using midpoint parameter value: {param_value}")
            
            # Get the metric value (confidence/mAP)
            metric_key = f"mean_{self.metric_type}"
            if metric_key in stats:
                metric_value = stats[metric_key]
                logging.info(f"Using {metric_key}: {metric_value}")
            else:
                metric_value = 0.0
                logging.warning(f"Missing {metric_key} for range {range_key}")
            
            passed = stats.get('pass', False)
            sample_size = stats.get('sample_size', 0)
            
            points.append((min_val, max_val, param_value, metric_value, passed, sample_size))
        
        # Sort by range minimum
        points.sort(key=lambda x: x[0])
        
        for min_val, max_val, param_value, metric_value, passed, sample_size in points:
            # Add range markers
            plt.plot([min_val, min_val], [-0.1, 0.1], 'k-', linewidth=1)
            plt.plot([max_val, max_val], [-0.1, 0.1], 'k-', linewidth=1)
            
            # Plot point at the parameter value (mean transform value)
            color = 'blue' if passed else 'red'
            plt.plot(param_value, 0, 'o', color=color, markersize=6, alpha=0.7)
            #############################################################Note this is for when the graph is too cluttered and i can't have the metric values in a good enough place to where they're sensible - TRE.
            # Add annotation showing the mean confidence/metric value 
            #plt.annotate(f"{metric_value:.2f}", 
                      ##  (param_value, 0), 
                      ##  textcoords="offset points",
                      ##  xytext=(0, 15), 
                      ##  ha='center')
                        
        # Add legend and formatting
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='blue', linestyle='None',
                      label=f'{self.metric_type} > {self.threshold}'),
            plt.Line2D([0], [0], marker='o', color='red', linestyle='None',
                      label=f'{self.metric_type} ≤ {self.threshold}')
        ]
        plt.legend(handles=legend_elements, loc='upper center', 
                  bbox_to_anchor=(0.5, 1.5), ncol=2)
        
        plt.yticks([])
        plt.grid(True, axis='x', alpha=0.3)
        
        title = f'{display_name} - {self.metric_type.capitalize()} by Parameter Range'
        if class_name:
            title += f' ({class_name})'
        plt.title(title, pad=20)
        plt.tight_layout()
        
        # Save the figure
        file_suffix = f"_{class_name}" if class_name else ""
        output_file = os.path.join(output_dir, f"{transform_key}_{self.metric_type}{file_suffix}.png")
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()
        
        logging.info(f"Saved 1D plot to {output_file}")
    
    def _create_2d_plot(self, transform_key, ranges, output_dir, class_name=''):
        logging.info(f"Creating 2D plot for {transform_key}")
        
        plt.figure(figsize=(12, 10))
        first_range = next(iter(ranges.values())) ###MIGHT HAVE A bug here. ####################################################################################################################################
        transform1 = first_range.get('transform1')
        transform2 = first_range.get('transform2')
        
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
        # Extract data points
        points = []
        for range_key, stats in ranges.items():
            # Parse ranges
            min1, max1 = self._parse_range(stats['range1'])
            min2, max2 = self._parse_range(stats['range2'])
            
            # Get midpoints for plotting
            mid1 = stats.get('mean_value1', (min1 + max1) / 2) ###KNOWN BUG HERE !!!!!!!!!!! ADDRESS-- TRE
            mid2 = stats.get('mean_value2', (min2 + max2) / 2)
            
            metric_val = stats[f"mean_{self.metric_type}"]
            passed = stats['pass']
            sample_size = stats['sample_size']
            
            points.append((min1, max1, min2, max2, mid1, mid2, metric_val, passed, sample_size))
        
        # Plot points
        for min1, max1, min2, max2, x, y, metric_val, passed, sample_size in points:
            color = 'blue' if passed else 'red'
            plt.scatter(x, y, c=color, s=50, alpha=0.7)
            
            ##Add annotation with metric value and sample size
            plt.annotate(f"{metric_val:.2f}", 
                        (x, y), 
                        textcoords="offset points",
                        xytext=(0, 7), 
                        ha='center')
            # Add rectangle to show range area
            plt.gca().add_patch(plt.Rectangle((min1, min2), 
                                              max1 - min1, 
                                              max2 - min2, 
                                              fill=False, 
                                              edgecolor='gray', 
                                              linestyle='--',
                                              alpha=0.5))
        
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
        
        title = f"{display_name1} vs {display_name2} - {self.metric_type.capitalize()} Scores"
        if class_name:
            title += f" ({class_name})"
        plt.title(title)
        
        # Save the figure
        file_suffix = f"_{class_name}" if class_name else ""
        output_file = os.path.join(output_dir, f"{transform_key}_{self.metric_type}{file_suffix}.png")
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()
        
        logging.info(f"Saved 2D plot to {output_file}")

    def generate_test_cases(self, grouped_results, output_dir, total_test_count, class_name=''):
        logging.info("Generating test cases CSV")
        
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
        for transform_key, ranges in grouped_results.items():
            # Prepare test case data for this transform combination
            test_cases = []
            passing_ranges = []
            
            logging.info(f"\n=== PROCESSING {transform_key} ===")
            
            for range_id, (range_key, stats) in enumerate(ranges.items(), 1):
                # Get transform names if possible
                transform1_name = transform_names.get(stats['transform1'], stats['transform1'])
                transform2_val = stats.get('transform2')
                transform2_name = transform_names.get(transform2_val, transform2_val) if transform2_val else 'N/A'
                
                # Create test case
                test_case = {
                    "test_id": f"TC_{transform_key}_{range_id}",
                    "transform1": stats['transform1'],
                    "transform1_name": transform1_name,
                    "transform2": stats.get('transform2', 'N/A'),
                    "transform2_name": transform2_name,
                    "range1": stats['range1'],
                    "range2": stats.get('range2', 'N/A'),
                    f"mean_{self.metric_type}": stats[f"mean_{self.metric_type}"],
                    "pass": stats['pass'],
                    "sample_size": stats['sample_size'],
                    "total_test_count": total_test_count
                }
                
                if class_name:
                    test_case["class_name"] = class_name
                
                test_cases.append(test_case)
                if stats['pass']: #####IF A PASSED TEST CASE
                    # Log passing range
                    if transform2_val:
                        log_msg = f"PASS: {transform1_name} {stats['range1']} + {transform2_name} {stats.get('range2', 'N/A')} -> {stats[f'mean_{self.metric_type}']:.3f}"
                    else:
                        log_msg = f"PASS: {transform1_name} {stats['range1']} -> {stats[f'mean_{self.metric_type}']:.3f}"
                    
                    logging.info(log_msg)
                    # Add to passing ranges list
                    passing_range = {
                        "transform_combination": transform_key,
                        "transform1": stats['transform1'],
                        "transform1_name": transform1_name,
                        "transform2": stats.get('transform2', 'N/A'),
                        "transform2_name": transform2_name,
                        "range1": stats['range1'],
                        "range2": stats.get('range2', 'N/A'),
                        f"mean_{self.metric_type}": stats[f"mean_{self.metric_type}"],
                        "sample_size": stats['sample_size']
                    }
                    
                    if class_name:
                        passing_range["class_name"] = class_name
                        
                    passing_ranges.append(passing_range)
            
            # Create DataFrames and save as CSVs for this transform combination
            if test_cases:
                test_case_df = pd.DataFrame(test_cases)
                
                # Add transform key and class name to filename if provided
                file_suffix = f"_{class_name}" if class_name else ""
                output_file = os.path.join(output_dir, f"test_cases_{transform_key}_{self.metric_type}{file_suffix}.csv")
                test_case_df.to_csv(output_file, index=False)
                
                logging.info(f"Generated {len(test_cases)} test cases for {transform_key}")
                logging.info(f"Test cases saved to {output_file}")
            
            if passing_ranges: ####THESE ARE NOT REALLY FOR THE USER BUT THESE ARE TO HELP ME SHOW WHICH REQUIREMENTS "PASSED" in my dissertation
                passing_df = pd.DataFrame(passing_ranges)
                file_suffix = f"_{class_name}" if class_name else ""
                passing_file = os.path.join(output_dir, f"passing_ranges_{transform_key}_{self.metric_type}{file_suffix}.csv")
                passing_df.to_csv(passing_file, index=False)
                
                logging.info(f"Generated summary of {len(passing_ranges)} passing ranges for {transform_key}")
                logging.info(f"Passing ranges saved to {passing_file}")
        return test_cases


def parse_args():
    parser = argparse.ArgumentParser(description='2D Transform Visualization and Test Case Generator') ###I REALLY LIKE CLI BECAUSE I CAN .bat FILE it ALL.
    parser.add_argument('csv_file', help='Path to transforms CSV file')

    metric_group = parser.add_mutually_exclusive_group() ## EITHER mAP or Confidence
    metric_group.add_argument('--confidence', action='store_true', default=True,
                            help='Plot confidence scores (default)')
    metric_group.add_argument('--map', action='store_true',
                            help='Plot mAP values instead of confidence')
    
    parser.add_argument('--threshold', type=float,
                       help='Threshold value (default: 0.85 for confidence, 0.5 for mAP)') ##User defined threshold for requirements, this should change if requirements change -TRE
    
    parser.add_argument('--output-dir', default='test_results',
                       help='Output directory for visualizations and test cases')
    
    parser.add_argument('--class-name', type=str, default='',
                       help='Class name to append to output files')
                       
    return parser.parse_args()


def main():
    try:
        args = parse_args()
        
        # Determine which metric to use
        metric = 'map' if args.map else 'confidence'
        
        # Set default threshold if not provided
        threshold = args.threshold
        if threshold is None:
            threshold = 0.5 if metric == 'map' else 0.85
            
        # Get class name
        class_name = args.class_name.strip() if args.class_name else ''
            
        logging.info(f"Starting visualization process for {metric} with threshold {threshold}")
        if class_name:
            logging.info(f"Class name: {class_name}")
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Initialize visualizer
        visualizer = TransformVisualizer(metric_type=metric, threshold=threshold)
        
        # Read data
        df = visualizer.read_data(args.csv_file)
        
        # Get total test count
        total_test_count = len(df)
        
        # Process data
        grouped_results = visualizer.process_data(df)
        
        # Create visualizations
        visualizer.create_visualizations(grouped_results, args.output_dir, class_name)
        
        # Generate test cases
        visualizer.generate_test_cases(grouped_results, args.output_dir, total_test_count, class_name)
        
        logging.info("Process completed successfully")
        
    except Exception as e:
        logging.error(f"Error: {str(e)}") ##I KNOW I NEED TO GET RID OF LOGGING BECAUSE COMPUTATION BUT, OH WELL, FUTURE WORK. ;) TRE
        raise


if __name__ == '__main__':
    main()
