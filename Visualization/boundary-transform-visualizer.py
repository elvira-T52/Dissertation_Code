import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from pathlib import Path
import logging
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class BoundaryTransformVisualizer:
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
            
            # More flexible column detection
            transform_cols = [col for col in df.columns if 'transform' in col.lower()]
            if not transform_cols:
                # Try finding columns with 'original' if no transform columns found
                transform_cols = [col for col in df.columns if 'original' in col.lower()]
            if not transform_cols:
                raise ValueError("No transform columns found in the CSV")
            
            range_cols = [col for col in df.columns if 'range' in col.lower()]
            if not range_cols:
                raise ValueError("No range columns found in the CSV")
            
            value_cols = [col for col in df.columns if 'applied' in col.lower()]
            if not value_cols:
                raise ValueError("No applied value columns found in the CSV")
            
            # Metric column - check both confidence and mAP
            if self.metric_type == 'confidence':
                if 'confidence_score' in df.columns:
                    metric_col = 'confidence_score'
                else:
                    raise ValueError("No confidence_score column found in CSV")
                logging.info(f"Using confidence column: {metric_col}")
            else:
                if 'mAP' not in df.columns:
                    raise ValueError("No mAP column found in CSV")
                metric_col = 'mAP'
            
            logging.info(f"Found transform columns: {transform_cols}")
            logging.info(f"Found range columns: {range_cols}")
            logging.info(f"Found value columns: {value_cols}")
            logging.info(f"Using metric column: {metric_col}")
            logging.info(f"Total records in CSV: {len(df)}")
            
            return df
            
        except Exception as e:
            logging.error(f"Error reading CSV: {str(e)}")
            raise

    def process_boundary_data(self, df):
        """Process data focusing on boundary points without averaging positions.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            dict: Grouped boundary results
        """
        logging.info("Processing boundary value data and calculating metrics")
        
        # Initialize results dictionary
        boundary_results = {}
        
        # Get unique combinations of transforms
        transform_pairs = []
        transform_cols = [col for col in df.columns if 'transform' in col.lower()]
        
        # Check if we have two transforms with valid data
        if len(transform_cols) == 2 and df[transform_cols[0]].notna().any() and df[transform_cols[1]].notna().any():
            transform_pairs.append((transform_cols[0], transform_cols[1]))
        else:
            # Single transform case
            for t_col in transform_cols:
                if df[t_col].notna().any():
                    transform_pairs.append((t_col, None))
        
        # Process each transform pair
        for t1_col, t2_col in transform_pairs:
            if t2_col is None:
                # Single transform (1D) - exactly 4 boundary points per range
                for t1_val in df[t1_col].unique():
                    if pd.isna(t1_val):
                        continue
                    
                    # Filter data for this transform
                    t1_data = df[df[t1_col] == t1_val].copy()
                    
                    # Get corresponding range and value columns
                    range_cols = [col for col in df.columns if 'range' in col.lower()]
                    value_cols = [col for col in df.columns if 'applied' in col.lower()]
                    
                    r1_col = range_cols[0]  # Use first range column
                    v1_col = value_cols[0]  # Use first applied value column
                    
                    for r1_val in t1_data[r1_col].unique():
                        if pd.isna(r1_val):
                            continue
                            
                        # Filter data for this range
                        range_data = t1_data[t1_data[r1_col] == r1_val].copy()
                        
                        if len(range_data) > 0:
                            # Extract min/max range values
                            min_val, max_val = self._parse_range(r1_val)
                            
                            # Group points by their exact values
                            boundary_points = defaultdict(list)
                            
                            # Extract boundary values and metrics
                            metric_col = 'confidence_score' if self.metric_type == 'confidence' else 'mAP'
                            for _, row in range_data.iterrows():
                                applied_val = row[v1_col]
                                metric_val = row[metric_col]
                                boundary_points[applied_val].append(metric_val)
                            
                            # Calculate mean confidence for each boundary point
                            boundary_metrics = {}
                            for point_val, metrics in boundary_points.items():
                                mean_metric = sum(metrics) / len(metrics)
                                passed = mean_metric > self.threshold
                                
                                # Determine point type based on its value
                                if point_val == min_val:
                                    point_type = 'min'
                                elif point_val > min_val and point_val < max_val:
                                    if abs(point_val - min_val) < abs(point_val - max_val):
                                        point_type = 'min_plus'
                                    else:
                                        point_type = 'max_minus'
                                else:
                                    point_type = 'max'
                                    
                                boundary_metrics[point_val] = {
                                    'mean_' + self.metric_type: mean_metric,
                                    'pass': passed,
                                    'sample_size': len(metrics),
                                    'point_type': point_type
                                }
                            
                            # Check if all boundary points pass
                            all_pass = all(info['pass'] for info in boundary_metrics.values())
                            
                            # Store results
                            transform_key = str(t1_val)
                            if transform_key not in boundary_results:
                                boundary_results[transform_key] = {}
                            
                            range_key = str(r1_val)
                            boundary_results[transform_key][range_key] = {
                                'transform1': t1_val,
                                'transform2': None,
                                'range1': r1_val,
                                'range2': None,
                                'boundary_points': boundary_metrics,
                                'all_pass': all_pass,
                                'total_points': len(boundary_metrics),
                                'range_min': min_val,
                                'range_max': max_val
                            }
            else:
                # Transform pair (2D) - expected 8 boundary points per range pair
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
                        
                        # Get corresponding range and value columns
                        range_cols = [col for col in df.columns if 'range' in col.lower()]
                        value_cols = [col for col in df.columns if 'applied' in col.lower()]
                        
                        r1_col = range_cols[0]  # Use first range column
                        r2_col = range_cols[1]  # Use second range column
                        v1_col = value_cols[0]  # Use first value column
                        v2_col = value_cols[1]  # Use second value column
                        
                        # Process each range combination
                        for r1_val in t_data[r1_col].unique():
                            if pd.isna(r1_val):
                                continue
                                
                            for r2_val in t_data[r2_col].unique():
                                if pd.isna(r2_val):
                                    continue
                                    
                                # Filter data for this range combination
                                range_data = t_data[(t_data[r1_col] == r1_val) & 
                                                  (t_data[r2_col] == r2_val)].copy()
                                
                                if len(range_data) > 0:
                                    # Group by boundary value pairs (not taking mean of positions)
                                    boundary_points = {}
                                    
                                    # Extract boundary values and metrics
                                    metric_col = 'confidence_score' if self.metric_type == 'confidence' else 'mAP'
                                    for _, row in range_data.iterrows():
                                        boundary_val1 = row[v1_col]
                                        boundary_val2 = row[v2_col]
                                        metric_val = row[metric_col]
                                        
                                        point_key = (boundary_val1, boundary_val2)
                                        if point_key not in boundary_points:
                                            boundary_points[point_key] = []
                                        boundary_points[point_key].append(metric_val)
                                    
                                    # Calculate mean confidence for each boundary point
                                    boundary_metrics = {}
                                    for point_key, metrics in boundary_points.items():
                                        mean_metric = sum(metrics) / len(metrics)
                                        passed = mean_metric > self.threshold
                                        boundary_metrics[point_key] = {
                                            'mean_' + self.metric_type: mean_metric,
                                            'pass': passed,
                                            'sample_size': len(metrics)
                                        }
                                    
                                    # Check if all boundary points pass
                                    all_pass = all(info['pass'] for info in boundary_metrics.values())
                                    
                                    # Extract min/max range values
                                    min_val1, max_val1 = self._parse_range(r1_val)
                                    min_val2, max_val2 = self._parse_range(r2_val)
                                    
                                    # Store results
                                    transform_key = f"{t1_val}_{t2_val}"
                                    if transform_key not in boundary_results:
                                        boundary_results[transform_key] = {}
                                    
                                    range_key = f"{r1_val}_{r2_val}"
                                    boundary_results[transform_key][range_key] = {
                                        'transform1': t1_val,
                                        'transform2': t2_val,
                                        'range1': r1_val,
                                        'range2': r2_val,
                                        'boundary_points': boundary_metrics,
                                        'all_pass': all_pass,
                                        'total_points': len(boundary_metrics),
                                        'range_min1': min_val1,
                                        'range_max1': max_val1,
                                        'range_min2': min_val2,
                                        'range_max2': max_val2
                                    }
        
        logging.info(f"Processed {len(boundary_results)} transform combinations with boundary value analysis")
        return boundary_results

    def _parse_range(self, range_str):
        """Parse range string to get min and max values."""
        try:
            min_val, max_val = map(float, range_str.strip('()').split(','))
            return min_val, max_val
        except Exception:
            logging.warning(f"Could not parse range string: {range_str}")
            return 0, 0
    
    def create_boundary_visualizations(self, boundary_results, output_dir, class_name='', show_metrics=False, separate_ranges=False):
        """Create visualizations for boundary value analysis results.
        
        Args:
            boundary_results (dict): Results from boundary value analysis
            output_dir (str): Directory to save visualizations
            class_name (str, optional): Class name for plot titles. Defaults to ''.
            show_metrics (bool, optional): Whether to show metrics on plots. Defaults to False.
            separate_ranges (bool, optional): Whether to create separate plots for each range. Defaults to False.
        """
        logging.info("Creating boundary value visualizations")
        os.makedirs(output_dir, exist_ok=True)
        
        for transform_key, data in boundary_results.items():
            # Get first range to check transform structure
            first_range = next(iter(data.values()))
            transform1 = first_range['transform1']
            transform2 = first_range.get('transform2')
            
            if transform2 is None:
                # 1D visualization
                logging.info(f"Creating 1D boundary plot for {transform1}")
                self._create_1d_boundary_plot(data, transform1, output_dir, class_name, show_metrics, separate_ranges)
            else:
                # 2D visualization
                logging.info(f"Creating 2D boundary plot for {transform1} and {transform2}")
                self._create_2d_boundary_plot(data, transform1, transform2, output_dir, class_name, show_metrics, separate_ranges)
    
    def _create_1d_boundary_plot(self, boundary_data, transform_key, output_dir, class_name='', show_metrics=False, separate_ranges=False):
        """Create 1D boundary plot showing pass/fail regions."""
        logging.info(f"Creating 1D boundary plot for {transform_key}")
        
        # Get transform name if possible
        transform_names = {
            'gaussian_blur': 'Gaussian Blur',
            'rotation': 'Rotation',
            'brightness': 'Brightness',
            'translation': 'Translation',
            'elastic_distortion': 'Elastic Distortion'
        }
        display_name = transform_names.get(transform_key, transform_key)
        
        # Create a single figure if not using separate ranges
        if not separate_ranges:
            plt.figure(figsize=(15, 3))
            plt.gca().get_yaxis().set_visible(False)
            plt.ylim(-0.2, 0.2)
            plt.axhline(y=0, color='black', linewidth=2)
        
        # Process each range
        for range_key, data in boundary_data.items():
            # Create a new figure only if using separate ranges
            if separate_ranges:
                plt.figure(figsize=(15, 3))
                plt.gca().get_yaxis().set_visible(False)
                plt.ylim(-0.2, 0.2)
                plt.axhline(y=0, color='black', linewidth=2)
            
            range_min = data['range_min']
            range_max = data['range_max']
            
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
            
            # Add vertical lines at range boundaries
            plt.plot([range_min, range_min], [-0.1, 0.1], 'k-', linewidth=1)
            plt.plot([range_max, range_max], [-0.1, 0.1], 'k-', linewidth=1)
            
            # Plot each boundary point
            for i, (point_val, point_info) in enumerate(data['boundary_points'].items(), 1):
                # Calculate vertical offset based on position
                if point_info['point_type'] in ['min', 'max']:  # Boundary points
                    y_offset = 0.05
                else:  # Inner points
                    y_offset = 0.03
                
                # Determine color based on pass/fail and point type
                color = 'blue' if point_info['pass'] else 'red'
                plt.plot(point_val, y_offset, 'o', color=color, markersize=3)
                
                # Add value annotation above the point if show_metrics is True
                if show_metrics:
                    plt.annotate(f"{point_val}", 
                                xy=(point_val, y_offset),
                                xytext=(5 if i % 2 == 0 else -5, 5),  # Alternate between left and right offset
                                textcoords='offset points',
                                ha='left' if i % 2 == 0 else 'right',  # Align text based on offset direction
                                va='bottom',
                                arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5))
            
            # If using separate ranges, add legend and save for each range
            if separate_ranges:
                # Add legend
                legend_elements = [
                    plt.Line2D([0], [0], marker='o', color='blue', linestyle='None',
                              label=f'Pass ({self.metric_type} > {self.threshold})'),
                    plt.Line2D([0], [0], marker='o', color='red', linestyle='None',
                              label=f'Fail ({self.metric_type} ≤ {self.threshold})')
                ]
                plt.legend(handles=legend_elements, loc='upper left')
                
                # Add grid and labels
                plt.grid(True, alpha=0.3)
                plt.xlabel(f'{display_name} Parameter Value')
                
                # Set title
                title = f'{display_name} - Boundary Values'
                if class_name:
                    title += f' ({class_name})'
                plt.title(title)
                
                # Save the figure with range-specific filename
                file_suffix = f"_{class_name}" if class_name else ""
                range_suffix = f"_range_{range_min}_{range_max}".replace('.', 'p')
                output_file = os.path.join(output_dir, f"{transform_key}_boundary{range_suffix}{file_suffix}.png")
                plt.savefig(output_file, bbox_inches='tight', dpi=300)
                plt.close()
                logging.info(f"Saved boundary plot to {output_file}")
        
        # If not using separate ranges, add legend and save single figure
        if not separate_ranges:
            # Add legend
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='blue', linestyle='None',
                          label=f'Pass ({self.metric_type} > {self.threshold})'),
                plt.Line2D([0], [0], marker='o', color='red', linestyle='None',
                          label=f'Fail ({self.metric_type} ≤ {self.threshold})')
            ]
            plt.legend(handles=legend_elements, loc='upper left')
            
            # Add grid and labels
            plt.grid(True, alpha=0.3)
            plt.xlabel(f'{display_name} Parameter Value')
            
            # Set title
            title = f'{display_name} - Boundary Values'
            if class_name:
                title += f' ({class_name})'
            plt.title(title)
            
            # Save the figure
            file_suffix = f"_{class_name}" if class_name else ""
            output_file = os.path.join(output_dir, f"{transform_key}_boundary{file_suffix}.png")
            plt.savefig(output_file, bbox_inches='tight', dpi=300)
            plt.close()
            logging.info(f"Saved boundary plot to {output_file}")
    
    def _create_2d_boundary_plot(self, boundary_data, transform1, transform2, output_dir, class_name='', show_metrics=False, separate_ranges=False):
        """Create 2D plot for transform pairs with 8 boundary points."""
        logging.info(f"Creating 2D boundary plot for {transform1} and {transform2}")
        
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
        
        # Create a single figure if not using separate ranges
        if not separate_ranges:
            plt.figure(figsize=(12, 10))
            
            # If we have custom ranges, draw them first
            if hasattr(self, 'custom_range_sets') and len(self.custom_range_sets) >= 2:
                x_ranges = self.custom_range_sets[0]
                y_ranges = self.custom_range_sets[1]
                from itertools import product
                for (x_min, x_max), (y_min, y_max) in product(x_ranges, y_ranges):
                    plt.gca().add_patch(plt.Rectangle(
                        (x_min, y_min),
                        x_max - x_min,
                        y_max - y_min,
                        facecolor='none',
                        alpha=0.7,
                        edgecolor='darkgreen',
                        linewidth=2
                    ))
        
        # Process each range combination
        for range_key, stats in boundary_data.items():
            # Create a new figure if using separate ranges
            if separate_ranges:
                plt.figure(figsize=(12, 10))
                
                # If we have custom ranges, draw them first
                if hasattr(self, 'custom_range_sets') and len(self.custom_range_sets) >= 2:
                    x_ranges = self.custom_range_sets[0]
                    y_ranges = self.custom_range_sets[1]
                    from itertools import product
                    for (x_min, x_max), (y_min, y_max) in product(x_ranges, y_ranges):
                        plt.gca().add_patch(plt.Rectangle(
                            (x_min, y_min),
                            x_max - x_min,
                            y_max - y_min,
                            facecolor='lightgreen',
                            alpha=0.3,
                            edgecolor='darkgreen',
                            linewidth=3
                        ))
            
            # Extract range values
            range_min1 = stats['range_min1']
            range_max1 = stats['range_max1']
            range_min2 = stats['range_min2']
            range_max2 = stats['range_max2']
            
            # Plot rectangle to show range boundaries
            rect = plt.Rectangle((range_min1, range_min2), 
                                range_max1 - range_min1, 
                                range_max2 - range_min2,
                                fill=False, 
                                edgecolor='black', 
                                linestyle='-',
                                linewidth=1)
            plt.gca().add_patch(rect)
            
            # Plot each boundary point
            boundary_points = stats['boundary_points']
            
            for (x, y), point_info in boundary_points.items():
                metric_val = point_info[f"mean_{self.metric_type}"]
                passed = point_info['pass']
                
                # Determine color based on pass/fail
                color = 'blue' if passed else 'red'
                
                # Plot point
                plt.scatter(x, y, c=color, s=30, alpha=0.8)
                
                # Add annotation with metric value if show_metrics is True
                if show_metrics:
                    plt.annotate(f"{metric_val:.2f}", 
                                (x, y), 
                                textcoords="offset points",
                                xytext=(0, 10), 
                                ha='center',
                                fontsize=9)
            
            # If using separate ranges, add legend and save for each range
            if separate_ranges:
                # Add legend
                legend_elements = [
                    plt.Line2D([0], [0], marker='o', color='blue', linestyle='None',
                              label=f'{self.metric_type} > {self.threshold}'),
                    plt.Line2D([0], [0], marker='o', color='red', linestyle='None',
                              label=f'{self.metric_type} ≤ {self.threshold}')
                ]
                plt.legend(handles=legend_elements, loc='upper left')
                
                plt.grid(True, alpha=0.3)
                plt.xlabel(f"{display_name1} Parameter Value")
                plt.ylabel(f"{display_name2} Parameter Value")
                
                title = f"{display_name1} vs {display_name2} - Boundary Value Analysis ({self.metric_type.capitalize()})"
                title += f"\nRange: ({range_min1},{range_max1}) × ({range_min2},{range_max2})"
                if class_name:
                    title += f" ({class_name})"
                plt.title(title)
                
                # Save the figure with range-specific filename
                file_suffix = f"_{class_name}" if class_name else ""
                range_suffix = f"_range_{range_min1}_{range_max1}_{range_min2}_{range_max2}".replace('.', 'p')
                output_file = os.path.join(output_dir, f"{transform1}_{transform2}_bva_{self.metric_type}{range_suffix}{file_suffix}.png")
                plt.savefig(output_file, bbox_inches='tight', dpi=300)
                plt.close()
                logging.info(f"Saved 2D boundary plot to {output_file}")
        
        # If not using separate ranges, add legend and save single figure
        if not separate_ranges:
            # Add legend
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='blue', linestyle='None',
                          label=f'{self.metric_type} > {self.threshold}'),
                plt.Line2D([0], [0], marker='o', color='red', linestyle='None',
                          label=f'{self.metric_type} ≤ {self.threshold}')
            ]
            plt.legend(handles=legend_elements, loc='upper left')
            
            plt.grid(True, alpha=0.3)
            plt.xlabel(f"{display_name1} Parameter Value")
            plt.ylabel(f"{display_name2} Parameter Value")
            
            title = f"{display_name1} vs {display_name2} - Boundary Value Analysis ({self.metric_type.capitalize()})"
            if class_name:
                title += f" ({class_name})"
            plt.title(title)
            
            # Save the figure
            file_suffix = f"_{class_name}" if class_name else ""
            output_file = os.path.join(output_dir, f"{transform1}_{transform2}_bva_{self.metric_type}{file_suffix}.png")
            plt.savefig(output_file, bbox_inches='tight', dpi=300)
            plt.close()
            logging.info(f"Saved 2D boundary plot to {output_file}")

    def generate_test_cases(self, boundary_results, output_dir, total_test_count, class_name=''):
        """Generate test cases CSV from the boundary value analysis.
        
        Args:
            boundary_results (dict): Processed boundary data
            output_dir (str): Directory to save output
            total_test_count (int): Total number of test cases in the original file
            class_name (str): Optional class name to append to output files
        """
        logging.info("Generating boundary test cases CSV")
        
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
        
        # Process each transform separately
        for transform_id, (transform_key, ranges) in enumerate(boundary_results.items(), 1):
            # Prepare test case data for this transform
            transform_test_cases = []
            
            for range_id, (range_key, stats) in enumerate(ranges.items(), 1):
                # Get transform names if possible
                transform1_name = transform_names.get(stats['transform1'], stats['transform1'])
                transform2_val = stats.get('transform2')
                transform2_name = transform_names.get(transform2_val, transform2_val) if transform2_val else 'N/A'
                
                # Create base test case
                base_test_case = {
                    "test_id": f"TC_BVA_{transform_id}_{range_id}",
                    "transform1": stats['transform1'],
                    "transform1_name": transform1_name,
                    "transform2": stats.get('transform2', 'N/A'),
                    "transform2_name": transform2_name,
                    "range1": stats['range1'],
                    "range2": stats.get('range2', 'N/A'),
                    "total_boundary_points": stats['total_points'],
                    "all_points_pass": stats['all_pass'],
                    "threshold": self.threshold,
                    "total_test_count": total_test_count
                }
                
                if class_name:
                    base_test_case["class_name"] = class_name
                
                # Add boundary points details
                boundary_points = stats['boundary_points']
                if transform2_val is None:
                    # 1D case (4 points)
                    for i, (point_val, point_info) in enumerate(boundary_points.items(), 1):
                        point_test_case = base_test_case.copy()
                        point_test_case["point_id"] = i
                        point_test_case["boundary_value"] = point_val
                        point_test_case["boundary_value2"] = "N/A"
                        point_test_case[f"mean_{self.metric_type}"] = point_info[f"mean_{self.metric_type}"]
                        point_test_case["point_pass"] = point_info['pass']
                        point_test_case["sample_size"] = point_info['sample_size']
                        transform_test_cases.append(point_test_case)
                else:
                    # 2D case (8 points)
                    for i, ((point_val1, point_val2), point_info) in enumerate(boundary_points.items(), 1):
                        point_test_case = base_test_case.copy()
                        point_test_case["point_id"] = i
                        point_test_case["boundary_value"] = point_val1
                        point_test_case["boundary_value2"] = point_val2
                        point_test_case[f"mean_{self.metric_type}"] = point_info[f"mean_{self.metric_type}"]
                        point_test_case["point_pass"] = point_info['pass']
                        point_test_case["sample_size"] = point_info['sample_size']
                        transform_test_cases.append(point_test_case)
            
            # Create DataFrame and save CSV for this transform
            if transform_test_cases:
                test_case_df = pd.DataFrame(transform_test_cases)
                file_suffix = f"_{class_name}" if class_name else ""
                output_file = os.path.join(output_dir, f"boundary_test_cases_{transform_key}{file_suffix}.csv")
                test_case_df.to_csv(output_file, index=False)
                logging.info(f"Generated {len(transform_test_cases)} boundary test cases for {transform_key}")
                logging.info(f"Test cases saved to {output_file}")
        
        # Create summary of passing ranges for each transform
        for transform_key, ranges in boundary_results.items():
            # Prepare passing ranges for this transform
            transform_passing_ranges = []
            
            # Print passing ranges to console/logger
            logging.info(f"\n=== PASSING BOUNDARY RANGES FOR {transform_key} ===")
            
            for range_key, stats in ranges.items():
                if stats['all_pass']:
                    # Get transform names
                    transform1_name = transform_names.get(stats['transform1'], stats['transform1'])
                    transform2_val = stats.get('transform2')
                    transform2_name = transform_names.get(transform2_val, transform2_val) if transform2_val else 'N/A'
                    
                    # Log passing range
                    if transform2_val:
                        log_msg = f"PASS: {transform1_name} {stats['range1']} + {transform2_name} {stats.get('range2', 'N/A')} -> All {stats['total_points']} boundary points pass"
                    else:
                        log_msg = f"PASS: {transform1_name} {stats['range1']} -> All {stats['total_points']} boundary points pass"
                    
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
                        "total_boundary_points": stats['total_points'],
                        "boundary_points_pass": stats['all_pass'],
                        "threshold": self.threshold
                    }
                    
                    if class_name:
                        passing_range["class_name"] = class_name
                        
                    transform_passing_ranges.append(passing_range)
            
            # Create DataFrame and save CSV for this transform's passing ranges
            if transform_passing_ranges:
                passing_df = pd.DataFrame(transform_passing_ranges)
                file_suffix = f"_{class_name}" if class_name else ""
                passing_file = os.path.join(output_dir, f"passing_boundary_ranges_{transform_key}{file_suffix}.csv")
                passing_df.to_csv(passing_file, index=False)
                logging.info(f"Generated summary of {len(transform_passing_ranges)} passing boundary ranges for {transform_key}")
                logging.info(f"Passing ranges saved to {passing_file}")
            
            logging.info("===========================")
        
        return boundary_results


def parse_args():
    parser = argparse.ArgumentParser(description='Boundary Transform Visualization and Test Case Generator')
    
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
    
    parser.add_argument('--output-dir', default='boundary_test_results',
                       help='Output directory for visualizations and test cases')
    
    parser.add_argument('--class-name', type=str, default='',
                       help='Class name to append to output files')
                       
    parser.add_argument('--range-file', type=str, default='',
                       help='Path to a file containing custom range sets')
                       
    parser.add_argument('--separate-ranges', action='store_true',
                       help='Plot each range in a separate graph instead of combining them')
                       
    parser.add_argument('--show-metrics', action='store_true',
                       help='Show metric values as annotations on the plot points')
                       
    return parser.parse_args()


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
            
        logging.info(f"Starting boundary visualization process for {metric} with threshold {threshold}")
        if class_name:
            logging.info(f"Class name: {class_name}")
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # OPTIONAL: Parse a range file if specified
        custom_range_sets = []
        if args.range_file:
            logging.info(f"Reading custom ranges from {args.range_file}")
            custom_range_sets = parse_range_file(args.range_file)
            logging.info(f"Found {len(custom_range_sets)} lines of ranges in the file")
        
        # Initialize visualizer
        visualizer = BoundaryTransformVisualizer(metric_type=metric, threshold=threshold)
        
        # Store custom ranges in visualizer if provided
        if custom_range_sets:
            visualizer.custom_range_sets = custom_range_sets

        # Read data
        df = visualizer.read_data(args.csv_file)
        
        # Get total test count
        total_test_count = len(df)
        
        # Process boundary data
        boundary_results = visualizer.process_boundary_data(df)
        
        # Create boundary visualizations with metrics turned off
        visualizer.create_boundary_visualizations(boundary_results, args.output_dir, class_name, 
                                               show_metrics=args.show_metrics, 
                                               separate_ranges=args.separate_ranges)
        
        # Generate boundary test cases
        visualizer.generate_test_cases(boundary_results, args.output_dir, total_test_count, class_name)
        
        logging.info("Boundary visualization process completed successfully")
        
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise


if __name__ == '__main__':
    main()
