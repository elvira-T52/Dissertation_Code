import argparse
import os
import re
from pathlib import Path

# Define transform types and their units
TRANSFORMS = {
    'gaussian_blur': 'standard deviation',
    'rotation': 'degrees',
    'brightness': 'brightness factor',
    'translation': 'pixels',
    'elastic_distortion': 'alpha'
}

def parse_args():
    parser = argparse.ArgumentParser(description='Generate requirements from transform ranges')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model (e.g., "object detector")')
    parser.add_argument('--class_name', type=str, required=True, help='Name of the class (e.g., "Person")')
    parser.add_argument('--ranges_file', type=str, required=True, help='Path to file containing ranges')
    parser.add_argument('--transform', type=str, required=True, 
                        choices=list(TRANSFORMS.keys()), 
                        help='Transform to use (only single transform supported)')
    parser.add_argument('--output', type=str, default='requirements.txt', help='Output file path')
    args = parser.parse_args()
    
    return args

def read_ranges(range_file):
    """Read ranges from the file, one range per line.
    
    Format expected: Each line contains one range in the format (min,max) or (min, max)
    """
    ranges = []
    
    with open(range_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            # Extract the range from the line
            pattern = r'\(([^)]+)\)'
            match = re.search(pattern, line)
            
            if not match:
                print(f"Warning: Could not parse line: {line}")
                continue
                
            try:
                # Handle both comma formats: (x,y) or (x, y)
                min_max = match.group(1).replace(' ', '')
                min_val, max_val = map(float, min_max.split(','))
                ranges.append((min_val, max_val))
            except ValueError:
                print(f"Warning: Could not parse range in line: {line}")
                continue
    
    return ranges

def generate_requirement(model_name, class_name, transform, min_val, max_val):
    """Generate a requirement for a single transform"""
    unit = TRANSFORMS[transform]
    
    # Fix article for words starting with vowels
    article = "an" if transform.replace('_', ' ')[0].lower() in "aeiou" else "a"
    
    # Check for article before unit
    unit_article = "an" if unit[0].lower() in "aeiou" else "a"
    
    # Better phrasing for certain transforms
    if transform == "rotation" or transform == "translation":
        return f"The {model_name} shall detect a \"{class_name}\" class under {article} {transform} of {min_val} to {max_val} {unit}."
    else:
        return f"The {model_name} shall detect a \"{class_name}\" class under {article} {transform.replace('_', ' ')} with {unit_article} {unit} between {min_val} to {max_val}."

def main():
    args = parse_args()
    
    # Read ranges from the file
    ranges = read_ranges(args.ranges_file)
    
    if not ranges:
        print(f"No valid ranges found in the file: {args.ranges_file}")
        return
    
    # Generate requirements
    requirements = []
    for min_val, max_val in ranges:
        req = generate_requirement(args.model_name, args.class_name, args.transform, min_val, max_val)
        requirements.append(req)
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_dir = output_path.parent
    if output_dir and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write requirements to output file
    with open(args.output, 'w') as f:
        for req in requirements:
            f.write(req + '\n')
    
    print(f"Generated {len(requirements)} requirements in {args.output}")
    
    # Display first few requirements as preview
    preview_count = min(5, len(requirements))
    if preview_count > 0:
        print("\nPreview of generated requirements:")
        for i in range(preview_count):
            print(f"{i+1}. {requirements[i]}")
        
        if len(requirements) > preview_count:
            print(f"... and {len(requirements) - preview_count} more.")

if __name__ == '__main__':
    main()
