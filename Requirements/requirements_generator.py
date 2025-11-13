import argparse
import os
import itertools
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
    parser.add_argument('--transforms', type=str, required=True, nargs='+', 
                        choices=list(TRANSFORMS.keys()), 
                        help='Transforms to use (1 or 2)')
    parser.add_argument('--output', type=str, default='requirements.txt', help='Output file path')
    args = parser.parse_args()
    
    # Validate transform count
    if len(args.transforms) > 2 or len(args.transforms) == 0:
        raise ValueError("Only 1 or 2 transforms can be specified")
    
    return args

def read_ranges(range_file, transforms):
    """Read ranges from the file for the specified transforms.
    
    Format expected: Each line contains ranges for one transform,
    with ranges in the format (min,max),(min,max),...
    """
    ranges_by_transform = {}
    
    with open(range_file, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    # Validate that we have enough lines for the transforms
    if len(transforms) > len(lines):
        raise ValueError(f"Not enough lines in file for transforms. Need {len(transforms)}, found {len(lines)}")
    
    # Process each transform with its corresponding line
    for i, transform in enumerate(transforms):
        if i >= len(lines):
            break
            
        # Extract all (min,max) pairs from the line
        pattern = r'\(([^)]+)\)'
        matches = re.findall(pattern, lines[i])
        
        transform_ranges = []
        for match in matches:
            try:
                min_val, max_val = map(float, match.split(','))
                transform_ranges.append((min_val, max_val))
            except ValueError:
                print(f"Warning: Could not parse range '{match}' for {transform}")
                continue
        
        ranges_by_transform[transform] = transform_ranges
    
    return ranges_by_transform

def generate_single_requirement(model_name, class_name, transform, min_val, max_val):
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

def generate_double_requirement(model_name, class_name, transform1, min1, max1, transform2, min2, max2):
    """Generate a requirement for two transforms"""
    unit1 = TRANSFORMS[transform1]
    unit2 = TRANSFORMS[transform2]
    
    # Fix articles for words starting with vowels
    article1 = "an" if transform1.replace('_', ' ')[0].lower() in "aeiou" else "a"
    article2 = "an" if transform2.replace('_', ' ')[0].lower() in "aeiou" else "a"
    
    # Also check for articles before units
    unit1_article = "an" if unit1[0].lower() in "aeiou" else "a"
    unit2_article = "an" if unit2[0].lower() in "aeiou" else "a"
    
    # Fix wording for better grammar
    if transform1 == "rotation" or transform1 == "translation":
        phrase1 = f"{transform1} of {min1} to {max1} {unit1}"
    else:
        phrase1 = f"{transform1.replace('_', ' ')} with {unit1_article} {unit1} between {min1} to {max1}"
    
    if transform2 == "rotation" or transform2 == "translation":
        phrase2 = f"{transform2} of {min2} to {max2} {unit2}"
    else:
        phrase2 = f"{transform2.replace('_', ' ')} with {unit2_article} {unit2} between {min2} to {max2}"
    
    return f"The {model_name} shall detect a \"{class_name}\" class under {article1} {phrase1} and {article2} {phrase2}."

def generate_requirements(model_name, class_name, transforms, ranges_by_transform):
    """Generate all requirements based on transforms and ranges"""
    requirements = []
    
    if len(transforms) == 1:
        # Single transform requirement
        transform = transforms[0]
        for min_val, max_val in ranges_by_transform[transform]:
            req = generate_single_requirement(model_name, class_name, transform, min_val, max_val)
            requirements.append(req)
    else:
        # Two transform requirements
        transform1, transform2 = transforms
        
        # Create all combinations of ranges for both transforms
        for (min1, max1), (min2, max2) in itertools.product(
            ranges_by_transform[transform1], 
            ranges_by_transform[transform2]
        ):
            req = generate_double_requirement(
                model_name, class_name, 
                transform1, min1, max1, 
                transform2, min2, max2
            )
            requirements.append(req)
    
    return requirements

def main():
    args = parse_args()
    
    # Read ranges for each transform
    ranges_by_transform = read_ranges(args.ranges_file, args.transforms)
    
    # Check if we got any valid ranges
    for transform in args.transforms:
        if transform not in ranges_by_transform or not ranges_by_transform[transform]:
            print(f"No valid ranges found for transform: {transform}")
            return
    
    # Generate requirements
    requirements = generate_requirements(args.model_name, args.class_name, args.transforms, ranges_by_transform)
    
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
