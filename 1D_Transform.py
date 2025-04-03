import argparse
import cv2
import numpy as np
import os
import csv
import random
import time
from pathlib import Path
from scipy.ndimage import gaussian_filter, map_coordinates

TRANSFORMS = {
    0: 'gaussian_blur',
    1: 'rotation',
    2: 'brightness',
    3: 'translation',
    4: 'elastic_distortion'
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--transform', type=int, choices=[0, 1, 2, 3, 4], required=True,
                        help='0: Gaussian Blur, 1: Rotation, 2: Brightness, 3: Translation, 4: Elastic Distortion')
    parser.add_argument('--mode', type=str, choices=['ECP','BVA'], required=True,
                        help='ECP = random within range, BVA = boundary value analysis')
    parser.add_argument('--submode', type=str, choices=['CS1','CS2'], required=True,
                        help='CS1 = apply to all images, CS2 = apply to one image per range/boundary')
    parser.add_argument('--boundary_offset', type=int, default=1,
                        help='Number of decimal places from boundary (e.g., 1 => 0.1, 2 => 0.01). Used for BVA.')
    parser.add_argument('--ranges', type=str, required=True,
                        help='Path to ranges.txt where each line is like (min_val,max_val)')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing images')
    return parser.parse_args()

def read_ranges(range_file):
    """
    Reads each line of 'ranges.txt' as '(min_val,max_val)' and returns a list of (min, max) floats.
    """
    if not os.path.exists(range_file):
        raise FileNotFoundError(f"Range file not found: {range_file}")
        
    ranges = []
    with open(range_file) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip().replace('(', '').replace(')', '')
            if not line:
                continue
            parts = line.split(',')
            if len(parts) != 2:
                raise ValueError(f"Line {line_num}: Invalid format. Expected '(min_val,max_val)', got '{line}'")
            min_val, max_val = map(float, parts)
            if min_val >= max_val:
                raise ValueError(f"Line {line_num}: Min value ({min_val}) must be less than max value ({max_val})")
            ranges.append((min_val, max_val))
    return ranges

class SingleTransformProcessor:
    def __init__(self, transform_num, mode, submode, ranges, boundary_offset):
        """
        transform_num: which transform (0..4)
        mode: 'ECP' or 'BVA'
        submode: 'CS1' or 'CS2'
        ranges: list of (min_val, max_val)
        boundary_offset: integer => offset = 10^(-boundary_offset), e.g. 0.1, 0.01, etc.
        """
        self.transform_type = TRANSFORMS[transform_num]
        self.mode = mode
        self.submode = submode
        self.ranges = ranges
        self.offset_value = 10 ** (-boundary_offset)  # e.g., 0.1 if boundary_offset=1

        # Create output directory:
        # e.g. "output/rotation_ECP_CS1"
        self.transform_dir = os.path.abspath(
            f"output/{self.transform_type}_{self.mode}_{self.submode}"
        )
        Path(self.transform_dir).mkdir(parents=True, exist_ok=True)

        # Build CSV file path, e.g. "output/rotation_ECP_CS1/transforms_rotation_ECP_CS1.csv"
        csv_name = f"transforms_{self.transform_type}_{self.mode}_{self.submode}.csv"
        self.csv_file = os.path.join(self.transform_dir, csv_name)

        # Prepare CSV
        self.setup_csv()

        # Fix random seed for reproducibility
        random.seed(42)

    def setup_csv(self):
        # If the CSV already exists, rename it with a timestamp so we don’t overwrite
        if os.path.exists(self.csv_file):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            base, ext = os.path.splitext(self.csv_file)
            new_csv_file = f"{base}_{timestamp}{ext}"
            print(f"CSV file exists. Creating new file: {new_csv_file}")
            self.csv_file = new_csv_file

        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Changed column name from 'boundary_value' to 'applied_value'
            writer.writerow(['image_name', 'original_path', 'transform', 'range', 'applied_value', 'output_path'])

    def process_images(self, input_dir):
        """
        This orchestrates the ECP/BVA + CS1/CS2 logic for a single transform.
        """
        input_dir = Path(input_dir)
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        # Gather valid images
        all_files = list(input_dir.rglob('*'))
        image_files = [f for f in all_files if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
        if not image_files:
            raise ValueError(f"No valid images found in {input_dir}")

        # Route to the correct method
        if self.mode == 'ECP':
            if self.submode == 'CS1':
                self._ecp_cs1(image_files)
            else:  # CS2
                self._ecp_cs2(image_files)
        else:  # BVA
            if self.submode == 'CS1':
                self._bva_cs1(image_files)
            else:  # CS2
                self._bva_cs2(image_files)

    # -----------------------------
    # ECP + CS1
    # -----------------------------
    def _ecp_cs1(self, image_files):
        """
        ECP + CS1:
          - For each (min_val, max_val) range,
          - pick a random value within [min_val, max_val] for EACH IMAGE,
          - transform that image, save + log to CSV.
        """
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for (range_min, range_max) in self.ranges:
                # Make a subdirectory for this range (similar to original code)
                range_dir = os.path.join(self.transform_dir, f"range_{range_min}_{range_max}")
                Path(range_dir).mkdir(exist_ok=True)

                for img_path in image_files:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue

                    # ECP => random
                    val = random.uniform(range_min, range_max)

                    transformed = self.apply_transform(img, val)

                    # Output filename
                    out_name = f"{range_min}_{range_max}_{img_path.name}"
                    out_path = os.path.join(range_dir, out_name)
                    out_path = self.check_output_path(out_path)  # handle duplicates

                    cv2.imwrite(out_path, transformed)

                    writer.writerow([
                        img_path.name,
                        str(img_path.resolve()),
                        self.transform_type,
                        f'({range_min},{range_max})',
                        val,  # now labeled 'applied_value' in the CSV
                        out_path
                    ])

    # -----------------------------
    # ECP + CS2
    # -----------------------------
    def _ecp_cs2(self, image_files):
        """
        ECP + CS2:
          - For each (min_val, max_val) range,
          - pick ONE image (random) from the folder,
          - pick a random value in [min_val, max_val],
          - transform + save + log.
        """
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for (range_min, range_max) in self.ranges:
                range_dir = os.path.join(self.transform_dir, f"range_{range_min}_{range_max}")
                Path(range_dir).mkdir(exist_ok=True)

                # Pick one random image
                img_path = random.choice(image_files)
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                val = random.uniform(range_min, range_max)

                transformed = self.apply_transform(img, val)
                out_name = f"{range_min}_{range_max}_{img_path.name}"
                out_path = os.path.join(range_dir, out_name)
                out_path = self.check_output_path(out_path)

                cv2.imwrite(out_path, transformed)

                writer.writerow([
                    img_path.name,
                    str(img_path.resolve()),
                    self.transform_type,
                    f'({range_min},{range_max})',
                    val,  # 'applied_value'
                    out_path
                ])

    # -----------------------------
    # BVA + CS1
    # -----------------------------
    def _bva_cs1(self, image_files):
        """
        BVA + CS1 (Cycling version):
        - For each (min_val, max_val) range,
        - compute boundary points (e.g., 4 points: min, min+offset, max-offset, max),
        - cycle through those boundary points in a round-robin fashion as you iterate over the dataset.
        
        Example pattern for 10 images with 4 boundary points:
            Image #1 => boundary point 0
            Image #2 => boundary point 1
            Image #3 => boundary point 2
            Image #4 => boundary point 3
            Image #5 => boundary point 0
            ...
        """
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for (range_min, range_max) in self.ranges:
                range_dir = os.path.join(self.transform_dir, f"range_{range_min}_{range_max}")
                Path(range_dir).mkdir(exist_ok=True)

                # Get the 4 (or however many) boundary points for this range
                boundary_points = self._get_boundary_points(range_min, range_max)
                num_points = len(boundary_points)

                # Go through each image, assigning boundary points in a round-robin sequence
                for i, img_path in enumerate(image_files):
                    bp = boundary_points[i % num_points]  # cycle boundary points

                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue

                    transformed = self.apply_transform(img, bp)

                    # Build output filename & path
                    out_name = f"boundary_{bp}_{img_path.name}"
                    out_path = os.path.join(range_dir, out_name)
                    out_path = self.check_output_path(out_path)

                    cv2.imwrite(out_path, transformed)

                    # Log to CSV
                    writer.writerow([
                        img_path.name,
                        str(img_path.resolve()),
                        self.transform_type,
                        f'({range_min},{range_max})',
                        bp,  # 'applied_value'
                        out_path
                    ])

    # -----------------------------
    # BVA + CS2
    # -----------------------------
    def _bva_cs2(self, image_files):
        """
        BVA + CS2:
          - For each (min_val, max_val) range,
          - compute boundary points (4 points),
          - for each point, pick ONE random image,
          - transform + save + log.
        """
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for (range_min, range_max) in self.ranges:
                range_dir = os.path.join(self.transform_dir, f"range_{range_min}_{range_max}")
                Path(range_dir).mkdir(exist_ok=True)

                boundary_points = self._get_boundary_points(range_min, range_max)

                for bp in boundary_points:
                    img_path = random.choice(image_files)
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue

                    transformed = self.apply_transform(img, bp)
                    out_name = f"boundary_{bp}_{img_path.name}"
                    out_path = os.path.join(range_dir, out_name)
                    out_path = self.check_output_path(out_path)

                    cv2.imwrite(out_path, transformed)

                    writer.writerow([
                        img_path.name,
                        str(img_path.resolve()),
                        self.transform_type,
                        f'({range_min},{range_max})',
                        bp,  # 'applied_value'
                        out_path
                    ])

    # -----------------------------
    # Utility methods
    # -----------------------------
    def _get_boundary_points(self, range_min, range_max):
        """
        Return 4 boundary points: 
          1) range_min
          2) range_min + offset_value
          3) range_max - offset_value
          4) range_max
        """
        return [
            range_min,
            range_min + self.offset_value,
            range_max - self.offset_value,
            range_max
        ]

    def check_output_path(self, output_path):
        """
        If output file exists, rename it with a timestamp 
        to avoid overwriting.
        """
        if os.path.exists(output_path):
            base, ext = os.path.splitext(output_path)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            new_path = f"{base}_{timestamp}{ext}"
            print(f"Output file '{output_path}' exists. Renaming to: {new_path}")
            return new_path
        return output_path

    def apply_transform(self, image, value):
        """
        Apply the selected single transform with 'value'.
        (Matches your original transformations.)
        """
        if self.transform_type == 'gaussian_blur':
        # Ensure kernel size is valid (odd and positive)
            kernel_size = int(max(3, 6 * value + 1))  # This scales with sigma
            if kernel_size % 2 == 0:  # Make sure it's odd
                kernel_size += 1
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), value)

        elif self.transform_type == 'rotation':
            rows, cols = image.shape[:2]
            matrix = cv2.getRotationMatrix2D((cols/2, rows/2), value, 1)
            return cv2.warpAffine(image, matrix, (cols, rows))

        elif self.transform_type == 'brightness':
            # 'value' is an additive brightness factor
            return cv2.convertScaleAbs(image, alpha=1, beta=value)

        elif self.transform_type == 'translation':
            rows, cols = image.shape[:2]
            matrix = np.float32([[1, 0, value], [0, 1, value]])
            return cv2.warpAffine(image, matrix, (cols, rows))

        elif self.transform_type == 'elastic_distortion':
            return self._apply_elastic_transform(image, value)

    def _apply_elastic_transform(self, image, alpha):
        """
        Elastic distortion using random displacement fields.
        """
        # Get the height and width (ignore channels if present):
        shape_2d = image.shape[:2]
        
        # Generate 2D random displacement fields
        dx = gaussian_filter((np.random.rand(*shape_2d) * 2 - 1), alpha) * alpha
        dy = gaussian_filter((np.random.rand(*shape_2d) * 2 - 1), alpha) * alpha

        # Build coordinate grids
        x, y = np.meshgrid(np.arange(shape_2d[1]), np.arange(shape_2d[0]))
        indices = (np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)))

        # If color image, warp each channel
        if len(image.shape) == 3:
            warped_channels = []
            for c in range(image.shape[2]):
                warped_c = map_coordinates(image[..., c], indices, order=1, mode='reflect')
                warped_c = warped_c.reshape(shape_2d)
                warped_channels.append(warped_c)
            return np.dstack(warped_channels)
        else:
            # Grayscale
            warped = map_coordinates(image, indices, order=1, mode='reflect')
            return warped.reshape(shape_2d)

def main():
    args = parse_args()

    # Read the single list of (min_val, max_val) from ranges.txt
    ranges_list = read_ranges(args.ranges)

    # Instantiate the processor with the chosen transform and mode
    processor = SingleTransformProcessor(
        transform_num=args.transform,
        mode=args.mode,
        submode=args.submode,
        ranges=ranges_list,
        boundary_offset=args.boundary_offset
    )

    # Process images in the input directory
    try:
        processor.process_images(args.input_dir)
        print("\nTransformation process completed successfully!")
    except Exception as e:
        print(f"\nError: {e}")
        raise

if __name__ == '__main__':
    main()
