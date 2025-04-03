import argparse
import cv2
import numpy as np
import os
import csv
import random
from pathlib import Path
from itertools import product
from scipy.ndimage import gaussian_filter, map_coordinates

# Map integer codes to transform names
TRANSFORMS = {
    0: 'gaussian_blur',
    1: 'rotation',
    2: 'brightness',
    3: 'translation',
    4: 'elastic_distortion'
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--transform1', type=int, choices=[0, 1, 2, 3, 4], required=True,
                        help='First transform code (0..4)')
    parser.add_argument('--transform2', type=int, choices=[0, 1, 2, 3, 4], required=True,
                        help='Second transform code (0..4)')
    parser.add_argument('--mode', type=str, choices=['ECP', 'BVA'], required=True,
                        help='Test mode: ECP (random) or BVA (boundary analysis)')
    parser.add_argument('--submode', type=str, choices=['CS1', 'CS2'], required=True,
                        help='Submode: CS1 (apply to all images) or CS2 (apply to single image)')
    parser.add_argument('--ranges', type=str, required=True,
                        help='Path to ranges.txt with exactly two lines (one for each transform).')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to the directory containing images')
    return parser.parse_args()


def read_two_lists(range_file):
    """
    Expects 'range_file' to have exactly two lines:
      Line 1: (min1,max1),(min2,max2),...  -> for Transform 1
      Line 2: (min1,max1),(min2,max2),...  -> for Transform 2

    We parse each line into a list of (min, max) pairs, then
    produce the cartesian product of those two lists.
    """
    with open(range_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) < 2:
        raise ValueError("ranges file must have at least two non-empty lines.")

    # Parse line 1 => list of (min, max) for Transform 1
    t1_list = parse_line_of_pairs(lines[0])
    # Parse line 2 => list of (min, max) for Transform 2
    t2_list = parse_line_of_pairs(lines[1])

    # Cartesian product to get every combination of transform1 range x transform2 range
    combined_ranges = list(product(t1_list, t2_list))
    return combined_ranges


def parse_line_of_pairs(line):
    """
    Given a line like: (0,10),(5,15),(10,20)
    Return a list of (min, max) tuples: [(0,10), (5,15), (10,20)]
    """
    pairs = []
    # Split by '),' - one approach. We remove leftover parentheses or whitespace
    segments = line.split('),')
    for seg in segments:
        seg = seg.replace('(', '').replace(')', '').strip()
        if not seg:
            continue
        min_str, max_str = seg.split(',')
        pairs.append((float(min_str), float(max_str)))
    return pairs


class TwoTransformProcessor:
    def __init__(self, transform1, transform2, mode, submode, ranges):
        """
        transform1, transform2: integer codes (0..4)
        mode: 'ECP' or 'BVA'
        submode: 'CS1' or 'CS2'
        ranges: list of [((min1, max1), (min2, max2)), ...] from read_two_lists()
        """
        self.transform1_type = TRANSFORMS[transform1]
        self.transform2_type = TRANSFORMS[transform2]
        self.mode = mode
        self.submode = submode
        self.ranges = ranges  # e.g., [(((0,10),(5,15))), (((0,10),(10,20))), ...]

        # Create output directory name
        self.output_dir = os.path.abspath(
            f"output/{self.transform1_type}_{self.transform2_type}_{self.mode}_{self.submode}"
        )
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Create CSV file path
        self.csv_file = os.path.join(
            self.output_dir,
            f"{self.transform1_type}_{self.transform2_type}_{self.mode}_{self.submode}.csv"
        )
        self.setup_csv()

        # Seed random for reproducibility
        random.seed(42)

        # Validate modes
        if self.mode not in ['ECP', 'BVA']:
            raise ValueError("Mode must be ECP or BVA.")
        if self.submode not in ['CS1', 'CS2']:
            raise ValueError("Submode must be CS1 or CS2.")

    def setup_csv(self):
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'image_name',
                'original_path',
                'transform1', 'range1', 'applied_value1',
                'transform2', 'range2', 'applied_value2',
                'output_path'
            ])

    def process_images(self, input_dir):
        input_dir = Path(input_dir).absolute()
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        # Gather valid image files
        all_files = list(input_dir.rglob('*'))
        image_paths = [p for p in all_files if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]

        if len(image_paths) == 0:
            print(f"No images found in {input_dir}")
            return

        if self.mode == 'ECP':
            # ECP => random picking
            if self.submode == 'CS1':
                self._ecp_cs1(image_paths)
            else:
                self._ecp_cs2(image_paths)
        else:
            # BVA => boundary values
            if self.submode == 'CS1':
                self._bva_cs1(image_paths)
            else:
                self._bva_cs2(image_paths)

    # -------------------
    # ECP + CS1
    # -------------------
    def _ecp_cs1(self, image_paths):
        """
        ECP + CS1:
          For each combination of ranges, pick random values for both transforms,
          then apply them to *every* image in the input folder.
        """
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for (range1, range2) in self.ranges:
                min1, max1 = range1
                min2, max2 = range2
                for img_path in image_paths:
                    image = cv2.imread(str(img_path))
                    if image is None:
                        continue

                    val1 = random.uniform(min1, max1)
                    val2 = random.uniform(min2, max2)

                    transformed = self.apply_two_transforms(image, val1, val2)

                    # Output name with no parentheses
                    out_name = f"{min1}_{max1}_{min2}_{max2}_{img_path.name}"
                    out_path = os.path.join(self.output_dir, out_name)
                    cv2.imwrite(out_path, transformed)

                    writer.writerow([
                        img_path.name,
                        str(img_path.absolute()),
                        self.transform1_type, f'({min1},{max1})', val1,
                        self.transform2_type, f'({min2},{max2})', val2,
                        os.path.abspath(out_path)
                    ])

    # -------------------
    # ECP + CS2
    # -------------------
    def _ecp_cs2(self, image_paths):
        """
        ECP + CS2:
          For each combination of ranges, pick ONE image from the folder (randomly),
          pick random values for the transforms, apply, and move on.
        """
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for (range1, range2) in self.ranges:
                min1, max1 = range1
                min2, max2 = range2

                # Pick one random image
                img_path = random.choice(image_paths)
                image = cv2.imread(str(img_path))
                if image is None:
                    continue

                val1 = random.uniform(min1, max1)
                val2 = random.uniform(min2, max2)

                transformed = self.apply_two_transforms(image, val1, val2)

                out_name = f"{min1}_{max1}_{min2}_{max2}_{img_path.name}"
                out_path = os.path.join(self.output_dir, out_name)
                cv2.imwrite(out_path, transformed)

                writer.writerow([
                    img_path.name,
                    str(img_path.absolute()),
                    self.transform1_type, f'({min1},{max1})', val1,
                    self.transform2_type, f'({min2},{max2})', val2,
                    os.path.abspath(out_path)
                ])

    # -------------------
    # BVA + CS1
    # -------------------
    def _bva_cs1(self, image_paths):
        """
        BVA + CS1 (Round-Robin Version):
        - For each (range1, range2) in self.ranges,
        - compute the 8 BVA pairs (val1, val2).
        - Then iterate over all images in image_paths, applying exactly one BVA pair
            in a cycle:
            Image #1 => bva_pairs[0]
            Image #2 => bva_pairs[1]
            ...
            Image #8 => bva_pairs[7]
            Image #9 => bva_pairs[0] (looping back)
            and so on.
        - This way, each image gets exactly ONE transform combination, rather than
            all images getting all 8 pairs.
        """
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            
            for (range1, range2) in self.ranges:
                bva_pairs = self._get_bva_pairs(range1, range2)  # e.g., 8 boundary combos
                num_pairs = len(bva_pairs)

                (min1, max1) = range1
                (min2, max2) = range2

                # Round-robin cycle across the images
                for i, img_path in enumerate(image_paths):
                    (val1, val2) = bva_pairs[i % num_pairs]

                    image = cv2.imread(str(img_path))
                    if image is None:
                        continue

                    # Apply the two transforms in sequence
                    transformed = self.apply_two_transforms(image, val1, val2)

                    # Build output filename (no parentheses)
                    out_name = f"{min1}_{max1}_{min2}_{max2}_{img_path.name}"
                    out_path = os.path.join(self.output_dir, out_name)
                    cv2.imwrite(out_path, transformed)

                    # Log to CSV
                    writer.writerow([
                        img_path.name,
                        str(img_path.absolute()),
                        self.transform1_type, f'({min1},{max1})', val1,
                        self.transform2_type, f'({min2},{max2})', val2,
                        os.path.abspath(out_path)
                    ])

    # -------------------
    # BVA + CS2
    # -------------------
    def _bva_cs2(self, image_paths):
        """
        BVA + CS2:
          For each combination of ranges, compute boundary-value pairs (8 combos).
          For each combo, transform exactly ONE image (randomly) from the folder.
        """
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for (range1, range2) in self.ranges:
                bva_pairs = self._get_bva_pairs(range1, range2)
                for (val1, val2) in bva_pairs:
                    img_path = random.choice(image_paths)
                    image = cv2.imread(str(img_path))
                    if image is None:
                        continue

                    transformed = self.apply_two_transforms(image, val1, val2)

                    (min1, max1) = range1
                    (min2, max2) = range2
                    out_name = f"{min1}_{max1}_{min2}_{max2}_{img_path.name}"
                    out_path = os.path.join(self.output_dir, out_name)
                    cv2.imwrite(out_path, transformed)

                    writer.writerow([
                        img_path.name,
                        str(img_path.absolute()),
                        self.transform1_type, f'({min1},{max1})', val1,
                        self.transform2_type, f'({min2},{max2})', val2,
                        os.path.abspath(out_path)
                    ])

    # -------------------
    # Helper to compute BVA pairs
    # -------------------
    def _get_bva_pairs(self, range1, range2):
        """
        Example BVA logic returning 8 combos:
          1) (min1, midpoint2)
          2) (min1+offset, midpoint2)
          3) (max1-offset, midpoint2)
          4) (max1, midpoint2)
          5) (midpoint1, min2)
          6) (midpoint1, min2+offset)
          7) (midpoint1, max2-offset)
          8) (midpoint1, max2)
        """
        (min1, max1) = range1
        (min2, max2) = range2

        midpoint1 = (min1 + max1) / 2.0
        midpoint2 = (min2 + max2) / 2.0

        offset1 = 0.1 * (max1 - min1)  # 10% of range1
        offset2 = 0.1 * (max2 - min2)  # 10% of range2

        bva_values = [
            (min1, midpoint2),
            (min1 + offset1, midpoint2),
            (max1 - offset1, midpoint2),
            (max1, midpoint2),
            (midpoint1, min2),
            (midpoint1, min2 + offset2),
            (midpoint1, max2 - offset2),
            (midpoint1, max2)
        ]
        return bva_values

    # -------------------
    # Apply two transforms in sequence
    # -------------------
    def apply_two_transforms(self, image, value1, value2):
        """
        Apply transform1 with 'value1', then transform2 with 'value2' on the result.
        """
        result1 = self.apply_transform(image, self.transform1_type, value1)
        result2 = self.apply_transform(result1, self.transform2_type, value2)
        return result2

    def apply_transform(self, image, transform_type, value):
        """
        Applies the specified transform to 'image' using 'value'.
        """
        if transform_type == 'gaussian_blur':
            # If sigma <= 0, treat it as "no blur" and just return the original image
            if value <= 0:
                return image
            
            # Create kernel size that scales with sigma
            kernel_size = int(max(3, 6 * value + 1))
            if kernel_size % 2 == 0:  # Make sure it's odd
                kernel_size += 1
            return cv2.GaussianBlur(image, (0, 0), value)

        elif transform_type == 'rotation':
            rows, cols = image.shape[:2]
            # value is angle
            matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), value, 1)
            return cv2.warpAffine(image, matrix, (cols, rows))

        elif transform_type == 'brightness':
            # value is additive brightness (beta)
            return cv2.convertScaleAbs(image, alpha=1, beta=value)

        elif transform_type == 'translation':
            # value is offset for x and y
            rows, cols = image.shape[:2]
            matrix = np.float32([[1, 0, value], [0, 1, value]])
            return cv2.warpAffine(image, matrix, (cols, rows))

        elif transform_type == 'elastic_distortion':
            return self._apply_elastic_transform(image, value)

    def _apply_elastic_transform(self, image, alpha):
        """
        Applies elastic distortion with intensity 'alpha'.
        """
        shape_2d = image.shape[:2]
        dx = gaussian_filter((np.random.rand(*shape_2d) * 2 - 1), alpha) * alpha
        dy = gaussian_filter((np.random.rand(*shape_2d) * 2 - 1), alpha) * alpha

        x, y = np.meshgrid(np.arange(shape_2d[1]), np.arange(shape_2d[0]))
        indices = (
            np.reshape(y + dy, (-1, 1)),
            np.reshape(x + dx, (-1, 1))
        )

        # If color image, warp each channel
        if len(image.shape) == 3:
            warped_channels = []
            for c in range(image.shape[2]):
                warped_c = map_coordinates(
                    image[..., c], indices, order=1, mode='reflect'
                ).reshape(shape_2d)
                warped_channels.append(warped_c)
            return np.dstack(warped_channels)
        else:
            warped = map_coordinates(
                image, indices, order=1, mode='reflect'
            ).reshape(shape_2d)
            return warped


def main():
    args = parse_args()
    # Read the two lines of (min, max) pairs from the file
    combined_ranges = read_two_lists(args.ranges)

    # Create processor
    processor = TwoTransformProcessor(
        transform1=args.transform1,
        transform2=args.transform2,
        mode=args.mode,
        submode=args.submode,
        ranges=combined_ranges
    )

    # Process images in the given directory
    processor.process_images(args.input_dir)


if __name__ == '__main__':
    main()
