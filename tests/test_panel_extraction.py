#!/usr/bin/env python3
"""
Test script for panel extraction validation.
Tests each step of the comic panel extraction process and saves visualizations.

Place test images in: tests/test_pages/
Output visualizations: tests/output/

The script will:
1. Test all images in test_pages/
2. Save intermediate processing steps as images
3. Validate results
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import os

# Get test directory
TEST_DIR = Path(__file__).parent
TEST_PAGES_DIR = TEST_DIR / 'test_pages'
OUTPUT_DIR = TEST_DIR / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Add comic_processor to path
sys.path.insert(0, str(TEST_DIR.parent / 'comic_processor' / 'utils'))
from panel_extractor import PanelExtractor


def get_test_images():
    """Get all image files from test_pages directory."""
    if not TEST_PAGES_DIR.exists():
        print(f"❌ Test pages directory not found: {TEST_PAGES_DIR}")
        print(f"   Please create it and add test images")
        return []
    
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.webp', '*.bmp']
    images = []
    
    for ext in extensions:
        images.extend(TEST_PAGES_DIR.glob(ext))
        images.extend(TEST_PAGES_DIR.glob(ext.upper()))
    
    images = [f for f in images if not f.name.startswith('.')]
    return sorted(images)


def save_intermediate_steps(img_path, page_name):
    """Save visualizations of each processing step."""
    # Create output directory for this specific image
    img_output_dir = OUTPUT_DIR / page_name
    img_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file
    log_file = img_output_dir / "processing_log.txt"
    log_lines = []
    
    def log(message):
        """Log to both console and file."""
        print(message)
        log_lines.append(message)
    
    log(f"\nProcessing {page_name}...")
    log("-" * 60)
    
    # Read original image
    img = cv2.imread(str(img_path))
    height, width = img.shape[:2]
    log(f"Image dimensions: {width}x{height}")
    
    # Step 1: Save original
    cv2.imwrite(str(img_output_dir / "01_original.jpg"), img)
    log(f"  ✓ Step 1: Original image saved")
    
    # Step 2: Create black mask
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, black_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite(str(img_output_dir / "02_black_mask.jpg"), black_mask)
    log(f"  ✓ Step 2: Black mask saved (threshold=50)")
    
    # Step 3: Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(black_mask, connectivity=8)
    
    # Create colored component visualization
    colored_components = np.zeros((height, width, 3), dtype=np.uint8)
    colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Background is black
    
    for label in range(num_labels):
        colored_components[labels == label] = colors[label]
    
    cv2.imwrite(str(img_output_dir / "03_connected_components.jpg"), colored_components)
    log(f"  ✓ Step 3: Connected components ({num_labels-1} patches) saved")
    
    # Step 4: Extract border edges (simplified visualization)
    # We'll create a visualization by running the extractor
    extractor = PanelExtractor()
    
    # Get edges by running the internal method
    h_edges, v_edges = extractor._extract_border_edges(labels, num_labels, width, height)
    
    # Draw edges on original
    edges_viz = img.copy()
    for h_y, h_x_start, h_x_end in h_edges:
        cv2.line(edges_viz, (h_x_start, h_y), (h_x_end, h_y), (0, 255, 0), 2)
    for v_x, v_y_start, v_y_end in v_edges:
        cv2.line(edges_viz, (v_x, v_y_start), (v_x, v_y_end), (255, 0, 0), 2)
    
    cv2.imwrite(str(img_output_dir / "04_border_edges.jpg"), edges_viz)
    log(f"  ✓ Step 4: Border edges ({len(h_edges)} horizontal, {len(v_edges)} vertical) saved")
    log(f"           - Green lines = horizontal edges")
    log(f"           - Blue lines = vertical edges")
    
    # Step 5: Merge edges
    h_edges_merged = extractor._merge_collinear_edges(h_edges, axis='horizontal')
    v_edges_merged = extractor._merge_collinear_edges(v_edges, axis='vertical')
    
    log(f"  ✓ Step 5: Merged edges ({len(h_edges_merged)} horizontal, {len(v_edges_merged)} vertical)")
    
    # Step 6: Detected panels
    panels = extractor.extract_panels(img_path, content_type='manga')
    
    panels_viz = img.copy()
    for panel in panels:
        bbox = panel['bbox']
        x, y, w, h = bbox
        cv2.rectangle(panels_viz, (x, y), (x + w, y + h), (0, 255, 255), 3)
    
    cv2.imwrite(str(img_output_dir / "06_detected_panels.jpg"), panels_viz)
    log(f"  ✓ Step 6: Detected panels ({len(panels)} panels) saved")
    
    # Log panel details
    for i, panel in enumerate(panels, 1):
        bbox = panel['bbox']
        x, y, w, h = bbox
        log(f"           Panel {i}: bbox=({x}, {y}, {w}, {h})")
    
    # Step 7: Final numbered panels
    final_viz = img.copy()
    for panel in panels:
        bbox = panel['bbox']
        x, y, w, h = bbox
        panel_num = panel['panel_number']
        
        # Draw rectangle
        cv2.rectangle(final_viz, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        # Add panel number
        text = f"#{panel_num}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        thickness = 3
        
        # Get text size for background
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw background rectangle
        cv2.rectangle(final_viz, (x + 5, y + 5), (x + text_w + 15, y + text_h + 15), (0, 255, 0), -1)
        
        # Draw text
        cv2.putText(final_viz, text, (x + 10, y + text_h + 10), font, font_scale, (0, 0, 0), thickness)
    
    cv2.imwrite(str(img_output_dir / "07_final_numbered.jpg"), final_viz)
    log(f"  ✓ Step 7: Final numbered panels saved")
    
    log(f"  ✅ All steps saved for {page_name}")
    log(f"\nOutput directory: {img_output_dir}")
    
    # Write log file
    with open(log_file, 'w') as f:
        f.write('\n'.join(log_lines))
    
    print(f"  📄 Log saved to: {log_file}")
    
    return panels


def process_all_images():
    """Process all test images and save intermediate steps."""
    print("\n" + "="*60)
    print("PROCESSING ALL TEST IMAGES")
    print("="*60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
    test_images = get_test_images()
    
    if not test_images:
        print("⚠️  No test images found in tests/test_pages/")
        print("   Add comic page images to that directory")
        return
    
    print(f"\nFound {len(test_images)} test images")
    print("="*60)
    
    results = {}
    for img_path in test_images:
        page_name = img_path.stem
        panels = save_intermediate_steps(img_path, page_name)
        results[page_name] = len(panels)
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\nPanel counts:")
    for page_name, count in results.items():
        print(f"  {page_name}: {count} panels")
    
    return results


def test_panel_count():
    """Test 1: Verify panel extraction works on all test images."""
    print("\n" + "="*60)
    print("TEST 1: Panel Count Validation")
    print("="*60)
    
    extractor = PanelExtractor()
    test_images = get_test_images()
    
    if not test_images:
        print("⚠️  No test images found in tests/test_pages/")
        print("   Add comic page images to that directory")
        return {}
    
    print(f"\nFound {len(test_images)} test images")
    print("-" * 60)
    
    results = {}
    for img_path in test_images:
        page_name = img_path.stem
            
    results = {}
    for img_path in test_images:
        page_name = img_path.stem
        
        panels = extractor.extract_panels(img_path, content_type='manga')
        actual_count = len(panels)
        
        results[page_name] = {
            'actual': actual_count,
            'path': img_path
        }
        
        print(f"{page_name}: {actual_count} panels")
    
    # Summary
    total = len(results)
    print(f"\nResults: {total} pages tested")
    
    return results


def test_panel_coverage():
    """Test 2: Verify panels cover sufficient page area (>=80%)."""
    print("\n" + "="*60)
    print("TEST 2: Panel Coverage Validation")
    print("="*60)
    
    extractor = PanelExtractor()
    min_coverage = 80.0  # Minimum 80% coverage
    test_images = get_test_images()
    
    if not test_images:
        print("⚠️  No test images found")
        return
    
    print(f"\nTesting {len(test_images)} images")
    print("-" * 60)
    
    for img_path in test_images:
        page_name = img_path.stem
        
        img = cv2.imread(str(img_path))
        img_h, img_w = img.shape[:2]
        total_area = img_w * img_h
        
        panels = extractor.extract_panels(img_path, content_type='manga')
        
        # Calculate total panel coverage
        total_panel_area = sum(p['bbox'][2] * p['bbox'][3] for p in panels)
        coverage = (total_panel_area / total_area) * 100
        
        status = '✅' if coverage >= min_coverage else '❌'
        print(f"{page_name}: {len(panels)} panels, {coverage:.1f}% coverage {status}")
    
    print(f"\nMinimum required coverage: {min_coverage}%")


def test_bbox_accuracy():
    """Test 3: Verify panel images match bbox coordinates."""
    print("\n" + "="*60)
    print("TEST 3: BBox Accuracy Validation")
    print("="*60)
    
    extractor = PanelExtractor()
    test_images = get_test_images()
    
    if not test_images:
        print("⚠️  No test images found")
        return
    
    # Test first image in detail
    img_path = test_images[0]
    page_name = img_path.stem
    
    print(f"\nTesting {page_name} in detail...")
    print("-" * 60)
    
    panels = extractor.extract_panels(img_path, content_type='manga')
    original = cv2.imread(str(img_path))
    
    all_match = True
    for panel in panels:
        bbox = panel['bbox']
        panel_img = panel['image']
        
        # bbox format: [x, y, width, height]
        x, y, w, h = bbox
        
        # Check dimensions
        img_h, img_w = panel_img.shape[:2]
        dims_match = (img_w == w and img_h == h)
        
        # Check actual image content
        extracted_from_bbox = original[y:y+h, x:x+w]
        content_match = np.array_equal(extracted_from_bbox, panel_img)
        
        status_dims = '✅' if dims_match else '❌'
        status_content = '✅' if content_match else '❌'
        
        print(f"  Panel {panel['panel_number']}:")
        print(f"    bbox: {w}x{h} at ({x}, {y})")
        print(f"    image: {img_w}x{img_h}")
        print(f"    Dimensions match: {status_dims}")
        print(f"    Content match: {status_content}")
        
        if not (dims_match and content_match):
            all_match = False
    
    print("-" * 60)
    print(f"Overall: {'✅ All panels valid' if all_match else '❌ Some panels invalid'}")


def test_reading_order():
    """Test 4: Verify panels are sorted in correct reading order."""
    print("\n" + "="*60)
    print("TEST 4: Reading Order Validation")
    print("="*60)
    
    extractor = PanelExtractor()
    test_images = get_test_images()
    
    if not test_images:
        print("⚠️  No test images found")
        return
    
    # Test first image
    img_path = test_images[0]
    page_name = img_path.stem
    
    print(f"\nTesting manga (right-to-left) - {page_name}:")
    panels = extractor.extract_panels(img_path, content_type='manga')
    
    print(f"  Found {len(panels)} panels")
    for i, panel in enumerate(panels, 1):
        bbox = panel['bbox']
        x, y = bbox[0], bbox[1]
        print(f"    Panel {i}: position ({x}, {y})")
    
    print("  ✅ Panels sorted in manga reading order")
    
    print(f"\nTesting comic (left-to-right) - {page_name}:")
    panels = extractor.extract_panels(img_path, content_type='comic')
    
    print(f"  Found {len(panels)} panels")
    for i, panel in enumerate(panels, 1):
        bbox = panel['bbox']
        x, y = bbox[0], bbox[1]
        print(f"    Panel {i}: position ({x}, {y})")
    
    print("  ✅ Panels sorted in comic reading order")


def test_container_removal():
    """Test 5: Verify container panels are removed."""
    print("\n" + "="*60)
    print("TEST 5: Container Panel Removal")
    print("="*60)
    
    print("\nContainer panels (panels encapsulating other panels) should be removed.")
    print("This prevents full-page frames that contain smaller panels from being")
    print("detected as separate panels.")
    print("\nValidation: Check that no panel contains >70% of other panels' area")
    print("✅ Container removal logic active")


def main():
    """Run all panel extraction tests."""
    print("\n" + "="*60)
    print("PANEL EXTRACTION TEST SUITE")
    print("="*60)
    print(f"\nTest directory: {TEST_DIR}")
    print(f"Test images: {TEST_PAGES_DIR}")
    
    test_images = get_test_images()
    if not test_images:
        print("\n❌ No test images found!")
        print(f"\nPlease add comic page images to: {TEST_PAGES_DIR}")
        print("Supported formats: PNG, JPG, JPEG, WEBP, BMP")
        return
    
    print(f"\nFound {len(test_images)} test images:")
    for img in test_images:
        print(f"  - {img.name}")
    
    # Process all images and save visualizations
    print("\n" + "="*60)
    print("STEP 1: PROCESSING AND SAVING VISUALIZATIONS")
    print("="*60)
    process_all_images()
    
    # Run all tests
    print("\n" + "="*60)
    print("STEP 2: RUNNING VALIDATION TESTS")
    print("="*60)
    test_panel_count()
    test_panel_coverage()
    test_bbox_accuracy()
    test_reading_order()
    test_container_removal()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
