"""Panel extraction using computer vision - Border Edge Detection Method."""

from pathlib import Path
from typing import List, Dict, Tuple
import cv2
import numpy as np


class PanelExtractor:
    """
    Extracts comic panels using edge-based detection.
    Based on test_border_edges.py - proven to work best for manga/comics.
    
    Strategy:
    1. Find white patches (dark content areas) via connected components
    2. Extract 2-pixel border edges (leftmost, rightmost, topmost, bottommost)
    3. Merge collinear line segments within 30px tolerance
    4. Form rectangles from edge combinations (max 20% gap filling)
    5. Sort panels in reading order
    """
    
    def __init__(self, min_panel_pct: float = 10.0, max_panel_pct: float = 95.0):
        """
        Args:
            min_panel_pct: Minimum panel size as percentage of page dimension
            max_panel_pct: Maximum panel size as percentage of page area
        """
        self.min_panel_width_pct = min_panel_pct / 100
        self.min_panel_height_pct = min_panel_pct / 100
        self.max_panel_area_pct = max_panel_pct / 100
        self.black_threshold = 50
        self.merge_tolerance = 30
        self.max_gap_pct = 0.20
    
    def extract_panels(self, image_path: Path, content_type: str = "comic") -> List[Dict]:
        """
        Extract all panels from a comic page using edge-based detection.
        
        Args:
            image_path: Path to comic page image
            content_type: "manga" for right-to-left, "comic" for left-to-right
            
        Returns:
            List of panel dictionaries with image, bbox, and metadata
        """
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            return []
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Step 1: Create black mask (dark areas = white patches)
        _, black_mask = cv2.threshold(gray, self.black_threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Step 2: Find connected components (white patches)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(black_mask, connectivity=8)
        
        # Step 3: Extract border edges
        h_edges, v_edges = self._extract_border_edges(labels, num_labels, width, height)
        
        # Step 4: Merge collinear edges
        h_edges = self._merge_collinear_edges(h_edges, axis='horizontal')
        v_edges = self._merge_collinear_edges(v_edges, axis='vertical')
        
        # Add image borders
        h_edges.append((0, 0, width - 1))
        h_edges.append((height - 1, 0, width - 1))
        v_edges.append((0, 0, height - 1))
        v_edges.append((width - 1, 0, height - 1))
        
        # Step 5: Detect right angles
        connections = self._detect_right_angles(h_edges, v_edges)
        
        # Step 6: Form rectangles from edges
        all_panels = self._form_rectangles_from_edges(h_edges, v_edges, connections, width, height)
        
        # Check if panels cover enough of the page (at least 80%)
        total_page_area = width * height
        total_panel_area = sum((right - left) * (bottom - top) for left, top, right, bottom in all_panels)
        coverage_ratio = total_panel_area / total_page_area if total_page_area > 0 else 0
        
        # If coverage is too low, use the full page as one panel
        if coverage_ratio < 0.80:
            print(f"   ⚠️  Low coverage ({coverage_ratio*100:.1f}%), using full page as single panel")
            all_panels = [(0, 0, width, height)]
        
        # Convert to panel dictionaries
        panels = []
        for i, (left, top, right, bottom) in enumerate(all_panels):
            panel_img = img[top:bottom, left:right].copy()
            area = (right - left) * (bottom - top)
            
            panels.append({
                'panel_number': i + 1,
                'bbox': [left, top, right - left, bottom - top],  # x, y, width, height
                'image': panel_img,
                'area': area
            })
        
        # Sort in reading order
        panels = self._sort_reading_order(panels, (height, width), content_type)
        
        # Renumber after sorting
        for i, panel in enumerate(panels):
            panel['panel_number'] = i + 1
        
        return panels
    
    def _extract_border_edges(self, labels, num_labels, width, height):
        """Extract 2-pixel edges from connected components, then consolidate (like test_border_edges.py)."""
        min_h_length = int(width * 0.15)  # 15% of width
        min_v_length = int(height * 0.10)  # 10% of height
        
        # Create border images (like test_border_edges.py)
        horizontal_borders = np.zeros((height, width), dtype=np.uint8)
        vertical_borders = np.zeros((height, width), dtype=np.uint8)
        
        for label in range(1, num_labels):
            patch_mask = (labels == label).astype(np.uint8) * 255
            
            # Extract horizontal edges (top and bottom)
            rows_with_white = np.where(np.any(patch_mask == 255, axis=1))[0]
            if len(rows_with_white) > 0:
                top_row = rows_with_white[0]
                bottom_row = rows_with_white[-1]
                
                # Process top edge (topmost 2 pixels)
                top_edge = patch_mask[top_row:min(top_row+2, height), :]
                if top_edge.shape[0] > 0:
                    row_idx = np.argmax(np.sum(top_edge == 255, axis=1))
                    edge_row = top_edge[row_idx, :]
                    cols_with_white = np.where(edge_row == 255)[0]
                    
                    if len(cols_with_white) >= min_h_length:
                        span = cols_with_white[-1] - cols_with_white[0] + 1
                        coverage = len(cols_with_white) / span
                        if coverage >= 0.8:  # 80% continuous
                            actual_row = top_row + row_idx
                            horizontal_borders[actual_row, cols_with_white] = 255
                
                # Process bottom edge (bottommost 2 pixels)
                bottom_edge = patch_mask[max(bottom_row-1, 0):bottom_row+1, :]
                if bottom_edge.shape[0] > 0:
                    row_idx = np.argmax(np.sum(bottom_edge == 255, axis=1))
                    edge_row = bottom_edge[row_idx, :]
                    cols_with_white = np.where(edge_row == 255)[0]
                    
                    if len(cols_with_white) >= min_h_length:
                        span = cols_with_white[-1] - cols_with_white[0] + 1
                        coverage = len(cols_with_white) / span
                        if coverage >= 0.8:
                            actual_row = max(bottom_row-1, 0) + row_idx
                            horizontal_borders[actual_row, cols_with_white] = 255
            
            # Extract vertical edges (left and right)
            cols_with_white = np.where(np.any(patch_mask == 255, axis=0))[0]
            if len(cols_with_white) > 0:
                left_col = cols_with_white[0]
                right_col = cols_with_white[-1]
                
                # Process left edge (leftmost 2 pixels)
                left_edge = patch_mask[:, left_col:min(left_col+2, width)]
                if left_edge.shape[1] > 0:
                    col_idx = np.argmax(np.sum(left_edge == 255, axis=0))
                    edge_col = left_edge[:, col_idx]
                    rows_with_white = np.where(edge_col == 255)[0]
                    
                    if len(rows_with_white) >= min_v_length:
                        span = rows_with_white[-1] - rows_with_white[0] + 1
                        coverage = len(rows_with_white) / span
                        if coverage >= 0.8:
                            actual_col = left_col + col_idx
                            vertical_borders[rows_with_white, actual_col] = 255
                
                # Process right edge (rightmost 2 pixels)
                right_edge = patch_mask[:, max(right_col-1, 0):right_col+1]
                if right_edge.shape[1] > 0:
                    col_idx = np.argmax(np.sum(right_edge == 255, axis=0))
                    edge_col = right_edge[:, col_idx]
                    rows_with_white = np.where(edge_col == 255)[0]
                    
                    if len(rows_with_white) >= min_v_length:
                        span = rows_with_white[-1] - rows_with_white[0] + 1
                        coverage = len(rows_with_white) / span
                        if coverage >= 0.8:
                            actual_col = max(right_col-1, 0) + col_idx
                            vertical_borders[rows_with_white, actual_col] = 255
        
        # Consolidate edges per row/column (like test_border_edges.py)
        # This combines edges from different components at the same y/x coordinate
        h_edges = []  # (y, x_start, x_end)
        for y in range(height):
            x_coords = np.where(horizontal_borders[y, :] == 255)[0]
            if len(x_coords) > 0:
                h_edges.append((y, x_coords[0], x_coords[-1]))
        
        v_edges = []  # (x, y_start, y_end)
        for x in range(width):
            y_coords = np.where(vertical_borders[:, x] == 255)[0]
            if len(y_coords) > 0:
                v_edges.append((x, y_coords[0], y_coords[-1]))
        
        return h_edges, v_edges
    
    def _merge_collinear_edges(self, edges: List[Tuple], axis: str) -> List[Tuple]:
        """Merge edges on the same line that are close together."""
        if not edges:
            return []
        
        sorted_edges = sorted(edges, key=lambda x: (x[0], x[1]))
        merged = []
        
        i = 0
        while i < len(sorted_edges):
            current_y, current_start, current_end = sorted_edges[i]
            merged_start = current_start
            merged_end = current_end
            
            j = i + 1
            while j < len(sorted_edges):
                next_y, next_start, next_end = sorted_edges[j]
                
                if abs(next_y - current_y) > 5:
                    break
                
                gap = next_start - merged_end
                if gap <= self.merge_tolerance:
                    merged_end = max(merged_end, next_end)
                    j += 1
                else:
                    break
            
            merged.append((current_y, merged_start, merged_end))
            i = j if j > i + 1 else i + 1
        
        return merged
    
    def _detect_right_angles(self, h_edges: List, v_edges: List) -> List[Tuple]:
        """Find points where horizontal and vertical edges intersect."""
        connections = []
        tolerance = 10
        
        for h_idx, (h_y, h_x_start, h_x_end) in enumerate(h_edges):
            for v_idx, (v_x, v_y_start, v_y_end) in enumerate(v_edges):
                if (v_y_start - tolerance <= h_y <= v_y_end + tolerance and
                    h_x_start - tolerance <= v_x <= h_x_end + tolerance):
                    connections.append((h_idx, v_idx, v_x, h_y))
        
        return connections
    
    def _form_rectangles_from_edges(self, h_edges: List, v_edges: List, 
                                     connections: List, width: int, height: int) -> List[Tuple]:
        """Form rectangles from edge combinations (matching test_border_edges.py exactly)."""
        min_width = int(width * 0.10)  # 10% of width
        min_height = int(height * 0.10)  # 10% of height
        max_gap_percent = 0.20  # Allow 20% gap (was 0.15 in test_border_edges but being more permissive)
        
        candidate_rectangles = []
        
        # For each pair of horizontal lines
        for i, (y1, x1_start, x1_end) in enumerate(h_edges):
            for j, (y2, x2_start, x2_end) in enumerate(h_edges):
                if i == j or abs(y2 - y1) < 10:  # Skip same or too close
                    continue
                
                rect_top = min(y1, y2)
                rect_bottom = max(y1, y2)
                rect_height = rect_bottom - rect_top
                
                # For each pair of vertical lines
                for k, (x1, y1a_start, y1a_end) in enumerate(v_edges):
                    for l, (x2, y2a_start, y2a_end) in enumerate(v_edges):
                        if k == l or abs(x2 - x1) < 10:  # Skip same or too close
                            continue
                        
                        rect_left = min(x1, x2)
                        rect_right = max(x1, x2)
                        rect_width = rect_right - rect_left
                        
                        # Minimum panel size: 10% of image dimensions
                        if rect_width < min_width or rect_height < min_height:  # Skip panels < 10% of image
                            continue
                        
                        # Check if this rectangle has at least one right angle connection
                        has_connection = False
                        corners = [
                            (i if y1 == rect_top else j, k if x1 == rect_left else l),  # top-left
                            (i if y1 == rect_top else j, l if x2 == rect_right else k),  # top-right
                            (j if y2 == rect_bottom else i, k if x1 == rect_left else l),  # bottom-left
                            (j if y2 == rect_bottom else i, l if x2 == rect_right else k),  # bottom-right
                        ]
                        
                        for h_idx, v_idx in corners:
                            for conn_h, conn_v, _, _ in connections:
                                if conn_h == h_idx and conn_v == v_idx:
                                    has_connection = True
                                    break
                            if has_connection:
                                break
                        
                        # Calculate gaps needed to close the rectangle
                        top_h_idx = i if y1 == rect_top else j
                        bottom_h_idx = j if y2 == rect_bottom else i
                        left_v_idx = k if x1 == rect_left else l
                        right_v_idx = l if x2 == rect_right else k
                        
                        h_top = h_edges[top_h_idx]
                        h_bottom = h_edges[bottom_h_idx]
                        v_left = v_edges[left_v_idx]
                        v_right = v_edges[right_v_idx]
                        
                        # Calculate how much of each edge is MISSING (needs to be drawn)
                        # Top edge: needs to span from rect_left to rect_right
                        top_needed = rect_width  # Total length needed
                        top_exists = max(0, min(h_top[2], rect_right) - max(h_top[1], rect_left))  # Existing coverage
                        top_gap = top_needed - top_exists
                        
                        # Bottom edge
                        bottom_needed = rect_width
                        bottom_exists = max(0, min(h_bottom[2], rect_right) - max(h_bottom[1], rect_left))
                        bottom_gap = bottom_needed - bottom_exists
                        
                        # Left edge: needs to span from rect_top to rect_bottom
                        left_needed = rect_height
                        left_exists = max(0, min(v_left[2], rect_bottom) - max(v_left[1], rect_top))
                        left_gap = left_needed - left_exists
                        
                        # Right edge
                        right_needed = rect_height
                        right_exists = max(0, min(v_right[2], rect_bottom) - max(v_right[1], rect_top))
                        right_gap = right_needed - right_exists
                        
                        total_gap = top_gap + bottom_gap + left_gap + right_gap
                        perimeter = 2 * (rect_width + rect_height)
                        gap_percent = total_gap / perimeter if perimeter > 0 else 1.0
                        
                        # Only accept if gap is ≤ 15%
                        if gap_percent <= max_gap_percent:
                            area = rect_width * rect_height
                            priority = 0 if has_connection else 1  # Prioritize rectangles with connections
                            candidate_rectangles.append({
                                'rect': (rect_left, rect_top, rect_right, rect_bottom),
                                'area': area,
                                'gap_percent': gap_percent,
                                'priority': priority,
                                'has_connection': has_connection
                            })
        
        # Sort candidates: first by priority (connections first), then by area (smaller first), then by gap
        candidate_rectangles.sort(key=lambda x: (x['priority'], x['area'], x['gap_percent']))
        
        # Select non-overlapping rectangles, starting with smallest ones with connections
        rectangles = []
        for candidate in candidate_rectangles:
            rect = candidate['rect']
            rect_left, rect_top, rect_right, rect_bottom = rect
            
            # Check for overlap with already selected rectangles
            overlaps = False
            for (ex_left, ex_top, ex_right, ex_bottom) in rectangles:
                # Check if rectangles overlap (not just touch)
                if not (rect_right <= ex_left or rect_left >= ex_right or 
                       rect_bottom <= ex_top or rect_top >= ex_bottom):
                    overlaps = True
                    break
            
            if not overlaps:
                rectangles.append(rect)
        
        return rectangles
    
    def _process_contour(self, contour, img: np.ndarray, img_area: int, 
                        current_panel_count: int) -> Dict:
        """Legacy method - not used in new implementation."""
        pass
    
    def _sort_reading_order(self, panels: List[Dict], img_shape: Tuple, 
                           content_type: str) -> List[Dict]:
        """Sort panels in correct reading order."""
        if not panels:
            return panels
        
        img_height = img_shape[0]
        rows = []
        current_row = []
        last_y = -1
        
        # Group by vertical position (rows)
        panels_by_y = sorted(panels, key=lambda p: p['bbox'][1])  # Sort by y
        
        for panel in panels_by_y:
            y = panel['bbox'][1]
            
            if last_y == -1 or abs(y - last_y) > img_height * 0.1:
                if current_row:
                    rows.append(current_row)
                current_row = [panel]
                last_y = y
            else:
                current_row.append(panel)
        
        if current_row:
            rows.append(current_row)
        
        # Sort each row horizontally
        sorted_panels = []
        for row in rows:
            if content_type == "manga":
                # Right to left for manga
                row_sorted = sorted(row, key=lambda p: p['bbox'][0], reverse=True)
            else:
                # Left to right for comics
                row_sorted = sorted(row, key=lambda p: p['bbox'][0])
            sorted_panels.extend(row_sorted)
        
        return sorted_panels

    def _detect_black_borders(self, gray: np.ndarray, img: np.ndarray) -> List:
        """Legacy method - not used in new implementation."""
        pass

    def _is_full_page_spread(self, gray: np.ndarray, img: np.ndarray) -> bool:
        """Legacy method - not used in new implementation."""
        pass
