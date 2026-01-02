import cv2
import numpy as np
import os
import glob
from scipy.spatial.distance import cdist

def get_four_corners(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int32(box)

    indices = []
    for p in box:
        diff = contour[:, 0, :] - p
        dist = np.sum(diff**2, axis=1)
        closest_idx = np.argmin(dist)
        indices.append(closest_idx)

    indices = sorted(indices)
    ordered_corners = [contour[i][0] for i in indices]
    return np.array(ordered_corners), indices

def extract_side(contour, start_idx, end_idx):
    if end_idx < start_idx:
        side = np.vstack((contour[start_idx:], contour[:end_idx+1]))
    else:
        side = contour[start_idx:end_idx+1]
    return side[:, 0, :]

def normalize_side(side, num_points=50):
    diffs = side[1:] - side[:-1]
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    cum_dist = np.insert(np.cumsum(dists), 0, 0)
    total_dist = cum_dist[-1]

    if total_dist == 0:
        return np.zeros((num_points, 2))

    x = side[:, 0]
    y = side[:, 1]

    new_dists = np.linspace(0, total_dist, num_points)
    new_x = np.interp(new_dists, cum_dist, x)
    new_y = np.interp(new_dists, cum_dist, y)

    resampled = np.column_stack((new_x, new_y))

    start = resampled[0]
    end = resampled[-1]

    resampled -= start

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    angle = np.arctan2(dy, dx)

    c, s = np.cos(-angle), np.sin(-angle)
    R = np.array([[c, -s], [s, c]])

    rotated = np.dot(resampled, R.T)

    scale = 1.0 / np.sqrt(dx**2 + dy**2)
    normalized = rotated * scale

    return normalized

def is_flat(norm_side, threshold=0.05):
    # Check max deviation from y=0
    max_dev = np.max(np.abs(norm_side[:, 1]))
    return max_dev < threshold

def process_piece(filename, img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    corners, corner_indices = get_four_corners(cnt)

    sides = []
    for i in range(4):
        start = corner_indices[i]
        end = corner_indices[(i+1)%4]
        raw_side = extract_side(cnt, start, end)
        norm_side = normalize_side(raw_side)
        sides.append({
            "raw": raw_side,
            "norm": norm_side,
            "side_idx": i,
            "is_flat": is_flat(norm_side)
        })

    return {
        "filename": filename,
        "sides": sides,
        "img": img # Keep image for visualization
    }

def main():
    pieces_dir = "pieces"
    image_files = glob.glob(os.path.join(pieces_dir, "*.jpg"))
    print(f"Found {len(image_files)} images.")

    pieces_data = []
    for f in image_files:
        img = cv2.imread(f)
        if img is None: continue
        name = os.path.basename(f)
        res = process_piece(name, img)
        if res:
            for side in res['sides']:
                side['norm_rev'] = normalize_side(side['raw'][::-1])
            pieces_data.append(res)

    print(f"Processed {len(pieces_data)} pieces.")

    matches = []
    for i in range(len(pieces_data)):
        p1 = pieces_data[i]
        for j in range(i+1, len(pieces_data)):
            p2 = pieces_data[j]

            for s1 in p1['sides']:
                for s2 in p2['sides']:
                    # Skip if both are flat
                    if s1['is_flat'] and s2['is_flat']:
                        continue

                    dist = np.sum((s1['norm'] - s2['norm_rev'])**2)

                    matches.append({
                        "p1": p1['filename'],
                        "s1": s1['side_idx'],
                        "p2": p2['filename'],
                        "s2": s2['side_idx'],
                        "score": dist
                    })

    matches.sort(key=lambda x: x['score'])

    print("Top 20 Non-Flat Matches:")
    with open("matches.txt", "w") as f:
        for m in matches[:50]:
            line = f"{m['p1']} (Side {m['s1']}) - {m['p2']} (Side {m['s2']}) : Score {m['score']:.4f}"
            print(line)
            f.write(line + "\n")

    # Visualize top match
    if matches:
        top = matches[0]
        print(f"Visualizing top match: {top['p1']} vs {top['p2']}")

        # Draw the normalized curves
        # Create a blank image
        vis = np.zeros((400, 800, 3), dtype=np.uint8)

        # Get the normalized curves
        # Find the piece objects
        p1_obj = next(p for p in pieces_data if p['filename'] == top['p1'])
        p2_obj = next(p for p in pieces_data if p['filename'] == top['p2'])

        s1_norm = p1_obj['sides'][top['s1']]['norm']
        s2_norm_rev = p2_obj['sides'][top['s2']]['norm_rev']

        # Scale and shift for visualization
        # s1: left side (0-400 width)
        # s2: right side (400-800 width)

        def draw_curve(img, curve, offset_x, offset_y, color):
            # curve is Nx2, values approx 0-1 (x) and -0.3 to 0.3 (y)
            # Scale x by 300, y by 300
            pts = []
            for pt in curve:
                x = int(pt[0] * 300 + offset_x)
                y = int(pt[1] * 300 + offset_y)
                pts.append((x, y))

            for i in range(len(pts)-1):
                cv2.line(img, pts[i], pts[i+1], color, 2)

        draw_curve(vis, s1_norm, 50, 200, (0, 255, 0)) # Green
        draw_curve(vis, s2_norm_rev, 50, 200, (0, 0, 255)) # Red (Overlaid)

        # Also draw them side by side
        draw_curve(vis, s1_norm, 450, 100, (0, 255, 0))
        draw_curve(vis, s2_norm_rev, 450, 300, (0, 0, 255))

        cv2.imwrite("match_vis.jpg", vis)
        print("Saved match_vis.jpg")

if __name__ == "__main__":
    main()
