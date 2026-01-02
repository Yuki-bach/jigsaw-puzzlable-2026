import cv2
import numpy as np
import solver
import os

def label_sides(filename):
    img = cv2.imread(os.path.join("pieces", filename))
    if img is None:
        print(f"Could not read {filename}")
        return

    res = solver.process_piece(filename, img)
    if not res:
        print(f"Could not process {filename}")
        return

    # Draw corners and side numbers
    corners = res['sides']
    for i, side in enumerate(corners):
        # Get midpoint of the side for labeling
        raw_side = side['raw']
        mid_idx = len(raw_side) // 2
        mid_pt = raw_side[mid_idx]

        # Draw side number
        cv2.putText(img, str(i), (int(mid_pt[0]), int(mid_pt[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 5)

        # Draw start point of side
        start_pt = raw_side[0]
        cv2.circle(img, (int(start_pt[0]), int(start_pt[1])), 20, (255, 0, 0), -1)

    output_filename = f"labeled_{filename}"
    cv2.imwrite(output_filename, img)
    print(f"Saved {output_filename}")

if __name__ == "__main__":
    label_sides("piece_072.jpg")
    label_sides("piece_036.jpg")
