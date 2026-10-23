import csv  #helps save the result tables
import re  #helps get words from the OCR text
from pathlib import Path  #helps find the project files

import cv2  #used for image processing
import numpy as np  #used for arrays and corner coordinates
import pytesseract  #used to read text from the images

#folders and settings
folder = Path(__file__).resolve().parent  #folder where this code is saved
output = folder / "output"  #folder for the result images and tables

scene_files = ["scene_1.jpg", "scene_2.jpg", "scene_3.jpg"]
document_files = ["doc_1.jpg", "doc_2.jpg", "doc_3.jpg"]

RATIO_THRESHOLD = 0.75  #keeps a match if it passes the ratio test
MIN_INLIERS = 10  #requires at least 10 matches that agree with the transformation
MIN_INLIER_RATIO = 0.50  #at least half of the ratio-passed matches must be inliers

#if Tesseract is installed but Python cannot find it,
#uncomment the next line and check the installation path.
# pytesseract.pytesseract.tesseract_cmd = (
#     r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# )

def read_image(filename):
    image = cv2.imread(str(folder / filename))  #loads an image from the project folder

    if image is None:
        raise FileNotFoundError(f"Cannot open {filename}")

    return image

def save_table(filename, headings, rows):
    with open(output / filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headings)  #writes the column names
        writer.writerows(rows)  #writes the recorded results

# ============================================================
# PART A: FIND THE BOX
# ============================================================
def find_box(box, scene, method="ORB"):
    #changes both images to grayscale before finding features
    box_gray = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)
    scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)

    if method == "ORB":
        detector = cv2.ORB_create(nfeatures=2000)  #asks ORB for up to 2000 features
        distance = cv2.NORM_HAMMING  #distance used for ORB descriptors
    else:
        detector = cv2.SIFT_create()
        distance = cv2.NORM_L2  #distance used for SIFT descriptors

    #finds feature points and the descriptors used to compare them
    points1, descriptors1 = detector.detectAndCompute(box_gray, None)
    points2, descriptors2 = detector.detectAndCompute(scene_gray, None)

    raw = 0
    good = []
    inliers = 0
    found = False
    mask_list = None
    result = scene.copy()

    if descriptors1 is not None and descriptors2 is not None:
        if len(descriptors2) >= 2:
            matcher = cv2.BFMatcher(distance)
            pairs = matcher.knnMatch(descriptors1, descriptors2, k=2)  #gets the two closest choices

            for pair in pairs:
                if len(pair) == 2:
                    raw += 1
                    first, second = pair

                    #keeps the first choice only if it is clearly better
                    if first.distance < RATIO_THRESHOLD * second.distance:
                        good.append(first)

    #needs at least four pairs of points to estimate a homography
    if len(good) >= 4:
        source = np.float32([
            points1[m.queryIdx].pt for m in good
        ]).reshape(-1, 1, 2)

        destination = np.float32([
            points2[m.trainIdx].pt for m in good
        ]).reshape(-1, 1, 2)

        #uses RANSAC to find a transformation and mark the inliers\
        matrix, mask = cv2.findHomography(
            source, destination, cv2.RANSAC, 5.0
        )

        if matrix is not None and mask is not None:
            inliers = int(mask.sum())  #counts the matches accepted by RANSAC
            mask_list = mask.ravel().tolist()

            height, width = box_gray.shape
            corners = np.float32([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ]).reshape(-1, 1, 2)

            outline = cv2.perspectiveTransform(corners, matrix)  #moves the box corners into the scene

            if np.isfinite(outline).all():
                area = abs(cv2.contourArea(outline))
                scene_area = scene.shape[0] * scene.shape[1]

                #checks the inliers and outline before saying the box was found
                found = (
                    inliers >= MIN_INLIERS
                    and inliers / len(good) >= MIN_INLIER_RATIO
                    and cv2.isContourConvex(outline)
                    and 100 < area < scene_area
                )

                if found:
                    cv2.polylines(
                        result, [np.int32(outline)],
                        True, (0, 255, 0), 3
                    )

    decision = "FOUND" if found else "NOT FOUND"

    cv2.putText(
        result, f"{method}: {decision}", (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
    )

    #draws the inlier matches when a RANSAC mask is available
    matches_image = cv2.drawMatches(
        box, points1, scene, points2, good, None,
        matchesMask=mask_list,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return result, matches_image, raw, len(good), inliers, found

def part_a(box):
    rows = []
    successful_scene = None

    for index, filename in enumerate(scene_files):
        if not (folder / filename).exists():
            print(f"{filename}: MISSING FILE - cannot test detection.")
            rows.append([filename, "ORB", "", "", "", "MISSING FILE"])
            continue
        scene = read_image(filename)
        methods = ["ORB", "SIFT"] if index == 0 else ["ORB"]  #tries SIFT only on the first scene

        for method in methods:
            result, matches, raw, good, inliers, found = find_box(
                box, scene, method
            )

            name = Path(filename).stem
            decision = "FOUND" if found else "NOT FOUND"

            cv2.imwrite(
                str(output / f"A_{name}_{method}_outline.jpg"), result
            )
            cv2.imwrite(
                str(output / f"A_{name}_{method}_matches.jpg"), matches
            )

            rows.append([filename, method, raw, good, inliers, decision])

            print(
                f"{filename} | {method} | Raw: {raw} | "
                f"Ratio passed: {good} | Inliers: {inliers} | {decision}"
            )

            #use the first successful ORB scene for Part C.
            if method == "ORB" and found and successful_scene is None:
                successful_scene = filename

    save_table(
        "A_results.csv",
        ["scene", "method", "raw", "ratio_passed", "inliers", "decision"],
        rows
    )
    return successful_scene

# ============================================================
# PART B: STRAIGHTEN DOCUMENTS
# ============================================================
def order_corners(points):
    #sort corners around their centre.
    centre = points.mean(axis=0)
    angles = np.arctan2(
        points[:, 1] - centre[1],
        points[:, 0] - centre[0]
    )
    ordered = points[np.argsort(angles)]

    #start with the top-left corner.
    start = np.argmin(ordered.sum(axis=1))
    return np.roll(ordered, -start, axis=0).astype(np.float32)

def straighten_document(image):
    #detect on a smaller image to keep processing manageable.
    scale = min(1.0, 1200 / max(image.shape[:2]))
    small = cv2.resize(image, None, fx=scale, fy=scale)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)  #finds possible paper edges after smoothing

    #connect small gaps in page boundaries.
    kernel = np.ones((5, 5), dtype=np.uint8)
    connected = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        connected, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(contours, key=cv2.contourArea, reverse=True)  #checks the larger regions first
    page = None
    minimum_area = 0.10 * small.shape[0] * small.shape[1]  #ignores very small regions

    #choose the largest convex quadrilateral above the area threshold.
    for contour in contours:
        if cv2.contourArea(contour) < minimum_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(
            contour, 0.02 * perimeter, True
        )

        if len(approximation) == 4 and cv2.isContourConvex(approximation):
            page = approximation.reshape(4, 2).astype(np.float32) / scale
            break

    outline = image.copy()

    if page is None:
        return None, outline, edges

    corners = order_corners(page)  #keeps the source and destination corner order the same
    top_left, top_right, bottom_right, bottom_left = corners

    width = int(max(
        np.linalg.norm(top_right - top_left),
        np.linalg.norm(bottom_right - bottom_left)
    ))

    height = int(max(
        np.linalg.norm(bottom_left - top_left),
        np.linalg.norm(bottom_right - top_right)
    ))

    if width < 2 or height < 2:
        return None, outline, edges

    destination = np.float32([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ])

    #maps the four detected corners to the corners of a rectangle
    matrix = cv2.getPerspectiveTransform(corners, destination)
    straightened = cv2.warpPerspective(image, matrix, (width, height))

    cv2.polylines(
        outline, [np.int32(corners)], True, (0, 255, 0), 3
    )
    return straightened, outline, edges

def preprocess(image):
    #same receipt pipeline as my Homework 1, including its final cleanup.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)  #improves local contrast
    denoised = cv2.bilateralFilter(contrast, 7, 50, 50)  #reduces noise while keeping edges
    #uses a separate threshold for each small image region
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 41, 18
    )
    binary = cv2.medianBlur(binary, 3)  #removes small dots after thresholding
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

def ocr_words(image):
    #match the professor's score_ocr.py: fixed PSM 6, case-sensitive word sets.
    text = pytesseract.image_to_string(image, config="--psm 6")  #uses the same layout setting as the scoring script
    words = set(re.findall(r"[A-Za-z0-9\.\-]+", text))
    return text, words

def part_b():
    try:
        pytesseract.get_tesseract_version()
        _, truth = ocr_words(read_image("receipt_clean.png"))  #gets reference words from the clean receipt
        if not truth:
            raise ValueError("No reference words were read from receipt_clean.png")
        ocr_available = True
    except pytesseract.TesseractNotFoundError:
        ocr_available = False
        print("Tesseract was not found. Scores are unavailable; images will be saved.")

    rows = []
    for filename in document_files:
        image = read_image(filename)
        name = Path(filename).stem
        straightened, outline, edges = straighten_document(image)
        cv2.imwrite(str(output / f"B_{name}_edges.png"), edges)
        cv2.imwrite(str(output / f"B_{name}_outline.jpg"), outline)
        before = preprocess(image)
        cv2.imwrite(str(output / f"B_{name}_before.png"), before)
        pictures = [("before", before)]
        if straightened is not None:
            after = preprocess(straightened)
            cv2.imwrite(str(output / f"B_{name}_straightened.jpg"), straightened)
            cv2.imwrite(str(output / f"B_{name}_after.png"), after)
            pictures.append(("after", after))
        else:
            print(f"{filename}: PAGE NOT FOUND. Check the edge image.")

        scores = {}
        counts = {}
        for label, picture in pictures:
            if ocr_available:
                text, words = ocr_words(picture)
                (output / f"B_{name}_{label}_ocr.txt").write_text(text, encoding="utf-8")
                counts[label] = len(words & truth)  #counts words also found in the reference
                scores[label] = counts[label] / len(truth)  #calculates the required word score
                print(f"{filename} {label}: {scores[label]:.2f} ({counts[label]} of {len(truth)} words)")

        #a four-corner candidate may still cover the wrong part of the page
        status = "CANDIDATE FOUND - CHECK OUTLINE" if straightened is not None else "PAGE NOT FOUND"
        rows.append([filename, status, scores.get("before", ""), scores.get("after", ""),
                     counts.get("before", ""), counts.get("after", ""),
                     len(truth) if ocr_available else ""])

    save_table("B_results.csv", ["document", "status", "before_score", "after_score",
                                "before_matched", "after_matched", "reference_words"], rows)

# ============================================================
# PART c: SHRINK UNTIL MATCHING FAILS
# ============================================================
def part_c(box, filename):
    if filename is None:
        print("Part C skipped: no successful ORB scene from Part A.")
        return

    scene = read_image(filename)
    rows = []

    #each image is resized directly from the original.
    scales = [1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]
    first_failure = None

    for scale in scales:
        smaller = cv2.resize(
            scene, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_AREA
        )

        result, _, raw, good, inliers, found = find_box(
            box, smaller, "ORB"
        )

        decision = "FOUND" if found else "NOT FOUND"
        rows.append([filename, scale, raw, good, inliers, decision])

        cv2.imwrite(
            str(output / f"C_scale_{scale:.2f}.jpg"), result
        )

        print(f"Scale: {scale:.2f} | Inliers: {inliers} | {decision}")

        #records the first failed scale in the order tested
        if not found and first_failure is None:
            first_failure = scale

    save_table(
        "C_results.csv",
        ["scene", "scale", "raw", "ratio_passed", "inliers", "decision"],
        rows
    )

    if first_failure is not None:
        print(f"First tested failure while shrinking: {first_failure:.2f}")
    else:
        print("No failure at the tested scales. Try smaller scales.")

# ============================================================
# RUN THE THREE (3) PARTS
# ============================================================
def main():
    output.mkdir(exist_ok=True)  #creates the output folder if needed
    cv2.setRNGSeed(0)  #sets a repeatable starting seed for OpenCV
    box = read_image("box.png")

    print("\nPART A: FIND THE BOX")
    successful_scene = part_a(box)

    print("\nPART B: STRAIGHTEN DOCUMENTS")
    part_b()

    print("\nPART C: SHRINK THE SCENE")
    part_c(box, successful_scene)

    print("\nFinished! Open the output folder to see the results.")

if __name__ == "__main__":
    main()