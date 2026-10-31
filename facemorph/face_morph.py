import cv2
import numpy as np
import mediapipe as mp
import requests
import subprocess
import os
from scipy.spatial import Delaunay

# ==========================================
# CONFIGURATION
# ==========================================

URL_A = "https://upload.wikimedia.org/wikipedia/commons/5/54/LORETTAYoung.jpg"

URL_B = "https://upload.wikimedia.org/wikipedia/commons/6/62/John_F._Kerry.jpg"

SIZE = 256
FPS = 16
DURATION = 2.5
OUTPUT = "face_morph.mp4"
RAW_VIDEO = "raw_morph.mp4"

# ==========================================
# DOWNLOAD IMAGES
# ==========================================

def download_image(url, filename):
    headers = {
        "User-Agent": (
            "FaceMorphAssignment/1.0 "
            "(educational computer vision project; Python requests)"
        ),
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Referer": "https://commons.wikimedia.org/"
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=30
    )

    response.raise_for_status()

    with open(filename, "wb") as f:
        f.write(response.content)

    image = cv2.imread(filename)

    if image is None:
        raise RuntimeError(f"Downloaded file is not a valid image: {filename}")

    print(f"Downloaded {filename}: {len(response.content):,} bytes")

    return image


print("Downloading photographs...")

img_a = download_image(URL_A, "face_a.jpg")
img_b = download_image(URL_B, "face_b.jpg")

# ==========================================
# FACE DETECTION
# ==========================================

mp_face_mesh = mp.solutions.face_mesh

detector = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)


def extract_face(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = detector.process(rgb)

    if not result.multi_face_landmarks:
        raise RuntimeError("No face detected!")

    landmarks = result.multi_face_landmarks[0].landmark

    points = np.array([
        [p.x * image.shape[1], p.y * image.shape[0]]
        for p in landmarks
    ], dtype=np.float32)

    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)

    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2

    side = int(np.ceil(
        max(x_max - x_min, y_max - y_min) * 1.45
    ))

    x1 = int(np.floor(cx - side / 2))
    y1 = int(np.floor(cy - side / 2))

    # Always create an exactly side x side crop
    crop = np.zeros((side, side, 3), dtype=np.uint8)

    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(image.shape[1], x1 + side)
    src_y2 = min(image.shape[0], y1 + side)

    dst_x1 = src_x1 - x1
    dst_y1 = src_y1 - y1

    width = src_x2 - src_x1
    height = src_y2 - src_y1

    crop[
        dst_y1:dst_y1 + height,
        dst_x1:dst_x1 + width
    ] = image[
        src_y1:src_y2,
        src_x1:src_x2
    ]

    # Transform landmarks
    points[:, 0] -= x1
    points[:, 1] -= y1

    scale = SIZE / side

    crop = cv2.resize(crop, (SIZE, SIZE))
    points *= scale

    return crop, points


print("Detecting facial landmarks...")

face_a, points_a = extract_face(img_a)
face_b, points_b = extract_face(img_b)

detector.close()

# ==========================================
# TRIANGULATION
# ==========================================

# Add four corner points to stabilize the image boundary
corners = np.array([
    [0, 0],
    [SIZE - 1, 0],
    [SIZE - 1, SIZE - 1],
    [0, SIZE - 1]
], dtype=np.float32)

points_a = np.vstack([points_a, corners])
points_b = np.vstack([points_b, corners])

average_points = (points_a + points_b) / 2

triangles = Delaunay(average_points).simplices

# ==========================================
# AFFINE TRIANGLE WARPING
# ==========================================

def warp_triangle(src, dst, t_src, t_dst):

    r1 = cv2.boundingRect(np.float32([t_src]))
    r2 = cv2.boundingRect(np.float32([t_dst]))

    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2

    if min(w1, h1, w2, h2) <= 0:
        return

    t1 = np.float32([
        [p[0] - x1, p[1] - y1]
        for p in t_src
    ])

    t2 = np.float32([
        [p[0] - x2, p[1] - y2]
        for p in t_dst
    ])

    src_crop = src[y1:y1+h1, x1:x1+w1]

    matrix = cv2.getAffineTransform(t1, t2)

    warped = cv2.warpAffine(
        src_crop,
        matrix,
        (w2, h2),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    mask = np.zeros((h2, w2), dtype=np.float32)

    cv2.fillConvexPoly(
        mask,
        np.int32(t2),
        1.0
    )

    roi = dst[y2:y2+h2, x2:x2+w2]

    mask3 = mask[:, :, None]

    dst[y2:y2+h2, x2:x2+w2] = (
        roi * (1 - mask3) +
        warped * mask3
    )


def morph_face(img1, img2, p1, p2, alpha):

    # Intermediate facial geometry
    points = (1 - alpha) * p1 + alpha * p2

    warp_a = np.zeros_like(img1, dtype=np.float32)
    warp_b = np.zeros_like(img2, dtype=np.float32)

    img1f = img1.astype(np.float32)
    img2f = img2.astype(np.float32)

    for tri in triangles:

        t1 = p1[tri]
        t2 = p2[tri]
        t = points[tri]

        warp_triangle(img1f, warp_a, t1, t)
        warp_triangle(img2f, warp_b, t2, t)

    # Cross-dissolve warped faces
    result = (
        (1 - alpha) * warp_a +
        alpha * warp_b
    )

    return np.clip(result, 0, 255).astype(np.uint8)


# ==========================================
# GENERATE VIDEO FRAMES
# ==========================================

print("Generating morph animation...")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    RAW_VIDEO,
    fourcc,
    FPS,
    (SIZE, SIZE)
)

total_frames = int(FPS * DURATION)

for i in range(total_frames):

    t = i / (total_frames - 1)

    # Smooth easing
    alpha = 3 * t**2 - 2 * t**3

    frame = morph_face(
        face_a,
        face_b,
        points_a,
        points_b,
        alpha
    )

    writer.write(frame)

writer.release()

# ==========================================
# COMPRESS TO UNDER 100 KB
# ==========================================

print("Compressing video...")

for bitrate in ["100k", "75k", "50k", "35k", "25k"]:

    subprocess.run([
        "ffmpeg",
        "-y",
        "-i", RAW_VIDEO,
        "-an",
        "-vf", "fps=12,scale=256:256:flags=lanczos",
        "-c:v", "libx264",
        "-preset", "veryslow",
        "-pix_fmt", "yuv420p",
        "-b:v", bitrate,
        "-maxrate", bitrate,
        "-bufsize", bitrate,
        "-movflags", "+faststart",
        OUTPUT
    ], stdout=subprocess.DEVNULL,
       stderr=subprocess.DEVNULL,
       check=True)

    size = os.path.getsize(OUTPUT)

    print(f"Bitrate: {bitrate} | Size: {size:,} bytes")

    if size < 100000:
        break

if os.path.getsize(OUTPUT) >= 100000:
    raise RuntimeError("Video exceeds 100 KB. Reduce resolution or duration.")

os.remove(RAW_VIDEO)

print("\nSUCCESS!")
print("Output:", OUTPUT)
print("Size:", os.path.getsize(OUTPUT), "bytes")
print("Resolution:", SIZE, "x", SIZE)
print("Duration:", DURATION, "seconds")
