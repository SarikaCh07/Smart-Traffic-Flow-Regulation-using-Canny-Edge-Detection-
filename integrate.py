import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import torch
from torchvision import transforms
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from CannyEdgeDetector import *

# Load COCO class labels
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# Load model
weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
model = ssdlite320_mobilenet_v3_large(weights=weights)
model.eval()

# App title
st.title("🚦 Density-Based Smart Traffic Control System")

# File uploader
uploaded_file = st.file_uploader("Upload a Traffic Image", type=["jpg", "jpeg", "png"])

# Helper functions
def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def apply_canny(image):
    gray_img = rgb2gray(np.array(image))
    edge = CannyEdgeDetector([gray_img], sigma=1.4, kernel_size=5, lowthreshold=0.09, highthreshold=0.20, weak_pixel=100)
    processed = edge.detect()[0]
    return (processed / processed.max() * 255).astype(np.uint8)

def count_white_pixels(img):
    return np.sum(img == 255)

def detect_vehicles(image):
    img_rgb = np.array(image.convert("RGB"))
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img_rgb).unsqueeze(0)

    with torch.no_grad():
        prediction = model(img_tensor)

    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']
    found = False

    for i in range(len(labels)):
        class_id = labels[i].item()
        score = scores[i].item()

        if class_id in [3, 4, 6, 8] and score > 0.85:
            x1, y1, x2, y2 = boxes[i].cpu().numpy().astype(int)
            width, height = x2 - x1, y2 - y1
            if width > 100 and height > 100:
                found = True
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = "Ambulance" if COCO_INSTANCE_CATEGORY_NAMES[class_id] in ["bus", "truck"] else COCO_INSTANCE_CATEGORY_NAMES[class_id]
                cv2.putText(img_rgb, f'{label} ({score:.2f})', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return img_rgb, found

# Main logic
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Traffic Image", use_column_width=True)
    image = Image.open(uploaded_file)

    # Step 1: Detect ambulance
    result, found = detect_vehicles(image)
    st.image(result, caption="Ambulance Detection Result", use_column_width=True)

    if found:
        st.success("🚑 Ambulance detected! Green signal time: 60 seconds or until it passes.")
    else:
        st.warning("❌ No ambulance detected. Proceeding with traffic density analysis...")

        # Step 2: Apply Canny
        result_img = apply_canny(image)
        st.image(result_img, caption="Canny Edge Output", use_column_width=True, clamp=True)

        # Step 3: Load reference and calculate white pixel density
        reference_path = "gray/refrence.png"
        if not os.path.exists(reference_path):
            st.error("Reference image not found. Please ensure gray/refrence.png exists.")
        else:
            reference = Image.open(reference_path).convert("L")
            ref_array = np.array(reference)

            sample_pixels = count_white_pixels(result_img)
            img_height, img_width = result_img.shape
            image_area = img_height * img_width
            white_density = sample_pixels / image_area  # value between 0 and 1

            st.success(f"Sample White Pixels: {sample_pixels}\nImage Area: {image_area}\nWhite Pixel Density: {white_density:.4f}")

            # Step 4: Allocate green time
            if white_density >= 0.18:
                st.info("🚦 Traffic is very high. Green signal time: 60 seconds")
            elif white_density >= 0.14:
                st.info("🚦 Traffic is high. Green signal time: 50 seconds")
            elif white_density >= 0.10:
                st.info("🚦 Traffic is moderate. Green signal time: 40 seconds")
            elif white_density >= 0.06:
                st.info("🚦 Traffic is low. Green signal time: 30 seconds")
            else:
                st.info("🚦 Traffic is very low. Green signal time: 20 seconds")






















????????????????????????????????????//
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import torch
from torchvision import transforms
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from CannyEdgeDetector import *

# Load model
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
model = ssdlite320_mobilenet_v3_large(weights=weights)
model.eval()

st.title("🚦 Density-Based Smart Traffic Control System")

uploaded_file = st.file_uploader("📤 Upload a Traffic Image", type=["jpg", "jpeg", "png"])

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def apply_canny(image):
    gray_img = rgb2gray(np.array(image))
    edge = CannyEdgeDetector([gray_img], sigma=1.4, kernel_size=5, lowthreshold=0.09, highthreshold=0.20, weak_pixel=100)
    processed = edge.detect()[0]
    return (processed / processed.max() * 255).astype(np.uint8)

def count_white_pixels(img):
    return np.sum(img == 255)

def detect_vehicles(image):
    img_rgb = np.array(image.convert("RGB"))
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img_rgb).unsqueeze(0)

    with torch.no_grad():
        prediction = model(img_tensor)

    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']
    found = False

    for i in range(len(labels)):
        class_id = labels[i].item()
        score = scores[i].item()

        if class_id in [3, 4, 6, 8] and score > 0.85:
            x1, y1, x2, y2 = boxes[i].cpu().numpy().astype(int)
            width, height = x2 - x1, y2 - y1
            if width > 100 and height > 100:
                found = True
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = "Ambulance" if COCO_INSTANCE_CATEGORY_NAMES[class_id] in ["bus", "truck"] else COCO_INSTANCE_CATEGORY_NAMES[class_id]
                cv2.putText(img_rgb, f'{label} ({score:.2f})', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return img_rgb, found

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    image = Image.open(uploaded_file)

    # Step 1 - Ambulance detection
    if st.button("🚑 Detect Ambulance"):
        result_img, found = detect_vehicles(image)
        st.image(result_img, caption="Detection Output", use_column_width=True)
        if found:
            st.success("🚑 Ambulance detected! Green signal time: 60 seconds or until it passes.")
            st.session_state['ambulance_detected'] = True
        else:
            st.warning("❌ No ambulance found.")
            st.session_state['ambulance_detected'] = False

    # Step 2 - Only allow next steps if ambulance is not found
    if 'ambulance_detected' in st.session_state and not st.session_state['ambulance_detected']:
        if st.button("⚙️ Run Canny Edge Detection"):
            canny_img = apply_canny(image)
            st.image(canny_img, caption="Canny Edge Output", use_column_width=True)
            st.session_state['canny_output'] = canny_img

        if st.button("📊 Analyze Traffic & Allocate Green Time"):
            if 'canny_output' not in st.session_state:
                st.warning("⚠️ Please run Canny Edge Detection first.")
            else:
                result_img = st.session_state['canny_output']
                sample_pixels = count_white_pixels(result_img)
                img_height, img_width = result_img.shape
                image_area = img_height * img_width
                white_density = sample_pixels / image_area

                st.success(f"Sample White Pixels: {sample_pixels}\nImage Area: {image_area}\nWhite Pixel Density: {white_density:.4f}")

                if white_density >= 0.18:
                    st.info("🚦 Traffic is very high. Green signal time: 60 seconds")
                elif white_density >= 0.14:
                    st.info("🚦 Traffic is high. Green signal time: 50 seconds")
                elif white_density >= 0.10:
                    st.info("🚦 Traffic is moderate. Green signal time: 40 seconds")
                elif white_density >= 0.06:
                    st.info("🚦 Traffic is low. Green signal time: 30 seconds")
                else:
                    st.info("🚦 Traffic is very low. Green signal time: 20 seconds")














??????????????/sidebar
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import torch
import time
from torchvision import transforms
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from CannyEdgeDetector import *

# Load model + COCO labels
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
model = ssdlite320_mobilenet_v3_large(weights=weights)
model.eval()

# UI
st.set_page_config(page_title="Smart Traffic Control", layout="centered")
st.title("🚦 Density-Based Smart Traffic Control System")
st.sidebar.title("📋 Console Logs")

uploaded_file = st.file_uploader("📤 Upload a Traffic Image", type=["jpg", "jpeg", "png"])

# --- Helper Functions ---
def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b

def apply_canny(image):
    gray_img = rgb2gray(np.array(image))
    edge = CannyEdgeDetector([gray_img], sigma=1.4, kernel_size=5, lowthreshold=0.09, highthreshold=0.20, weak_pixel=100)
    processed = edge.detect()[0]
    return (processed / processed.max() * 255).astype(np.uint8)

def count_white_pixels(img):
    return np.sum(img == 255)

def detect_vehicles(image):
    img_rgb = np.array(image.convert("RGB"))
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img_rgb).unsqueeze(0)

    with torch.no_grad():
        prediction = model(img_tensor)

    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']
    found = False

    for i in range(len(labels)):
        class_id = labels[i].item()
        score = scores[i].item()

        if class_id in [3, 4, 6, 8] and score > 0.85:
            x1, y1, x2, y2 = boxes[i].cpu().numpy().astype(int)
            width, height = x2 - x1, y2 - y1
            if width > 100 and height > 100:
                found = True
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = "Ambulance" if COCO_INSTANCE_CATEGORY_NAMES[class_id] in ["bus", "truck"] else COCO_INSTANCE_CATEGORY_NAMES[class_id]
                cv2.putText(img_rgb, f'{label} ({score:.2f})', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return img_rgb, found

# --- Main App Logic ---
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    image = Image.open(uploaded_file)

    # Detect Ambulance
    if st.button("🚑 Detect Ambulance"):
        with st.spinner("Detecting vehicles..."):
            result_img, found = detect_vehicles(image)
            st.image(result_img, caption="Detection Output", use_column_width=True)
            st.sidebar.write("✅ Object detection complete.")

        if found:
            st.success("🚑 Ambulance detected! Green signal time: 60 seconds or until it passes.")
            st.session_state['ambulance_detected'] = True
            st.sidebar.success("Ambulance detected — override density logic.")
        else:
            st.warning("❌ No ambulance found.")
            st.session_state['ambulance_detected'] = False
            st.sidebar.info("No ambulance — continue to density analysis.")

    # Step 2 - If ambulance not detected
    if 'ambulance_detected' in st.session_state and not st.session_state['ambulance_detected']:
        if st.button("⚙️ Run Canny Edge Detection"):
            with st.spinner("Processing Canny Edge Detection..."):
                canny_img = apply_canny(image)
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.005)
                    progress.progress(i + 1)
                st.image(canny_img, caption="Canny Edge Output", use_column_width=True)
                st.session_state['canny_output'] = canny_img
                st.sidebar.success("Canny edges extracted successfully.")

        if st.button("📊 Analyze Traffic & Allocate Green Time"):
            if 'canny_output' not in st.session_state:
                st.warning("⚠️ Please run Canny Edge Detection first.")
            else:
                with st.spinner("Analyzing white pixel density..."):
                    result_img = st.session_state['canny_output']
                    sample_pixels = count_white_pixels(result_img)
                    img_height, img_width = result_img.shape
                    image_area = img_height * img_width
                    white_density = sample_pixels / image_area

                    st.success(f"Sample White Pixels: {sample_pixels}\nImage Area: {image_area}\nWhite Pixel Density: {white_density:.4f}")
                    st.sidebar.write(f"📏 White pixel density: {white_density:.4f}")

                    if white_density >= 0.18:
                        st.info("🚦 Traffic is very high. Green signal time: 60 seconds")
                    elif white_density >= 0.14:
                        st.info("🚦 Traffic is high. Green signal time: 50 seconds")
                    elif white_density >= 0.10:
                        st.info("🚦 Traffic is moderate. Green signal time: 40 seconds")
                    elif white_density >= 0.06:
                        st.info("🚦 Traffic is low. Green signal time: 30 seconds")
                    else:
                        st.info("🚦 Traffic is very low. Green signal time: 20 seconds")


?????????????????//2nd sidebr
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import torch
from torchvision import transforms
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from CannyEdgeDetector import *

# Load COCO class labels
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# Load model
weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
model = ssdlite320_mobilenet_v3_large(weights=weights)
model.eval()

# App title
st.title("🚦 Density-Based Smart Traffic Control System")

# File uploader
uploaded_file = st.file_uploader("Upload a Traffic Image", type=["jpg", "jpeg", "png"])

# Helper functions
def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def apply_canny(image):
    gray_img = rgb2gray(np.array(image))
    edge = CannyEdgeDetector([gray_img], sigma=1.4, kernel_size=5, lowthreshold=0.09, highthreshold=0.20, weak_pixel=100)
    processed = edge.detect()[0]
    return (processed / processed.max() * 255).astype(np.uint8)

def count_white_pixels(img):
    return np.sum(img == 255)

def detect_vehicles(image):
    img_rgb = np.array(image.convert("RGB"))
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img_rgb).unsqueeze(0)

    with torch.no_grad():
        prediction = model(img_tensor)

    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']
    found = False

    for i in range(len(labels)):
        class_id = labels[i].item()
        score = scores[i].item()

        if class_id in [3, 4, 6, 8] and score > 0.85:
            x1, y1, x2, y2 = boxes[i].cpu().numpy().astype(int)
            width, height = x2 - x1, y2 - y1
            if width > 100 and height > 100:
                found = True
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = "Ambulance" if COCO_INSTANCE_CATEGORY_NAMES[class_id] in ["bus", "truck"] else COCO_INSTANCE_CATEGORY_NAMES[class_id]
                cv2.putText(img_rgb, f'{label} ({score:.2f})', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return img_rgb, found

# Main logic
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Traffic Image", use_column_width=True)
    image = Image.open(uploaded_file)

    # Step 1: Detect ambulance
    result, found = detect_vehicles(image)
    st.image(result, caption="Ambulance Detection Result", use_column_width=True)

    if found:
        st.success("🚑 Ambulance detected! Green signal time: 60 seconds or until it passes.")
    else:
        st.warning("❌ No ambulance detected. Proceeding with traffic density analysis...")

        # Step 2: Apply Canny
        result_img = apply_canny(image)
        st.image(result_img, caption="Canny Edge Output", use_column_width=True, clamp=True)

        # Step 3: Load reference and calculate white pixel density
        reference_path = "gray/refrence.png"
        if not os.path.exists(reference_path):
            st.error("Reference image not found. Please ensure gray/refrence.png exists.")
        else:
            reference = Image.open(reference_path).convert("L")
            ref_array = np.array(reference)

            sample_pixels = count_white_pixels(result_img)
            img_height, img_width = result_img.shape
            image_area = img_height * img_width
            white_density = sample_pixels / image_area  # value between 0 and 1

            st.success(f"Sample White Pixels: {sample_pixels}\nImage Area: {image_area}\nWhite Pixel Density: {white_density:.4f}")

            # Step 4: Allocate green time
            if white_density >= 0.18:
                st.info("🚦 Traffic is very high. Green signal time: 60 seconds")
            elif white_density >= 0.14:
                st.info("🚦 Traffic is high. Green signal time: 50 seconds")
            elif white_density >= 0.10:
                st.info("🚦 Traffic is moderate. Green signal time: 40 seconds")
            elif white_density >= 0.06:
                st.info("🚦 Traffic is low. Green signal time: 30 seconds")
            else:
                st.info("🚦 Traffic is very low. Green signal time: 20 seconds")






















????????????????????????????????????//
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import torch
from torchvision import transforms
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from CannyEdgeDetector import *

# Load model
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
model = ssdlite320_mobilenet_v3_large(weights=weights)
model.eval()

st.title("🚦 Density-Based Smart Traffic Control System")

uploaded_file = st.file_uploader("📤 Upload a Traffic Image", type=["jpg", "jpeg", "png"])

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def apply_canny(image):
    gray_img = rgb2gray(np.array(image))
    edge = CannyEdgeDetector([gray_img], sigma=1.4, kernel_size=5, lowthreshold=0.09, highthreshold=0.20, weak_pixel=100)
    processed = edge.detect()[0]
    return (processed / processed.max() * 255).astype(np.uint8)

def count_white_pixels(img):
    return np.sum(img == 255)

def detect_vehicles(image):
    img_rgb = np.array(image.convert("RGB"))
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img_rgb).unsqueeze(0)

    with torch.no_grad():
        prediction = model(img_tensor)

    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']
    found = False

    for i in range(len(labels)):
        class_id = labels[i].item()
        score = scores[i].item()

        if class_id in [3, 4, 6, 8] and score > 0.85:
            x1, y1, x2, y2 = boxes[i].cpu().numpy().astype(int)
            width, height = x2 - x1, y2 - y1
            if width > 100 and height > 100:
                found = True
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = "Ambulance" if COCO_INSTANCE_CATEGORY_NAMES[class_id] in ["bus", "truck"] else COCO_INSTANCE_CATEGORY_NAMES[class_id]
                cv2.putText(img_rgb, f'{label} ({score:.2f})', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return img_rgb, found

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    image = Image.open(uploaded_file)

    # Step 1 - Ambulance detection
    if st.button("🚑 Detect Ambulance"):
        result_img, found = detect_vehicles(image)
        st.image(result_img, caption="Detection Output", use_column_width=True)
        if found:
            st.success("🚑 Ambulance detected! Green signal time: 60 seconds or until it passes.")
            st.session_state['ambulance_detected'] = True
        else:
            st.warning("❌ No ambulance found.")
            st.session_state['ambulance_detected'] = False

    # Step 2 - Only allow next steps if ambulance is not found
    if 'ambulance_detected' in st.session_state and not st.session_state['ambulance_detected']:
        if st.button("⚙️ Run Canny Edge Detection"):
            canny_img = apply_canny(image)
            st.image(canny_img, caption="Canny Edge Output", use_column_width=True)
            st.session_state['canny_output'] = canny_img

        if st.button("📊 Analyze Traffic & Allocate Green Time"):
            if 'canny_output' not in st.session_state:
                st.warning("⚠️ Please run Canny Edge Detection first.")
            else:
                result_img = st.session_state['canny_output']
                sample_pixels = count_white_pixels(result_img)
                img_height, img_width = result_img.shape
                image_area = img_height * img_width
                white_density = sample_pixels / image_area

                st.success(f"Sample White Pixels: {sample_pixels}\nImage Area: {image_area}\nWhite Pixel Density: {white_density:.4f}")

                if white_density >= 0.18:
                    st.info("🚦 Traffic is very high. Green signal time: 60 seconds")
                elif white_density >= 0.14:
                    st.info("🚦 Traffic is high. Green signal time: 50 seconds")
                elif white_density >= 0.10:
                    st.info("🚦 Traffic is moderate. Green signal time: 40 seconds")
                elif white_density >= 0.06:
                    st.info("🚦 Traffic is low. Green signal time: 30 seconds")
                else:
                    st.info("🚦 Traffic is very low. Green signal time: 20 seconds")














??????????????/sidebar
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import torch
import time
from torchvision import transforms
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from CannyEdgeDetector import *

# Load model + COCO labels
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
model = ssdlite320_mobilenet_v3_large(weights=weights)
model.eval()

# UI
st.set_page_config(page_title="Smart Traffic Control", layout="centered")
st.title("🚦 Density-Based Smart Traffic Control System")
st.sidebar.title("📋 Console Logs")

uploaded_file = st.file_uploader("📤 Upload a Traffic Image", type=["jpg", "jpeg", "png"])

# --- Helper Functions ---
def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b

def apply_canny(image):
    gray_img = rgb2gray(np.array(image))
    edge = CannyEdgeDetector([gray_img], sigma=1.4, kernel_size=5, lowthreshold=0.09, highthreshold=0.20, weak_pixel=100)
    processed = edge.detect()[0]
    return (processed / processed.max() * 255).astype(np.uint8)

def count_white_pixels(img):
    return np.sum(img == 255)

def detect_vehicles(image):
    img_rgb = np.array(image.convert("RGB"))
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img_rgb).unsqueeze(0)

    with torch.no_grad():
        prediction = model(img_tensor)

    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']
    found = False

    for i in range(len(labels)):
        class_id = labels[i].item()
        score = scores[i].item()

        if class_id in [3, 4, 6, 8] and score > 0.85:
            x1, y1, x2, y2 = boxes[i].cpu().numpy().astype(int)
            width, height = x2 - x1, y2 - y1
            if width > 100 and height > 100:
                found = True
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = "Ambulance" if COCO_INSTANCE_CATEGORY_NAMES[class_id] in ["bus", "truck"] else COCO_INSTANCE_CATEGORY_NAMES[class_id]
                cv2.putText(img_rgb, f'{label} ({score:.2f})', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return img_rgb, found

# --- Main App Logic ---
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    image = Image.open(uploaded_file)

    # Detect Ambulance
    if st.button("🚑 Detect Ambulance"):
        with st.spinner("Detecting vehicles..."):
            result_img, found = detect_vehicles(image)
            st.image(result_img, caption="Detection Output", use_column_width=True)
            st.sidebar.write("✅ Object detection complete.")

        if found:
            st.success("🚑 Ambulance detected! Green signal time: 60 seconds or until it passes.")
            st.session_state['ambulance_detected'] = True
            st.sidebar.success("Ambulance detected — override density logic.")
        else:
            st.warning("❌ No ambulance found.")
            st.session_state['ambulance_detected'] = False
            st.sidebar.info("No ambulance — continue to density analysis.")

    # Step 2 - If ambulance not detected
    if 'ambulance_detected' in st.session_state and not st.session_state['ambulance_detected']:
        if st.button("⚙️ Run Canny Edge Detection"):
            with st.spinner("Processing Canny Edge Detection..."):
                canny_img = apply_canny(image)
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.005)
                    progress.progress(i + 1)
                st.image(canny_img, caption="Canny Edge Output", use_column_width=True)
                st.session_state['canny_output'] = canny_img
                st.sidebar.success("Canny edges extracted successfully.")

        if st.button("📊 Analyze Traffic & Allocate Green Time"):
            if 'canny_output' not in st.session_state:
                st.warning("⚠️ Please run Canny Edge Detection first.")
            else:
                with st.spinner("Analyzing white pixel density..."):
                    result_img = st.session_state['canny_output']
                    sample_pixels = count_white_pixels(result_img)
                    img_height, img_width = result_img.shape
                    image_area = img_height * img_width
                    white_density = sample_pixels / image_area

                    st.success(f"Sample White Pixels: {sample_pixels}\nImage Area: {image_area}\nWhite Pixel Density: {white_density:.4f}")
                    st.sidebar.write(f"📏 White pixel density: {white_density:.4f}")

                    if white_density >= 0.18:
                        st.info("🚦 Traffic is very high. Green signal time: 60 seconds")
                    elif white_density >= 0.14:
                        st.info("🚦 Traffic is high. Green signal time: 50 seconds")
                    elif white_density >= 0.10:
                        st.info("🚦 Traffic is moderate. Green signal time: 40 seconds")
                    elif white_density >= 0.06:
                        st.info("🚦 Traffic is low. Green signal time: 30 seconds")
                    else:
                        st.info("🚦 Traffic is very low. Green signal time: 20 seconds")

# Sidebar for showing step-by-step status
st.sidebar.title("Process Status")

if st.session_state['uploaded_image'] is None:
    st.sidebar.write("❌ No image uploaded.")
else:
    st.sidebar.write("✅ Image uploaded.")

if st.session_state['ambulance_detected'] is None:
    st.sidebar.write("🚑 Ambulance Detection: Not done yet.")
elif st.session_state['ambulance_detected'] is True:
    st.sidebar.success("🚑 Ambulance detected!")
else:
    st.sidebar.info("🚑 No ambulance detected.")

if st.session_state['canny_output'] is None:
    st.sidebar.write("⚙️ Canny Edge Detection: Not done yet.")
else:
    st.sidebar.success("⚙️ Canny Edge Detection completed.")

if st.session_state['traffic_density'] is None:
    st.sidebar.write("📊 Traffic Analysis: Not done yet.")
else:
    dens = st.session_state['traffic_density']
    st.sidebar.success(f"📊 Traffic Density: {dens:.4f}")







???????????????????????????????/done with side bar????????????????????/
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
import time
from torchvision import transforms
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from CannyEdgeDetector import *

# Load model + COCO labels
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
model = ssdlite320_mobilenet_v3_large(weights=weights)
model.eval()

# Initialize session state variables
if 'ambulance_detected' not in st.session_state:
    st.session_state['ambulance_detected'] = None

if 'canny_output' not in st.session_state:
    st.session_state['canny_output'] = None

if 'traffic_density' not in st.session_state:
    st.session_state['traffic_density'] = None

if 'uploaded_image' not in st.session_state:
    st.session_state['uploaded_image'] = None

# UI
st.set_page_config(page_title="Smart Traffic Control", layout="centered")
st.title("🚦 Density-Based Smart Traffic Control System")
st.sidebar.title("📋 Console Logs")

uploaded_file = st.file_uploader("📤 Upload a Traffic Image", type=["jpg", "jpeg", "png"])

# --- Helper Functions ---
def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b

def apply_canny(image):
    gray_img = rgb2gray(np.array(image))
    edge = CannyEdgeDetector([gray_img], sigma=1.4, kernel_size=5, lowthreshold=0.09, highthreshold=0.20, weak_pixel=100)
    processed = edge.detect()[0]
    return (processed / processed.max() * 255).astype(np.uint8)

def count_white_pixels(img):
    return np.sum(img == 255)

def detect_vehicles(image):
    img_rgb = np.array(image.convert("RGB"))
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img_rgb).unsqueeze(0)

    with torch.no_grad():
        prediction = model(img_tensor)

    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']
    found = False

    for i in range(len(labels)):
        class_id = labels[i].item()
        score = scores[i].item()

        if class_id in [3, 4, 6, 8] and score > 0.85:
            x1, y1, x2, y2 = boxes[i].cpu().numpy().astype(int)
            width, height = x2 - x1, y2 - y1
            if width > 100 and height > 100:
                found = True
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = "Ambulance" if COCO_INSTANCE_CATEGORY_NAMES[class_id] in ["bus", "truck"] else COCO_INSTANCE_CATEGORY_NAMES[class_id]
                cv2.putText(img_rgb, f'{label} ({score:.2f})', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return img_rgb, found

# --- Main App Logic ---
if uploaded_file:
    st.session_state['uploaded_image'] = Image.open(uploaded_file)
    st.image(st.session_state['uploaded_image'], caption="Uploaded Image", use_column_width=True)

if st.session_state['uploaded_image']:

    image = st.session_state['uploaded_image']

    # Detect Ambulance
    if st.button("🚑 Detect Ambulance"):
        with st.spinner("Detecting vehicles..."):
            result_img, found = detect_vehicles(image)
            st.image(result_img, caption="Detection Output", use_column_width=True)
            st.sidebar.write("✅ Object detection complete.")

        if found:
            st.success("🚑 Ambulance detected! Green signal time: 60 seconds or until it passes.")
            st.session_state['ambulance_detected'] = True
            st.sidebar.success("Ambulance detected — override density logic.")
            # Reset downstream steps
            st.session_state['canny_output'] = None
            st.session_state['traffic_density'] = None
        else:
            st.warning("❌ No ambulance found.")
            st.session_state['ambulance_detected'] = False
            st.sidebar.info("No ambulance — continue to density analysis.")
            # Reset downstream steps
            st.session_state['canny_output'] = None
            st.session_state['traffic_density'] = None

    # Step 2 - If ambulance not detected
    if st.session_state['ambulance_detected'] == False:

        if st.button("⚙️ Run Canny Edge Detection"):
            with st.spinner("Processing Canny Edge Detection..."):
                canny_img = apply_canny(image)
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.005)
                    progress.progress(i + 1)
                st.image(canny_img, caption="Canny Edge Output", use_column_width=True)
                st.session_state['canny_output'] = canny_img
                st.sidebar.success("Canny edges extracted successfully.")

        if st.button("📊 Analyze Traffic & Allocate Green Time"):
            if st.session_state['canny_output'] is None:
                st.warning("⚠️ Please run Canny Edge Detection first.")
            else:
                with st.spinner("Analyzing white pixel density..."):
                    result_img = st.session_state['canny_output']
                    sample_pixels = count_white_pixels(result_img)
                    img_height, img_width = result_img.shape
                    image_area = img_height * img_width
                    white_density = sample_pixels / image_area
                    st.session_state['traffic_density'] = white_density

                    st.success(f"Sample White Pixels: {sample_pixels}\nImage Area: {image_area}\nWhite Pixel Density: {white_density:.4f}")
                    st.sidebar.write(f"📏 White pixel density: {white_density:.4f}")

                    if white_density >= 0.18:
                        st.info("🚦 Traffic is very high. Green signal time: 60 seconds")
                    elif white_density >= 0.14:
                        st.info("🚦 Traffic is high. Green signal time: 50 seconds")
                    elif white_density >= 0.10:
                        st.info("🚦 Traffic is moderate. Green signal time: 40 seconds")
                    elif white_density >= 0.06:
                        st.info("🚦 Traffic is low. Green signal time: 30 seconds")
                    else:
                        st.info("🚦 Traffic is very low. Green signal time: 20 seconds")

# Sidebar for showing step-by-step status
st.sidebar.title("Process Status")

if st.session_state['uploaded_image'] is None:
    st.sidebar.write("❌ No image uploaded.")
else:
    st.sidebar.write("✅ Image uploaded.")

if st.session_state['ambulance_detected'] is None:
    st.sidebar.write("🚑 Ambulance Detection: Not done yet.")
elif st.session_state['ambulance_detected'] is True:
    st.sidebar.success("🚑 Ambulance detected!")
else:
    st.sidebar.info("🚑 No ambulance detected.")

if st.session_state['canny_output'] is None:
    st.sidebar.write("⚙️ Canny Edge Detection: Not done yet.")
else:
    st.sidebar.success("⚙️ Canny Edge Detection completed.")

if st.session_state['traffic_density'] is None:
    st.sidebar.write("📊 Traffic Analysis: Not done yet.")
else:
    dens = st.session_state['traffic_density']
    st.sidebar.success(f"📊 Traffic Density: {dens:.4f}")

???????????????????/final 1 side bar refresh akll done ?????
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
import time
from torchvision import transforms
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from CannyEdgeDetector import *

# Load model + COCO labels
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
model = ssdlite320_mobilenet_v3_large(weights=weights)
model.eval()

# Initialize session state variables
if 'ambulance_detected' not in st.session_state:
    st.session_state['ambulance_detected'] = None

if 'canny_output' not in st.session_state:
    st.session_state['canny_output'] = None

if 'traffic_density' not in st.session_state:
    st.session_state['traffic_density'] = None

if 'uploaded_image' not in st.session_state:
    st.session_state['uploaded_image'] = None

if 'uploaded_file_raw' not in st.session_state:
    st.session_state['uploaded_file_raw'] = None

# UI
st.set_page_config(page_title="Smart Traffic Control", layout="centered")
st.title("🚦 Density-Based Smart Traffic Control System")
st.sidebar.title("📋 Console Logs")

uploaded_file = st.file_uploader("📤 Upload a Traffic Image", type=["jpg", "jpeg", "png"])

# --- Reset logic on new upload or removal ---
if uploaded_file != st.session_state.get('uploaded_file_raw'):
    # Reset all related session state variables
    st.session_state['ambulance_detected'] = None
    st.session_state['canny_output'] = None
    st.session_state['traffic_density'] = None
    st.session_state['uploaded_image'] = None

    if uploaded_file is not None:
        st.session_state['uploaded_image'] = Image.open(uploaded_file)

    st.session_state['uploaded_file_raw'] = uploaded_file

# --- Helper Functions ---
def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b

def apply_canny(image):
    gray_img = rgb2gray(np.array(image))
    edge = CannyEdgeDetector([gray_img], sigma=1.4, kernel_size=5, lowthreshold=0.09, highthreshold=0.20, weak_pixel=100)
    processed = edge.detect()[0]
    return (processed / processed.max() * 255).astype(np.uint8)

def count_white_pixels(img):
    return np.sum(img == 255)

def detect_vehicles(image):
    img_rgb = np.array(image.convert("RGB"))
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img_rgb).unsqueeze(0)

    with torch.no_grad():
        prediction = model(img_tensor)

    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']
    found = False

    for i in range(len(labels)):
        class_id = labels[i].item()
        score = scores[i].item()

        if class_id in [3, 4, 6, 8] and score > 0.85:
            x1, y1, x2, y2 = boxes[i].cpu().numpy().astype(int)
            width, height = x2 - x1, y2 - y1
            if width > 100 and height > 100:
                found = True
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = "Ambulance" if COCO_INSTANCE_CATEGORY_NAMES[class_id] in ["bus", "truck"] else COCO_INSTANCE_CATEGORY_NAMES[class_id]
                cv2.putText(img_rgb, f'{label} ({score:.2f})', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return img_rgb, found

# --- Main App Logic ---
if st.session_state['uploaded_image']:
    image = st.session_state['uploaded_image']
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Detect Ambulance
    if st.button("🚑 Detect Ambulance"):
        with st.spinner("Detecting vehicles..."):
            result_img, found = detect_vehicles(image)
            st.image(result_img, caption="Detection Output", use_column_width=True)
            st.sidebar.write("✅ Object detection complete.")

        if found:
            st.success("🚑 Ambulance detected! Green signal time: 60 seconds or until it passes.")
            st.session_state['ambulance_detected'] = True
            st.sidebar.success("Ambulance detected — override density logic.")
            st.session_state['canny_output'] = None
            st.session_state['traffic_density'] = None
        else:
            st.warning("❌ No ambulance found.")
            st.session_state['ambulance_detected'] = False
            st.sidebar.info("No ambulance — continue to density analysis.")
            st.session_state['canny_output'] = None
            st.session_state['traffic_density'] = None

    # Step 2 - If ambulance not detected
    if st.session_state['ambulance_detected'] == False:

        if st.button("⚙️ Run Canny Edge Detection"):
            with st.spinner("Processing Canny Edge Detection..."):
                canny_img = apply_canny(image)
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.005)
                    progress.progress(i + 1)
                st.image(canny_img, caption="Canny Edge Output", use_column_width=True)
                st.session_state['canny_output'] = canny_img
                st.sidebar.success("Canny edges extracted successfully.")

        if st.button("📊 Analyze Traffic & Allocate Green Time"):
            if st.session_state['canny_output'] is None:
                st.warning("⚠️ Please run Canny Edge Detection first.")
            else:
                with st.spinner("Analyzing white pixel density..."):
                    result_img = st.session_state['canny_output']
                    sample_pixels = count_white_pixels(result_img)
                    img_height, img_width = result_img.shape
                    image_area = img_height * img_width
                    white_density = sample_pixels / image_area
                    st.session_state['traffic_density'] = white_density

                    st.success(f"Sample White Pixels: {sample_pixels}\nImage Area: {image_area}\nWhite Pixel Density: {white_density:.4f}")
                    st.sidebar.write(f"📏 White pixel density: {white_density:.4f}")

                    if white_density >= 0.18:
                        st.info("🚦 Traffic is very high. Green signal time: 60 seconds")
                    elif white_density >= 0.14:
                        st.info("🚦 Traffic is high. Green signal time: 50 seconds")
                    elif white_density >= 0.10:
                        st.info("🚦 Traffic is moderate. Green signal time: 40 seconds")
                    elif white_density >= 0.06:
                        st.info("🚦 Traffic is low. Green signal time: 30 seconds")
                    else:
                        st.info("🚦 Traffic is very low. Green signal time: 20 seconds")

# Sidebar Status Summary
st.sidebar.title("Process Status")

if st.session_state['uploaded_image'] is None:
    st.sidebar.write("❌ No image uploaded.")
else:
    st.sidebar.write("✅ Image uploaded.")

if st.session_state['ambulance_detected'] is None:
    st.sidebar.write("🚑 Ambulance Detection: Not done yet.")
elif st.session_state['ambulance_detected'] is True:
    st.sidebar.success("🚑 Ambulance detected!")
else:
    st.sidebar.info("🚑 No ambulance detected.")

if st.session_state['canny_output'] is None:
    st.sidebar.write("⚙️ Canny Edge Detection: Not done yet.")
else:
    st.sidebar.success("⚙️ Canny Edge Detection completed.")

if st.session_state['traffic_density'] is None:
    st.sidebar.write("📊 Traffic Analysis: Not done yet.")
else:
    dens = st.session_state['traffic_density']
    st.sidebar.success(f"📊 Traffic Density: {dens:.4f}")


# --- Add How To Use Info Panel ---
with st.sidebar.expander("ℹ️ How to Use This App"):
    st.markdown("""
    - **Upload Image:** Upload a traffic image to analyze.
    - **Detect Ambulance:** Detect if any ambulance is present in the image.
    - **Canny Edge Detection:** Highlights edges to estimate traffic density.
    - **Traffic Density:** Based on white pixel density, shows traffic load and green light timing.

    **Traffic Density Meaning:**
    - Very High (≥ 0.18): Heavy traffic — longer green light.
    - High (≥ 0.14): High traffic — moderate green light.
    - Moderate (≥ 0.10): Normal traffic.
    - Low (≥ 0.06): Light traffic.
    - Very Low (< 0.06): Very light traffic — shortest green light.

    **Ambulance Detected:** Overrides traffic density logic to allow faster passage.
    """)
if st.button("🔄 Reset All"):
    st.session_state.clear()
    st.rerun()  # Updated from st.experimental_rerun()














































































import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
import time
from torchvision import transforms
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from CannyEdgeDetector import *
import pytesseract
import easyocr
import difflib

# Set path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize EasyOCR
easyocr_reader = easyocr.Reader(['en'], gpu=False)

# Load model
COCO_INSTANCE_CATEGORY_NAMES = [...]  # Same as before
weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
model = ssdlite320_mobilenet_v3_large(weights=weights)
model.eval()

# Utility Functions
def is_ambulance_text(text):
    text = text.upper()
    words = text.split()
    for word in words:
        word_clean = ''.join(filter(str.isalpha, word))
        if 5 <= len(word_clean) <= 10:
            if difflib.get_close_matches(word_clean, ["AMBULANCE"], cutoff=0.7):
                return True
    return False

def preprocess_for_ocr(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    thresh = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    return thresh

def ocr_pytesseract(image):
    return pytesseract.image_to_string(image, config=r'--oem 3 --psm 6')

def ocr_easyocr(image, conf_threshold=0.3):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = easyocr_reader.readtext(rgb)
    filtered_texts = [res[1] for res in results if res[2] > conf_threshold]
    return " ".join(filtered_texts)

def preprocess_and_ocr_all_rotations(crop):
    angles = [0, 90, 180, 270]
    pytess_texts = []
    easyocr_texts = []
    for angle in angles:
        rotated = crop if angle == 0 else cv2.warpAffine(
            crop, cv2.getRotationMatrix2D((crop.shape[1]//2, crop.shape[0]//2), angle, 1.0),
            (crop.shape[1], crop.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        processed = preprocess_for_ocr(rotated)
        text_pt = ocr_pytesseract(processed).strip()
        pytess_texts.append(text_pt)
        text_eo = ocr_easyocr(rotated).strip()
        easyocr_texts.append(text_eo)

        if is_ambulance_text(text_pt) or is_ambulance_text(text_eo):
            return text_pt + " | " + text_eo

    return " ".join(pytess_texts + easyocr_texts)

# Streamlit setup
st.set_page_config(page_title="Smart Traffic Control", layout="centered")
st.title("🚦 Density-Based Smart Traffic Control System")
st.sidebar.title("📋 Console Logs")

# Session state
for key in ['ambulance_detected', 'canny_output', 'traffic_density', 'uploaded_image', 'uploaded_file_raw']:
    if key not in st.session_state:
        st.session_state[key] = None

# File uploader
uploaded_file = st.file_uploader("📤 Upload a Traffic Image", type=["jpg", "jpeg", "png"])
if uploaded_file != st.session_state['uploaded_file_raw']:
    st.session_state.update({
        'ambulance_detected': None,
        'canny_output': None,
        'traffic_density': None,
        'uploaded_image': Image.open(uploaded_file) if uploaded_file else None,
        'uploaded_file_raw': uploaded_file
    })

# Core functions
def apply_canny(image):
    gray_img = np.dot(np.array(image)[..., :3], [0.2989, 0.5870, 0.1140])
    edge = CannyEdgeDetector([gray_img], sigma=1.4, kernel_size=5, lowthreshold=0.09, highthreshold=0.20, weak_pixel=100)
    processed = edge.detect()[0]
    return (processed / processed.max() * 255).astype(np.uint8)

def count_white_pixels(img):
    return np.sum(img == 255)

def detect_vehicles_with_ocr(image):
    img_rgb = np.array(image.convert("RGB"))
    img_tensor = transforms.ToTensor()(img_rgb).unsqueeze(0)
    with torch.no_grad():
        prediction = model(img_tensor)

    found = False
    ambulance_text_detected = ""
    for i in range(len(prediction[0]['labels'])):
        label = prediction[0]['labels'][i].item()
        score = prediction[0]['scores'][i].item()
        if label in [3, 6, 8] and score > 0.3:
            box = prediction[0]['boxes'][i].cpu().numpy().astype(int)
            crop = img_rgb[box[1]:box[3], box[0]:box[2]]
            text = preprocess_and_ocr_all_rotations(crop)
            if is_ambulance_text(text):
                found = True
                ambulance_text_detected = text
                cv2.rectangle(img_rgb, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(img_rgb, "AMBULANCE", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    return img_rgb, found, ambulance_text_detected

# Main interaction
if st.session_state['uploaded_image']:
    image = st.session_state['uploaded_image']
    st.image(image, caption="📸 Uploaded Image", use_column_width=True)

    if st.button("🚑 Detect Ambulance"):
        with st.spinner("Analyzing vehicles and text..."):
            result_img, found, ambulance_text = detect_vehicles_with_ocr(image)
            st.image(result_img, caption="🚓 Detection Output", use_column_width=True)
            st.sidebar.write("✅ Detection complete.")

        if found:
            st.success("🚨 **Ambulance Detected!** Green signal time: 60 seconds.")
            st.markdown(f"""<div style='padding:10px; background-color:#d1f0d1; border-left:5px solid green'>
                <b>🔍 OCR Detected Text:</b><br><span style='font-size:18px'>{ambulance_text}</span></div>
                """, unsafe_allow_html=True)
            st.session_state['ambulance_detected'] = True
        else:
            st.warning("❌ No ambulance found.")
            st.session_state['ambulance_detected'] = False

    if st.session_state['ambulance_detected'] is False:
        if st.button("⚙️ Run Canny Edge Detection"):
            with st.spinner("Detecting edges..."):
                canny_img = apply_canny(image)
                for i in range(100): time.sleep(0.005); st.progress(i + 1)
                st.image(canny_img, caption="🖼️ Canny Edge Output", use_column_width=True)
                st.session_state['canny_output'] = canny_img

        if st.button("📊 Analyze Traffic & Allocate Green Time"):
            if st.session_state['canny_output'] is None:
                st.warning("⚠️ Please run Canny Edge Detection first.")
            else:
                with st.spinner("Calculating white pixel density..."):
                    img = st.session_state['canny_output']
                    white_pixels = count_white_pixels(img)
                    img_area = img.shape[0] * img.shape[1]
                    density = white_pixels / img_area
                    st.session_state['traffic_density'] = density

                    # Visual feedback
                    if density >= 0.18:
                        level, time_sec, color = "🚗 Very High", 60, "#ffebee"
                    elif density >= 0.14:
                        level, time_sec, color = "🚙 High", 50, "#fff3e0"
                    elif density >= 0.10:
                        level, time_sec, color = "🚕 Moderate", 40, "#f0f4c3"
                    elif density >= 0.06:
                        level, time_sec, color = "🚘 Low", 30, "#e0f7fa"
                    else:
                        level, time_sec, color = "🛵 Very Low", 20, "#f3e5f5"

                    st.metric(label="📈 Traffic Density", value=f"{density:.4f}", delta=level)
                    st.metric(label="⏱️ Green Light Time", value=f"{time_sec} seconds")

                    st.markdown(f"""
                        <div style='padding:12px; background-color:{color}; border-left:6px solid #1976d2;'>
                            <b>📌 White Pixels:</b> {white_pixels}<br>
                            <b>📐 Image Area:</b> {img_area}<br>
                            <b>🚦 Density:</b> {density:.4f} ({level})<br>
                            <b>🟢 Green Time:</b> {time_sec} seconds
                        </div>
                    """, unsafe_allow_html=True)

# Sidebar Summary
st.sidebar.write("✅ Image uploaded." if st.session_state['uploaded_image'] else "❌ No image uploaded.")
if st.session_state['ambulance_detected'] is None:
    st.sidebar.write("🚑 Ambulance Detection: Not yet run.")
elif st.session_state['ambulance_detected']:
    st.sidebar.success("🚑 Ambulance Detected!")
else:
    st.sidebar.info("🚑 No Ambulance Detected.")

st.sidebar.write("⚙️ Canny Edge Detection: Done." if st.session_state['canny_output'] is not None else "⚙️ Not done.")
st.sidebar.write("📊 Traffic Analysis: Done." if st.session_state['traffic_density'] is not None else "📊 Not done.")

# Help
with st.sidebar.expander("ℹ️ How to Use This App"):
    st.markdown("""
    - **Upload Image:** Upload a traffic scene image.
    - **Ambulance Detection:** Identifies ambulance using object detection and OCR.
    - **Canny Edge:** Runs edge detection to measure traffic.
    - **Density Analysis:** Computes white pixel ratio to allocate green signal time.
    """)

# Reset
if st.button("🔄 Reset All"):
    st.session_state.clear()
    st.rerun()













































































































import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
import time
from torchvision import transforms
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from CannyEdgeDetector import *
import pytesseract
import easyocr
import difflib

# Set path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize EasyOCR
easyocr_reader = easyocr.Reader(['en'], gpu=False)

# Load model
COCO_INSTANCE_CATEGORY_NAMES = [...]  # Same as before
weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
model = ssdlite320_mobilenet_v3_large(weights=weights)
model.eval()

# Utility Functions
def is_ambulance_text(text):
    text = text.upper()
    words = text.split()
    for word in words:
        word_clean = ''.join(filter(str.isalpha, word))
        if 5 <= len(word_clean) <= 10:
            if difflib.get_close_matches(word_clean, ["AMBULANCE"], cutoff=0.7):
                return True
    return False

def preprocess_for_ocr(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    thresh = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    return thresh

def ocr_pytesseract(image):
    return pytesseract.image_to_string(image, config=r'--oem 3 --psm 6')

def ocr_easyocr(image, conf_threshold=0.3):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = easyocr_reader.readtext(rgb)
    filtered_texts = [res[1] for res in results if res[2] > conf_threshold]
    return " ".join(filtered_texts)

def preprocess_and_ocr_all_rotations(crop):
    angles = [0, 90, 180, 270]
    pytess_texts = []
    easyocr_texts = []
    for angle in angles:
        rotated = crop if angle == 0 else cv2.warpAffine(
            crop, cv2.getRotationMatrix2D((crop.shape[1]//2, crop.shape[0]//2), angle, 1.0),
            (crop.shape[1], crop.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        processed = preprocess_for_ocr(rotated)
        text_pt = ocr_pytesseract(processed).strip()
        pytess_texts.append(text_pt)
        text_eo = ocr_easyocr(rotated).strip()
        easyocr_texts.append(text_eo)

        if is_ambulance_text(text_pt) or is_ambulance_text(text_eo):
            return text_pt + " | " + text_eo

    return " ".join(pytess_texts + easyocr_texts)

# Streamlit setup
st.set_page_config(page_title="Smart Traffic Control", layout="centered")
st.title("🚦 Density-Based Smart Traffic Control System")
st.sidebar.title("📋 Console Logs")

# Session state
for key in ['ambulance_detected', 'canny_output', 'traffic_density', 'uploaded_image', 'uploaded_file_raw']:
    if key not in st.session_state:
        st.session_state[key] = None

# File uploader
uploaded_file = st.file_uploader("📤 Upload a Traffic Image", type=["jpg", "jpeg", "png"])
if uploaded_file != st.session_state['uploaded_file_raw']:
    st.session_state.update({
        'ambulance_detected': None,
        'canny_output': None,
        'traffic_density': None,
        'uploaded_image': Image.open(uploaded_file) if uploaded_file else None,
        'uploaded_file_raw': uploaded_file
    })

# Core functions
def apply_canny(image):
    gray_img = np.dot(np.array(image)[..., :3], [0.2989, 0.5870, 0.1140])
    edge = CannyEdgeDetector([gray_img], sigma=1.4, kernel_size=5, lowthreshold=0.09, highthreshold=0.20, weak_pixel=100)
    processed = edge.detect()[0]
    return (processed / processed.max() * 255).astype(np.uint8)

def count_white_pixels(img):
    return np.sum(img == 255)

def detect_vehicles_with_ocr(image):
    img_rgb = np.array(image.convert("RGB"))
    img_tensor = transforms.ToTensor()(img_rgb).unsqueeze(0)
    with torch.no_grad():
        prediction = model(img_tensor)

    found = False
    ambulance_text_detected = ""
    for i in range(len(prediction[0]['labels'])):
        label = prediction[0]['labels'][i].item()
        score = prediction[0]['scores'][i].item()
        if label in [3, 6, 8] and score > 0.3:
            box = prediction[0]['boxes'][i].cpu().numpy().astype(int)
            crop = img_rgb[box[1]:box[3], box[0]:box[2]]
            text = preprocess_and_ocr_all_rotations(crop)
            if is_ambulance_text(text):
                found = True
                ambulance_text_detected = text
                cv2.rectangle(img_rgb, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(img_rgb, "AMBULANCE", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    return img_rgb, found, ambulance_text_detected

# Main interaction
if st.session_state['uploaded_image']:
    image = st.session_state['uploaded_image']
    st.image(image, caption="📸 Uploaded Image", use_column_width=True)

    if st.button("🚑 Detect Ambulance"):
        with st.spinner("Analyzing vehicles and text..."):
            result_img, found, ambulance_text = detect_vehicles_with_ocr(image)
            st.image(result_img, caption="🚓 Detection Output", use_column_width=True)
            st.sidebar.write("✅ Detection complete.")

        if found:
            st.success("🚨 **Ambulance Detected!** Green signal time: 60 seconds.")
            st.markdown(f"""<div style='padding:10px; background-color:#d1f0d1; border-left:5px solid green'>
                <b>🔍 OCR Detected Text:</b><br><span style='font-size:18px'>{ambulance_text}</span></div>
                """, unsafe_allow_html=True)
            st.session_state['ambulance_detected'] = True
        else:
            st.warning("❌ No ambulance found.")
            st.session_state['ambulance_detected'] = False

    if st.session_state['ambulance_detected'] is False:
        if st.button("⚙️ Run Canny Edge Detection"):
            progress_bar = st.progress(0, text="Canny Edge Detection in progress...")  # Initialize bar

            with st.spinner("Detecting edges..."):
            # Simulate progress
                for percent_complete in range(100):
                    time.sleep(0.01)  # Simulate work
                    progress_bar.progress(percent_complete + 1, text="Canny Edge Detection in progress...")

                canny_img = apply_canny(image)
                progress_bar.empty()  # Hide progress bar

                st.image(canny_img, caption="🖼️ Canny Edge Output", use_column_width=True)
                st.session_state['canny_output'] = canny_img




        if st.button("📊 Analyze Traffic & Allocate Green Time"):
            if st.session_state['canny_output'] is None:
                st.warning("⚠️ Please run Canny Edge Detection first.")
            else:
                with st.spinner("Calculating white pixel density..."):
                    img = st.session_state['canny_output']
                    white_pixels = count_white_pixels(img)
                    img_area = img.shape[0] * img.shape[1]
                    density = white_pixels / img_area
                    st.session_state['traffic_density'] = density

                    # Visual feedback
                    if density >= 0.18:
                        level, time_sec, color = "🚗 Very High", 60, "#ffebee"
                    elif density >= 0.14:
                        level, time_sec, color = "🚙 High", 50, "#fff3e0"
                    elif density >= 0.10:
                        level, time_sec, color = "🚕 Moderate", 40, "#f0f4c3"
                    elif density >= 0.06:
                        level, time_sec, color = "🚘 Low", 30, "#e0f7fa"
                    else:
                        level, time_sec, color = "🛵 Very Low", 20, "#f3e5f5"

                    st.metric(label="📈 Traffic Density", value=f"{density:.4f}", delta=level)
                    st.metric(label="⏱️ Green Light Time", value=f"{time_sec} seconds")

                    st.markdown(f"""
                        <div style='padding:12px; background-color:{color}; border-left:6px solid #1976d2;'>
                            <b>📌 White Pixels:</b> {white_pixels}<br>
                            <b>📐 Image Area:</b> {img_area}<br>
                            <b>🚦 Density:</b> {density:.4f} ({level})<br>
                            <b>🟢 Green Time:</b> {time_sec} seconds
                        </div>
                    """, unsafe_allow_html=True)

# Sidebar # Sidebar Status Summary
st.sidebar.title("Process Status")

if st.session_state['uploaded_image'] is None:
    st.sidebar.write("❌ No image uploaded.")
else:
    st.sidebar.write("✅ Image uploaded.")

if st.session_state['ambulance_detected'] is None:
    st.sidebar.write("🚑 Ambulance Detection: Not done yet.")
elif st.session_state['ambulance_detected'] is True:
    st.sidebar.success("🚑 Ambulance detected!")
else:
    st.sidebar.info("🚑 No ambulance detected.")

if st.session_state['canny_output'] is None:
    st.sidebar.write("⚙️ Canny Edge Detection: Not done yet.")
else:
    st.sidebar.success("⚙️ Canny Edge Detection completed.")

if st.session_state['traffic_density'] is None:
    st.sidebar.write("📊 Traffic Analysis: Not done yet.")
else:
    dens = st.session_state['traffic_density']
    st.sidebar.success(f"📊 Traffic Density: {dens:.4f}")

# --- Add How To Use Info Panel ---
with st.sidebar.expander("ℹ️ How to Use This App"):
    st.markdown("""
    - **Upload Image:** Upload a traffic image to analyze.
    - **Detect Ambulance:** Detect if any ambulance is present in the image using OCR and object detection.
    - **Canny Edge Detection:** Highlights edges to estimate traffic density.
    - **Traffic Density:** Based on white pixel density, shows traffic load and green light timing.

    **Traffic Density Meaning:**
    - Very High (≥ 0.18): Heavy traffic — longer green light.
    - High (≥ 0.14): High traffic — moderate green light.
    - Moderate (≥ 0.10): Normal traffic.
    - Low (≥ 0.06): Light traffic.
    - Very Low (< 0.06): Very light traffic — shortest green light.

    **Ambulance Detected:** Overrides traffic density logic to allow faster passage.
    """)

# Reset
if st.button("🔄 Reset All"):
    st.session_state.clear()
    st.rerun()



























































































    

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
import time
from torchvision import transforms
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from CannyEdgeDetector import *
import pytesseract
import easyocr
import difflib

# Set path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize EasyOCR
easyocr_reader = easyocr.Reader(['en'], gpu=False)

# Load model
COCO_INSTANCE_CATEGORY_NAMES = [...]  # Same as before
weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
model = ssdlite320_mobilenet_v3_large(weights=weights)
model.eval()
# Load the EAST text detector
east_net = cv2.dnn.readNet(r"C:\Users\Slnrockstones\Desktop\newmodel\frozen_east_text_detection.pb")


# Utility Functions
def is_ambulance_text(text):
    text = text.upper()
    words = text.split()
    for word in words:
        word_clean = ''.join(filter(str.isalpha, word))
        if 5 <= len(word_clean) <= 10:
            # Check normal orientation
            if difflib.get_close_matches(word_clean, ["AMBULANCE"], cutoff=0.7):
                return True
            # Check reversed orientation
            if difflib.get_close_matches(word_clean[::-1], ["AMBULANCE"], cutoff=0.7):
                return True
    return False

def preprocess_for_ocr(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    thresh = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    return thresh

def ocr_pytesseract(image):
    return pytesseract.image_to_string(image, config=r'--oem 3 --psm 6')

def ocr_easyocr(image, conf_threshold=0.3):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = easyocr_reader.readtext(rgb)
    filtered_texts = [res[1] for res in results if res[2] > conf_threshold]
    return " ".join(filtered_texts)

def detect_text_with_east_and_ocr(crop):
    detected_texts = []
    text_boxes = detect_text_regions_east(crop)
    print(f"[DEBUG] EAST detected {len(text_boxes)} text boxes.")
    for (startX, startY, endX, endY) in text_boxes:
        # Debug: Draw red rectangles on the original crop to visualize detected boxes
        cv2.rectangle(crop, (startX, startY), (endX, endY), (0, 0, 255), 2)


    for (startX, startY, endX, endY) in text_boxes:
        # Ensure box is within bounds
        startX, startY = max(0, startX), max(0, startY)
        endX, endY = min(crop.shape[1], endX), min(crop.shape[0], endY)
        roi = crop[startY:endY, startX:endX]

        processed = preprocess_for_ocr(roi)
        text_pt = ocr_pytesseract(processed).strip()
        text_eo = ocr_easyocr(roi).strip()

        combined = text_pt + " " + text_eo
        detected_texts.append(combined)

        if is_ambulance_text(combined) or is_ambulance_text(combined[::-1]):
            return combined

    return " ".join(detected_texts)

# Streamlit setup
st.set_page_config(page_title="Smart Traffic Control", layout="centered")
st.title("🚦 Density-Based Smart Traffic Control System")
st.sidebar.title("📋 Console Logs")

# Session state
for key in ['ambulance_detected', 'canny_output', 'traffic_density', 'uploaded_image', 'uploaded_file_raw']:
    if key not in st.session_state:
        st.session_state[key] = None

# File uploader
uploaded_file = st.file_uploader("📤 Upload a Traffic Image", type=["jpg", "jpeg", "png"])
if uploaded_file != st.session_state['uploaded_file_raw']:
    st.session_state.update({
        'ambulance_detected': None,
        'canny_output': None,
        'traffic_density': None,
        'uploaded_image': Image.open(uploaded_file) if uploaded_file else None,
        'uploaded_file_raw': uploaded_file
    })

# Core functions
def apply_canny(image):
    gray_img = np.dot(np.array(image)[..., :3], [0.2989, 0.5870, 0.1140])
    edge = CannyEdgeDetector([gray_img], sigma=1.4, kernel_size=5, lowthreshold=0.09, highthreshold=0.20, weak_pixel=100)
    processed = edge.detect()[0]
    return (processed / processed.max() * 255).astype(np.uint8)

def count_white_pixels(img):
    return np.sum(img == 255)
def decode_predictions(scores, geometry, min_confidence):
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(numCols):
            if scoresData[x] < min_confidence:
                continue

            offsetX = x * 4.0
            offsetY = y * 4.0

            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + cos * xData1[x] + sin * xData2[x])
            endY = int(offsetY - sin * xData1[x] + cos * xData2[x])
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, int(w), int(h)))
            confidences.append(float(scoresData[x]))

    return (rects, confidences)


def detect_text_regions_east(image, width=320, height=320, min_confidence=0.5):
    orig = image.copy()
    (H, W) = image.shape[:2]
    rW = W / float(width)
    rH = H / float(height)

    # Resize and prepare blob
    resized = cv2.resize(image, (width, height))
    blob = cv2.dnn.blobFromImage(resized, 1.0, (width, height),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    east_net.setInput(blob)
    (scores, geometry) = east_net.forward([
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"
    ])

    (boxes, confidences) = decode_predictions(scores, geometry, min_confidence)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)

    final_boxes = []
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y, w, h) = boxes[i]
            startX = int(x * rW)
            startY = int(y * rH)
            endX = int((x + w) * rW)
            endY = int((y + h) * rH)
            final_boxes.append((startX, startY, endX, endY))

    return final_boxes



def detect_vehicles_with_ocr(image):
    img_rgb = np.array(image.convert("RGB"))
    img_tensor = transforms.ToTensor()(img_rgb).unsqueeze(0)
    with torch.no_grad():
        prediction = model(img_tensor)

    found = False
    ambulance_text_detected = ""
    for i in range(len(prediction[0]['labels'])):
        label = prediction[0]['labels'][i].item()
        score = prediction[0]['scores'][i].item()
        if label in [3, 6, 8] and score > 0.3:
            box = prediction[0]['boxes'][i].cpu().numpy().astype(int)
            crop = img_rgb[box[1]:box[3], box[0]:box[2]]
            text = detect_text_with_east_and_ocr(crop)

            if is_ambulance_text(text):
                found = True
                ambulance_text_detected = text
                cv2.rectangle(img_rgb, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(img_rgb, "AMBULANCE", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    return img_rgb, found, ambulance_text_detected

# Main interaction
if st.session_state['uploaded_image']:
    image = st.session_state['uploaded_image']
    st.image(image, caption="📸 Uploaded Image", use_column_width=True)

    if st.button("🚑 Detect Ambulance"):
        with st.spinner("Analyzing vehicles and text..."):
            result_img, found, ambulance_text = detect_vehicles_with_ocr(image)
            st.image(result_img, caption="🚓 Detection Output", use_column_width=True)
            st.sidebar.write("✅ Detection complete.")

        if found:
            st.success("🚨 **Ambulance Detected!** Green signal time: 60 seconds.")
            st.markdown(f"""<div style='padding:10px; background-color:#d1f0d1; border-left:5px solid green'>
                <b>🔍 OCR Detected Text:</b><br><span style='font-size:18px'>{ambulance_text}</span></div>
                """, unsafe_allow_html=True)
            st.session_state['ambulance_detected'] = True
        else:
            st.warning("❌ No ambulance found.")
            st.session_state['ambulance_detected'] = False

    if st.session_state['ambulance_detected'] is False:
        if st.button("⚙️ Run Canny Edge Detection"):
            progress_bar = st.progress(0, text="Canny Edge Detection in progress...")  # Initialize bar

            with st.spinner("Detecting edges..."):
            # Simulate progress
                for percent_complete in range(100):
                    time.sleep(0.01)  # Simulate work
                    progress_bar.progress(percent_complete + 1, text="Canny Edge Detection in progress...")

                canny_img = apply_canny(image)
                progress_bar.empty()  # Hide progress bar

                st.image(canny_img, caption="🖼️ Canny Edge Output", use_column_width=True)
                st.session_state['canny_output'] = canny_img




        if st.button("📊 Analyze Traffic & Allocate Green Time"):
            if st.session_state['canny_output'] is None:
                st.warning("⚠️ Please run Canny Edge Detection first.")
            else:
                with st.spinner("Calculating white pixel density..."):
                    img = st.session_state['canny_output']
                    white_pixels = count_white_pixels(img)
                    img_area = img.shape[0] * img.shape[1]
                    density = white_pixels / img_area
                    st.session_state['traffic_density'] = density

                    # Visual feedback
                    if density >= 0.18:
                        level, time_sec, color = "🚗 Very High", 60, "#ffebee"
                    elif density >= 0.14:
                        level, time_sec, color = "🚙 High", 50, "#fff3e0"
                    elif density >= 0.10:
                        level, time_sec, color = "🚕 Moderate", 40, "#f0f4c3"
                    elif density >= 0.06:
                        level, time_sec, color = "🚘 Low", 30, "#e0f7fa"
                    else:
                        level, time_sec, color = "🛵 Very Low", 20, "#f3e5f5"

                    st.metric(label="📈 Traffic Density", value=f"{density:.4f}", delta=level)
                    st.metric(label="⏱️ Green Light Time", value=f"{time_sec} seconds")

                    st.markdown(f"""
                        <div style='padding:12px; background-color:{color}; border-left:6px solid #1976d2;'>
                            <b>📌 White Pixels:</b> {white_pixels}<br>
                            <b>📐 Image Area:</b> {img_area}<br>
                            <b>🚦 Density:</b> {density:.4f} ({level})<br>
                            <b>🟢 Green Time:</b> {time_sec} seconds
                        </div>
                    """, unsafe_allow_html=True)

# Sidebar # Sidebar Status Summary
st.sidebar.title("Process Status")

if st.session_state['uploaded_image'] is None:
    st.sidebar.write("❌ No image uploaded.")
else:
    st.sidebar.write("✅ Image uploaded.")

if st.session_state['ambulance_detected'] is None:
    st.sidebar.write("🚑 Ambulance Detection: Not done yet.")
elif st.session_state['ambulance_detected'] is True:
    st.sidebar.success("🚑 Ambulance detected!")
else:
    st.sidebar.info("🚑 No ambulance detected.")

if st.session_state['canny_output'] is None:
    st.sidebar.write("⚙️ Canny Edge Detection: Not done yet.")
else:
    st.sidebar.success("⚙️ Canny Edge Detection completed.")

if st.session_state['traffic_density'] is None:
    st.sidebar.write("📊 Traffic Analysis: Not done yet.")
else:
    dens = st.session_state['traffic_density']
    st.sidebar.success(f"📊 Traffic Density: {dens:.4f}")

# --- Add How To Use Info Panel ---
with st.sidebar.expander("ℹ️ How to Use This App"):
    st.markdown("""
    - **Upload Image:** Upload a traffic image to analyze.
    - **Detect Ambulance:** Detect if any ambulance is present in the image using OCR and object detection.
    - **Canny Edge Detection:** Highlights edges to estimate traffic density.
    - **Traffic Density:** Based on white pixel density, shows traffic load and green light timing.

    **Traffic Density Meaning:**
    - Very High (≥ 0.18): Heavy traffic — longer green light.
    - High (≥ 0.14): High traffic — moderate green light.
    - Moderate (≥ 0.10): Normal traffic.
    - Low (≥ 0.06): Light traffic.
    - Very Low (< 0.06): Very light traffic — shortest green light.

    **Ambulance Detected:** Overrides traffic density logic to allow faster passage.
    """)

# Reset
if st.button("🔄 Reset All"):
    st.session_state.clear()
    st.rerun()

















































BLUE BOX NEAR TEXT????????????????????????????



import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
import time
from torchvision import transforms
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from CannyEdgeDetector import *
import pytesseract
import easyocr
import difflib

# Set path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize EasyOCR
easyocr_reader = easyocr.Reader(['en'], gpu=False)

# Load model
COCO_INSTANCE_CATEGORY_NAMES = [...]  # Same as before
weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
model = ssdlite320_mobilenet_v3_large(weights=weights)
model.eval()
# Load the EAST text detector
east_net = cv2.dnn.readNet(r"C:\Users\Slnrockstones\Desktop\newmodel\frozen_east_text_detection.pb")


# Utility Functions
def is_ambulance_text(text):
    text = text.upper()
    words = text.split()
    for word in words:
        word_clean = ''.join(filter(str.isalpha, word))
        if 5 <= len(word_clean) <= 10:
            # Check normal and reversed
            if difflib.get_close_matches(word_clean, ["AMBULANCE"], cutoff=0.7):
                return True
            if difflib.get_close_matches(word_clean[::-1], ["AMBULANCE"], cutoff=0.7):
                return True
    return False

def preprocess_for_ocr(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    thresh = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    return thresh

def ocr_pytesseract(image):
    return pytesseract.image_to_string(image, config=r'--oem 3 --psm 6')

def ocr_easyocr(image, conf_threshold=0.3):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = easyocr_reader.readtext(rgb)
    filtered_texts = [res[1] for res in results if res[2] > conf_threshold]
    return " ".join(filtered_texts)

def detect_text_with_east_and_ocr(crop):
    detected_texts = []
    text_boxes = detect_text_regions_east(crop)
    print(f"[DEBUG] EAST detected {len(text_boxes)} text boxes.")
    
    for (startX, startY, endX, endY) in text_boxes:
        # Clip coordinates
        startX, startY = max(0, startX), max(0, startY)
        endX, endY = min(crop.shape[1], endX), min(crop.shape[0], endY)
        roi = crop[startY:endY, startX:endX]

        processed = preprocess_for_ocr(roi)
        text_pt = ocr_pytesseract(processed).strip()
        text_eo = ocr_easyocr(roi).strip()

        combined = (text_pt + " " + text_eo).strip()
        detected_texts.append(combined)

        print(f"[OCR DEBUG] pytesseract: '{text_pt}', easyocr: '{text_eo}', combined: '{combined}'")

        if is_ambulance_text(combined):
            return combined

    return " ".join(detected_texts)


# Streamlit setup
st.set_page_config(page_title="Smart Traffic Control", layout="centered")
st.title("🚦 Density-Based Smart Traffic Control System")
st.sidebar.title("📋 Console Logs")

# Session state
for key in ['ambulance_detected', 'canny_output', 'traffic_density', 'uploaded_image', 'uploaded_file_raw']:
    if key not in st.session_state:
        st.session_state[key] = None

# File uploader
uploaded_file = st.file_uploader("📤 Upload a Traffic Image", type=["jpg", "jpeg", "png"])
if uploaded_file != st.session_state['uploaded_file_raw']:
    st.session_state.update({
        'ambulance_detected': None,
        'canny_output': None,
        'traffic_density': None,
        'uploaded_image': Image.open(uploaded_file) if uploaded_file else None,
        'uploaded_file_raw': uploaded_file
    })

# Core functions
def apply_canny(image):
    gray_img = np.dot(np.array(image)[..., :3], [0.2989, 0.5870, 0.1140])
    edge = CannyEdgeDetector([gray_img], sigma=1.4, kernel_size=5, lowthreshold=0.09, highthreshold=0.20, weak_pixel=100)
    processed = edge.detect()[0]
    return (processed / processed.max() * 255).astype(np.uint8)

def count_white_pixels(img):
    return np.sum(img == 255)
def decode_predictions(scores, geometry, min_confidence):
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(numCols):
            if scoresData[x] < min_confidence:
                continue

            offsetX = x * 4.0
            offsetY = y * 4.0

            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + cos * xData1[x] + sin * xData2[x])
            endY = int(offsetY - sin * xData1[x] + cos * xData2[x])
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, int(w), int(h)))
            confidences.append(float(scoresData[x]))

    return (rects, confidences)


def detect_text_regions_east(image, width=320, height=320, min_confidence=0.5):
    orig = image.copy()
    (H, W) = image.shape[:2]
    rW = W / float(width)
    rH = H / float(height)

    # Resize and prepare blob
    resized = cv2.resize(image, (width, height))
    blob = cv2.dnn.blobFromImage(resized, 1.0, (width, height),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    east_net.setInput(blob)
    (scores, geometry) = east_net.forward([
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"
    ])

    (boxes, confidences) = decode_predictions(scores, geometry, min_confidence)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)

    final_boxes = []
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y, w, h) = boxes[i]
            startX = int(x * rW)
            startY = int(y * rH)
            endX = int((x + w) * rW)
            endY = int((y + h) * rH)
            final_boxes.append((startX, startY, endX, endY))

    return final_boxes



def detect_vehicles_with_ocr(image):
    img_rgb = np.array(image.convert("RGB"))
    img_tensor = transforms.ToTensor()(img_rgb).unsqueeze(0)
    with torch.no_grad():
        prediction = model(img_tensor)

    found = False
    ambulance_text_detected = ""
    for i in range(len(prediction[0]['labels'])):
        label = prediction[0]['labels'][i].item()
        score = prediction[0]['scores'][i].item()
        if label in [3, 6, 8] and score > 0.3:
            box = prediction[0]['boxes'][i].cpu().numpy().astype(int)
            crop = img_rgb[box[1]:box[3], box[0]:box[2]]
            text = detect_text_with_east_and_ocr(crop)

            if text.strip().lower() != "":

                found = True
                ambulance_text_detected = text
                cv2.rectangle(img_rgb, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(img_rgb, "AMBULANCE", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    return img_rgb, found, ambulance_text_detected

# Main interaction
if st.session_state['uploaded_image']:
    image = st.session_state['uploaded_image']
    st.image(image, caption="📸 Uploaded Image", use_column_width=True)

    if st.button("🚑 Detect Ambulance"):
        with st.spinner("Analyzing vehicles and text..."):
            result_img, found, ambulance_text = detect_vehicles_with_ocr(image)
            st.image(result_img, caption="🚓 Detection Output", use_column_width=True)
            st.sidebar.write("✅ Detection complete.")

        if found:
            st.success("🚨 **Ambulance Detected!** Green signal time: 60 seconds.")
            st.markdown(f"""<div style='padding:10px; background-color:#d1f0d1; border-left:5px solid green'>
                <b>🔍 OCR Detected Text:</b><br><span style='font-size:18px'>{ambulance_text}</span></div>
                """, unsafe_allow_html=True)
            st.session_state['ambulance_detected'] = True
        else:
            st.warning("❌ No ambulance found.")
            st.session_state['ambulance_detected'] = False

    if st.session_state['ambulance_detected'] is False:
        if st.button("⚙️ Run Canny Edge Detection"):
            progress_bar = st.progress(0, text="Canny Edge Detection in progress...")  # Initialize bar

            with st.spinner("Detecting edges..."):
            # Simulate progress
                for percent_complete in range(100):
                    time.sleep(0.01)  # Simulate work
                    progress_bar.progress(percent_complete + 1, text="Canny Edge Detection in progress...")

                canny_img = apply_canny(image)
                progress_bar.empty()  # Hide progress bar

                st.image(canny_img, caption="🖼️ Canny Edge Output", use_column_width=True)
                st.session_state['canny_output'] = canny_img




        if st.button("📊 Analyze Traffic & Allocate Green Time"):
            if st.session_state['canny_output'] is None:
                st.warning("⚠️ Please run Canny Edge Detection first.")
            else:
                with st.spinner("Calculating white pixel density..."):
                    img = st.session_state['canny_output']
                    white_pixels = count_white_pixels(img)
                    img_area = img.shape[0] * img.shape[1]
                    density = white_pixels / img_area
                    st.session_state['traffic_density'] = density

                    # Visual feedback
                    if density >= 0.18:
                        level, time_sec, color = "🚗 Very High", 60, "#ffebee"
                    elif density >= 0.14:
                        level, time_sec, color = "🚙 High", 50, "#fff3e0"
                    elif density >= 0.10:
                        level, time_sec, color = "🚕 Moderate", 40, "#f0f4c3"
                    elif density >= 0.06:
                        level, time_sec, color = "🚘 Low", 30, "#e0f7fa"
                    else:
                        level, time_sec, color = "🛵 Very Low", 20, "#f3e5f5"

                    st.metric(label="📈 Traffic Density", value=f"{density:.4f}", delta=level)
                    st.metric(label="⏱️ Green Light Time", value=f"{time_sec} seconds")

                    st.markdown(f"""
                        <div style='padding:12px; background-color:{color}; border-left:6px solid #1976d2;'>
                            <b>📌 White Pixels:</b> {white_pixels}<br>
                            <b>📐 Image Area:</b> {img_area}<br>
                            <b>🚦 Density:</b> {density:.4f} ({level})<br>
                            <b>🟢 Green Time:</b> {time_sec} seconds
                        </div>
                    """, unsafe_allow_html=True)

# Sidebar # Sidebar Status Summary
st.sidebar.title("Process Status")

if st.session_state['uploaded_image'] is None:
    st.sidebar.write("❌ No image uploaded.")
else:
    st.sidebar.write("✅ Image uploaded.")

if st.session_state['ambulance_detected'] is None:
    st.sidebar.write("🚑 Ambulance Detection: Not done yet.")
elif st.session_state['ambulance_detected'] is True:
    st.sidebar.success("🚑 Ambulance detected!")
else:
    st.sidebar.info("🚑 No ambulance detected.")

if st.session_state['canny_output'] is None:
    st.sidebar.write("⚙️ Canny Edge Detection: Not done yet.")
else:
    st.sidebar.success("⚙️ Canny Edge Detection completed.")

if st.session_state['traffic_density'] is None:
    st.sidebar.write("📊 Traffic Analysis: Not done yet.")
else:
    dens = st.session_state['traffic_density']
    st.sidebar.success(f"📊 Traffic Density: {dens:.4f}")

# --- Add How To Use Info Panel ---
with st.sidebar.expander("ℹ️ How to Use This App"):
    st.markdown("""
    - **Upload Image:** Upload a traffic image to analyze.
    - **Detect Ambulance:** Detect if any ambulance is present in the image using OCR and object detection.
    - **Canny Edge Detection:** Highlights edges to estimate traffic density.
    - **Traffic Density:** Based on white pixel density, shows traffic load and green light timing.

    **Traffic Density Meaning:**
    - Very High (≥ 0.18): Heavy traffic — longer green light.
    - High (≥ 0.14): High traffic — moderate green light.
    - Moderate (≥ 0.10): Normal traffic.
    - Low (≥ 0.06): Light traffic.
    - Very Low (< 0.06): Very light traffic — shortest green light.

    **Ambulance Detected:** Overrides traffic density logic to allow faster passage.
    """)

# Reset
if st.button("🔄 Reset All"):
    st.session_state.clear()
    st.rerun()














































giving ALL TEXT GREEN BOUNDING IG 


import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
import time
from torchvision import transforms
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from CannyEdgeDetector import *
import pytesseract
import easyocr
import difflib

# Set path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize EasyOCR
easyocr_reader = easyocr.Reader(['en'], gpu=False)

# Load model
COCO_INSTANCE_CATEGORY_NAMES = [...]  # Same as before
weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
model = ssdlite320_mobilenet_v3_large(weights=weights)
model.eval()
# Load the EAST text detector
east_net = cv2.dnn.readNet(r"C:\Users\Slnrockstones\Desktop\newmodel\frozen_east_text_detection.pb")


# Utility Functions
def is_ambulance_text(text):
    text = text.upper()
    words = text.split()
    for word in words:
        word_clean = ''.join(filter(str.isalpha, word))
        if 5 <= len(word_clean) <= 10:
            # Check normal and reversed
            if difflib.get_close_matches(word_clean, ["AMBULANCE"], cutoff=0.7):
                return True
            if difflib.get_close_matches(word_clean[::-1], ["AMBULANCE"], cutoff=0.7):
                return True
    return False

def preprocess_for_ocr(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    thresh = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    return thresh

def ocr_pytesseract(image):
    return pytesseract.image_to_string(image, config=r'--oem 3 --psm 6')

def ocr_easyocr(image, conf_threshold=0.3):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = easyocr_reader.readtext(rgb)
    filtered_texts = [res[1] for res in results if res[2] > conf_threshold]
    return " ".join(filtered_texts)

def detect_text_with_east_and_ocr(crop):
    detected_texts = []
    ambulance_found = False
    text_boxes = detect_text_regions_east(crop)
    print(f"[DEBUG] EAST detected {len(text_boxes)} text boxes.")

    for (startX, startY, endX, endY) in text_boxes:
        # Clip coordinates
        startX, startY = max(0, startX), max(0, startY)
        endX, endY = min(crop.shape[1], endX), min(crop.shape[0], endY)
        roi = crop[startY:endY, startX:endX]

        processed = preprocess_for_ocr(roi)
        text_pt = ocr_pytesseract(processed).strip()
        text_eo = ocr_easyocr(roi).strip()

        combined = (text_pt + " " + text_eo).strip()
        print(f"[OCR DEBUG] pytesseract: '{text_pt}', easyocr: '{text_eo}', combined: '{combined}'")

        if combined:
            detected_texts.append(combined)
            if is_ambulance_text(combined):
                ambulance_found = True

    return " | ".join(detected_texts), ambulance_found



# Streamlit setup
st.set_page_config(page_title="Smart Traffic Control", layout="centered")
st.title("🚦 Density-Based Smart Traffic Control System")
st.sidebar.title("📋 Console Logs")

# Session state
for key in ['ambulance_detected', 'canny_output', 'traffic_density', 'uploaded_image', 'uploaded_file_raw']:
    if key not in st.session_state:
        st.session_state[key] = None

# File uploader
uploaded_file = st.file_uploader("📤 Upload a Traffic Image", type=["jpg", "jpeg", "png"])
if uploaded_file != st.session_state['uploaded_file_raw']:
    st.session_state.update({
        'ambulance_detected': None,
        'canny_output': None,
        'traffic_density': None,
        'uploaded_image': Image.open(uploaded_file) if uploaded_file else None,
        'uploaded_file_raw': uploaded_file
    })

# Core functions
def apply_canny(image):
    gray_img = np.dot(np.array(image)[..., :3], [0.2989, 0.5870, 0.1140])
    edge = CannyEdgeDetector([gray_img], sigma=1.4, kernel_size=5, lowthreshold=0.09, highthreshold=0.20, weak_pixel=100)
    processed = edge.detect()[0]
    return (processed / processed.max() * 255).astype(np.uint8)

def count_white_pixels(img):
    return np.sum(img == 255)
def decode_predictions(scores, geometry, min_confidence):
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(numCols):
            if scoresData[x] < min_confidence:
                continue

            offsetX = x * 4.0
            offsetY = y * 4.0

            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + cos * xData1[x] + sin * xData2[x])
            endY = int(offsetY - sin * xData1[x] + cos * xData2[x])
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, int(w), int(h)))
            confidences.append(float(scoresData[x]))

    return (rects, confidences)


def detect_text_regions_east(image, width=320, height=320, min_confidence=0.5):
    orig = image.copy()
    (H, W) = image.shape[:2]
    rW = W / float(width)
    rH = H / float(height)

    # Resize and prepare blob
    resized = cv2.resize(image, (width, height))
    blob = cv2.dnn.blobFromImage(resized, 1.0, (width, height),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    east_net.setInput(blob)
    (scores, geometry) = east_net.forward([
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"
    ])

    (boxes, confidences) = decode_predictions(scores, geometry, min_confidence)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)

    final_boxes = []
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y, w, h) = boxes[i]
            startX = int(x * rW)
            startY = int(y * rH)
            endX = int((x + w) * rW)
            endY = int((y + h) * rH)
            final_boxes.append((startX, startY, endX, endY))

    return final_boxes



def detect_vehicles_with_ocr(image):
    img_rgb = np.array(image.convert("RGB"))
    img_tensor = transforms.ToTensor()(img_rgb).unsqueeze(0)
    with torch.no_grad():
        prediction = model(img_tensor)

    found = False
    ambulance_text_detected = ""
    for i in range(len(prediction[0]['labels'])):
        label = prediction[0]['labels'][i].item()
        score = prediction[0]['scores'][i].item()
        if label in [3, 6, 8] and score > 0.3:
            box = prediction[0]['boxes'][i].cpu().numpy().astype(int)
            crop = img_rgb[box[1]:box[3], box[0]:box[2]]
            text, is_amb = detect_text_with_east_and_ocr(crop)

            if text.strip() != "":
                ambulance_text_detected += text + " | "

            if is_amb:
                found = True
                cv2.rectangle(img_rgb, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(img_rgb, "AMBULANCE", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)


            if "AMBULANCE DETECTED" in text:
                found = True
                ambulance_text_detected = text

# Optional: Even if not ambulance, keep all text for display
            elif text.strip() != "":
                ambulance_text_detected = text

                cv2.rectangle(img_rgb, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(img_rgb, "AMBULANCE", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    return img_rgb, found, ambulance_text_detected

# Main interaction
if st.session_state['uploaded_image']:
    image = st.session_state['uploaded_image']
    st.image(image, caption="📸 Uploaded Image", use_column_width=True)

    if st.button("🚑 Detect Ambulance"):
        with st.spinner("Analyzing vehicles and text..."):
            result_img, found, ambulance_text = detect_vehicles_with_ocr(image)
            st.image(result_img, caption="🚓 Detection Output", use_column_width=True)
            st.sidebar.write("✅ Detection complete.")

        if found:
            st.success("🚨 **Ambulance Detected!** Green signal time: 60 seconds.")
            st.markdown(f"""<div style='padding:10px; background-color:#d1f0d1; border-left:5px solid green'>
                <b>🔍 OCR Detected Text:</b><br><span style='font-size:18px'>{ambulance_text}</span></div>
                """, unsafe_allow_html=True)
            st.session_state['ambulance_detected'] = True
        else:
            st.warning("❌ No ambulance found.")
            st.session_state['ambulance_detected'] = False

    if st.session_state['ambulance_detected'] is False:
        if st.button("⚙️ Run Canny Edge Detection"):
            progress_bar = st.progress(0, text="Canny Edge Detection in progress...")  # Initialize bar

            with st.spinner("Detecting edges..."):
            # Simulate progress
                for percent_complete in range(100):
                    time.sleep(0.01)  # Simulate work
                    progress_bar.progress(percent_complete + 1, text="Canny Edge Detection in progress...")

                canny_img = apply_canny(image)
                progress_bar.empty()  # Hide progress bar

                st.image(canny_img, caption="🖼️ Canny Edge Output", use_column_width=True)
                st.session_state['canny_output'] = canny_img




        if st.button("📊 Analyze Traffic & Allocate Green Time"):
            if st.session_state['canny_output'] is None:
                st.warning("⚠️ Please run Canny Edge Detection first.")
            else:
                with st.spinner("Calculating white pixel density..."):
                    img = st.session_state['canny_output']
                    white_pixels = count_white_pixels(img)
                    img_area = img.shape[0] * img.shape[1]
                    density = white_pixels / img_area
                    st.session_state['traffic_density'] = density

                    # Visual feedback
                    if density >= 0.18:
                        level, time_sec, color = "🚗 Very High", 60, "#ffebee"
                    elif density >= 0.14:
                        level, time_sec, color = "🚙 High", 50, "#fff3e0"
                    elif density >= 0.10:
                        level, time_sec, color = "🚕 Moderate", 40, "#f0f4c3"
                    elif density >= 0.06:
                        level, time_sec, color = "🚘 Low", 30, "#e0f7fa"
                    else:
                        level, time_sec, color = "🛵 Very Low", 20, "#f3e5f5"

                    st.metric(label="📈 Traffic Density", value=f"{density:.4f}", delta=level)
                    st.metric(label="⏱️ Green Light Time", value=f"{time_sec} seconds")

                    st.markdown(f"""
                        <div style='padding:12px; background-color:{color}; border-left:6px solid #1976d2;'>
                            <b>📌 White Pixels:</b> {white_pixels}<br>
                            <b>📐 Image Area:</b> {img_area}<br>
                            <b>🚦 Density:</b> {density:.4f} ({level})<br>
                            <b>🟢 Green Time:</b> {time_sec} seconds
                        </div>
                    """, unsafe_allow_html=True)

# Sidebar # Sidebar Status Summary
st.sidebar.title("Process Status")

if st.session_state['uploaded_image'] is None:
    st.sidebar.write("❌ No image uploaded.")
else:
    st.sidebar.write("✅ Image uploaded.")

if st.session_state['ambulance_detected'] is None:
    st.sidebar.write("🚑 Ambulance Detection: Not done yet.")
elif st.session_state['ambulance_detected'] is True:
    st.sidebar.success("🚑 Ambulance detected!")
else:
    st.sidebar.info("🚑 No ambulance detected.")

if st.session_state['canny_output'] is None:
    st.sidebar.write("⚙️ Canny Edge Detection: Not done yet.")
else:
    st.sidebar.success("⚙️ Canny Edge Detection completed.")

if st.session_state['traffic_density'] is None:
    st.sidebar.write("📊 Traffic Analysis: Not done yet.")
else:
    dens = st.session_state['traffic_density']
    st.sidebar.success(f"📊 Traffic Density: {dens:.4f}")

# --- Add How To Use Info Panel ---
with st.sidebar.expander("ℹ️ How to Use This App"):
    st.markdown("""
    - **Upload Image:** Upload a traffic image to analyze.
    - **Detect Ambulance:** Detect if any ambulance is present in the image using OCR and object detection.
    - **Canny Edge Detection:** Highlights edges to estimate traffic density.
    - **Traffic Density:** Based on white pixel density, shows traffic load and green light timing.

    **Traffic Density Meaning:**
    - Very High (≥ 0.18): Heavy traffic — longer green light.
    - High (≥ 0.14): High traffic — moderate green light.
    - Moderate (≥ 0.10): Normal traffic.
    - Low (≥ 0.06): Light traffic.
    - Very Low (< 0.06): Very light traffic — shortest green light.

    **Ambulance Detected:** Overrides traffic density logic to allow faster passage.
    """)

# Reset
if st.button("🔄 Reset All"):
    st.session_state.clear()
    st.rerun()
