import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from CannyEdgeDetector import *
import io

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

from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights

weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
model = ssdlite320_mobilenet_v3_large(weights=weights)
model.eval()


st.title("🚦 Density-Based Smart Traffic Control System")

uploaded_file = st.file_uploader("Upload a Traffic Image", type=["jpg", "jpeg", "png"])

sample_pixels = None
reference_pixels = None

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

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Traffic Image", use_column_width=True)
    image = Image.open(uploaded_file)

    if st.button("🧪 Preprocess Image (Canny Edge Detection)"):
        result_img = apply_canny(image)
        st.image(result_img, caption="Canny Edge Output", use_column_width=True, clamp=True)
        st.session_state['sample'] = result_img

   # Load reference image from file once (you must have this file in your repo)
reference_path = "gray/refrence.png"
if not os.path.exists(reference_path):
    st.error("Reference image not found. Please ensure gray/refrence.png exists.")
else:
    reference = Image.open(reference_path).convert("L")
    ref_array = np.array(reference)
    st.session_state['reference'] = ref_array


    if st.button("📊 Count White Pixels & Allocate Time"):
        if 'sample' in st.session_state and 'reference' in st.session_state:
            sample_pixels = count_white_pixels(st.session_state['sample'])
            img_height, img_width = st.session_state['sample'].shape
            image_area = img_height * img_width
            white_density = sample_pixels / image_area  # value between 0 and 1

            st.success(f"Sample White Pixels: {sample_pixels}\nImage Area: {image_area}\nWhite Pixel Density: {white_density:.4f}")

            # Allocate green time based on white pixel density thresholds
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


        else:
            st.warning("Upload and preprocess both images first.")


    if st.button("🚑 Detect Ambulance"):
        result, found = detect_vehicles(image)
        st.image(result, caption="Ambulance Detection Result", use_column_width=True)
        if found:
            st.success("🚑 Ambulance-like vehicle detected!")
        else:
            st.warning("❌ No ambulance-like vehicle found.")





















???????///////general he done in meet?????????????????/
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askopenfilename
import numpy as np 
from CannyEdgeDetector import *
import skimage
import matplotlib.image as mpimg
import os
import scipy.misc as sm
import cv2
import matplotlib.pyplot as plt 
import torch
import torchvision
from torchvision import transforms
from PIL import Image



main = tkinter.Tk()

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

weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
model.eval()

main.title("Density Based Smart Traffic Control System")
main.geometry("1300x1200")

global filename
global refrence_pixels
global sample_pixels

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def uploadTrafficImage():
    global filename
    filename = filedialog.askopenfilename(initialdir="images")
    pathlabel.config(text=filename)

def visualize(imgs, format=None, gray=False):
    j = 0
    plt.figure(figsize=(20, 40))
    for i, img in enumerate(imgs):
        if img.shape[0] == 3:
            img = img.transpose(1,2,0)
        plt_idx = i+1
        plt.subplot(2, 2, plt_idx)
        if j == 0:
            plt.title('Sample Image')
            plt.imshow(img, format)
            j = j + 1
        elif j > 0:
            plt.title('Reference Image')
            plt.imshow(img, format)
            
    plt.show()
    
def applyCanny():
    imgs = []

    # Read and convert image to grayscale
    img = mpimg.imread(filename)
    img = rgb2gray(img)

    imgs.append(img)

    # Apply Canny Edge Detection
    edge = CannyEdgeDetector(imgs, sigma=1.4, kernel_size=5, lowthreshold=0.09, highthreshold=0.20, weak_pixel=100)
    imgs = edge.detect()

    # Ensure gray directory exists
    os.makedirs("gray", exist_ok=True)

    # Take the first image (processed)
    processed_img = imgs[0]

    # Normalize to 0-255 and convert to uint8 before saving with OpenCV
    processed_img_uint8 = (processed_img / processed_img.max() * 255).astype(np.uint8)

    # Save the processed image
    cv2.imwrite("gray/test.png", processed_img_uint8)

    # Confirm the file is saved correctly
    if not os.path.exists("gray/test.png"):
        messagebox.showerror("Error", "Failed to save gray/test.png.")
        return

    # Load images for visualization
    if not os.path.exists("gray/refrence.png"):
        messagebox.showerror("Missing Reference", "gray/refrence.png not found. Add the reference image.")
        return

    temp = []
    img1 = mpimg.imread('gray/test.png')
    img2 = mpimg.imread('gray/refrence.png')

    temp.append(img1)
    temp.append(img2)

    visualize(temp)


def pixelcount():
    global refrence_pixels
    global sample_pixels
    img = cv2.imread('gray/test.png', cv2.IMREAD_GRAYSCALE)
    sample_pixels = np.sum(img == 255)
    
    img = cv2.imread('gray/refrence.png', cv2.IMREAD_GRAYSCALE)
    refrence_pixels = np.sum(img == 255)
    messagebox.showinfo("Pixel Counts", "Total Sample White Pixels Count : "+str(sample_pixels)+"\nTotal Refrence White Pixels Count : "+str(refrence_pixels))


def timeAllocation():
    avg = (sample_pixels/refrence_pixels) *100
    if avg >= 90:
        messagebox.showinfo("Green Signal Allocation Time","Traffic is very high allocation green signal time : 60 secs")
    if avg > 85 and avg < 90:
        messagebox.showinfo("Green Signal Allocation Time","Traffic is high allocation green signal time : 50 secs")
    if avg > 75 and avg <= 85:
        messagebox.showinfo("Green Signal Allocation Time","Traffic is moderate green signal time : 40 secs")
    if avg > 50 and avg <= 75:
        messagebox.showinfo("Green Signal Allocation Time","Traffic is low allocation green signal time : 30 secs")
    if avg <= 50:
        messagebox.showinfo("Green Signal Allocation Time","Traffic is very low allocation green signal time : 20 secs")        
        
def detectAmbulance():
    global filename
    if not filename:
        messagebox.showerror("Error", "Please upload an image first.")
        return

    img = cv2.imread(filename)
    if img is None:
        messagebox.showerror("Error", "Could not read uploaded image.")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(pil_img).unsqueeze(0)

    with torch.no_grad():
        prediction = model(img_tensor)

    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']

    ambulance_found = False
    for i in range(len(labels)):
        class_id = labels[i].item()
        score = scores[i].item()

        if class_id < len(COCO_INSTANCE_CATEGORY_NAMES):
            name = COCO_INSTANCE_CATEGORY_NAMES[class_id]
        else:
            continue

        if class_id in [3, 4, 6, 8] and score > 0.85:  # car, motorcycle, bus, truck
            x1, y1, x2, y2 = boxes[i].cpu().numpy().astype(int)
            box_width = x2 - x1
            box_height = y2 - y1

            if box_width > 100 and box_height > 100:
                ambulance_found = True
                label_name = "Ambulance" if name in ["bus", "truck"] else name
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f'{label_name} ({score:.2f})', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if ambulance_found:
        messagebox.showinfo("Ambulance Detection", "🚑 Ambulance-like vehicle detected!")
    else:
        messagebox.showinfo("Ambulance Detection", "❌ No ambulance-like vehicle found.")

    cv2.imshow("Ambulance Detection Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def exit():
    main.destroy()
    

    
font = ('times', 16, 'bold')
title = Label(main, text='                           Density Based Smart Traffic Control System Using Canny Edge Detection Algorithm for Congregating Traffic Information',anchor=W, justify=CENTER)
title.config(bg='yellow4', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)


font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Traffic Image", command=uploadTrafficImage)
upload.place(x=50,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='yellow4', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=50,y=150)

process = Button(main, text="Image Preprocessing Using Canny Edge Detection", command=applyCanny)
process.place(x=50,y=200)
process.config(font=font1)

count = Button(main, text="White Pixel Count", command=pixelcount)
count.place(x=50,y=250)
count.config(font=font1)

count = Button(main, text="Calculate Green Signal Time Allocation", command=timeAllocation)
count.place(x=50,y=300)
count.config(font=font1)
ambulanceBtn = Button(main, text="Detect Ambulance in Image", command=detectAmbulance)
ambulanceBtn.place(x=50, y=400)
ambulanceBtn.config(font=font1)


exitButton = Button(main, text="Exit", command=exit)
exitButton.place(x=50,y=350)
exitButton.config(font=font1)


main.config(bg='magenta3')
main.mainloop()






???????????????????streamlit     using streamlit  ????// we should upload reference img also???????????????
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from CannyEdgeDetector import *
import io

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

weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
model.eval()

st.title("🚦 Density-Based Smart Traffic Control System")

uploaded_file = st.file_uploader("Upload a Traffic Image", type=["jpg", "jpeg", "png"])

sample_pixels = None
reference_pixels = None

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

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Traffic Image", use_column_width=True)
    image = Image.open(uploaded_file)

    if st.button("🧪 Preprocess Image (Canny Edge Detection)"):
        result_img = apply_canny(image)
        st.image(result_img, caption="Canny Edge Output", use_column_width=True, clamp=True)
        st.session_state['sample'] = result_img

    ref_img = st.file_uploader("Upload Reference Image (Preprocessed)", type=["png", "jpg", "jpeg"], key="ref")
    if ref_img is not None:
        reference = Image.open(ref_img).convert("L")
        ref_array = np.array(reference)
        st.image(reference, caption="Reference Image", use_column_width=True)
        st.session_state['reference'] = ref_array

    if st.button("📊 Count White Pixels & Allocate Time"):
        if 'sample' in st.session_state and 'reference' in st.session_state:
            sample_pixels = count_white_pixels(st.session_state['sample'])
            reference_pixels = count_white_pixels(st.session_state['reference'])
            st.success(f"Sample White Pixels: {sample_pixels}\nReference White Pixels: {reference_pixels}")

            ratio = (sample_pixels / reference_pixels) * 100
            if ratio >= 90:
                st.info("Traffic is very high. Green signal time: 60 seconds")
            elif ratio > 85:
                st.info("Traffic is high. Green signal time: 50 seconds")
            elif ratio > 75:
                st.info("Traffic is moderate. Green signal time: 40 seconds")
            elif ratio > 50:
                st.info("Traffic is low. Green signal time: 30 seconds")
            else:
                st.info("Traffic is very low. Green signal time: 20 seconds")
        else:
            st.warning("Upload and preprocess both images first.")

    if st.button("🚑 Detect Ambulance"):
        result, found = detect_vehicles(image)
        st.image(result, caption="Ambulance Detection Result", use_column_width=True)
        if found:
            st.success("🚑 Ambulance-like vehicle detected!")
        else:
            st.warning("❌ No ambulance-like vehicle found.")
















??????????????????????????????????????????????????????????with out 2nd img upload????????????
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from CannyEdgeDetector import *
import io

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

weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
model.eval()

st.title("🚦 Density-Based Smart Traffic Control System")

uploaded_file = st.file_uploader("Upload a Traffic Image", type=["jpg", "jpeg", "png"])

sample_pixels = None
reference_pixels = None

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

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Traffic Image", use_column_width=True)
    image = Image.open(uploaded_file)

    if st.button("🧪 Preprocess Image (Canny Edge Detection)"):
        result_img = apply_canny(image)
        st.image(result_img, caption="Canny Edge Output", use_column_width=True, clamp=True)
        st.session_state['sample'] = result_img

   # Load reference image from file once (you must have this file in your repo)
reference_path = "gray/refrence.png"
if not os.path.exists(reference_path):
    st.error("Reference image not found. Please ensure gray/refrence.png exists.")
else:
    reference = Image.open(reference_path).convert("L")
    ref_array = np.array(reference)
    st.session_state['reference'] = ref_array


    if st.button("📊 Count White Pixels & Allocate Time"):
        if 'sample' in st.session_state and 'reference' in st.session_state:
            sample_pixels = count_white_pixels(st.session_state['sample'])
            reference_pixels = count_white_pixels(st.session_state['reference'])
            st.success(f"Sample White Pixels: {sample_pixels}\nReference White Pixels: {reference_pixels}")

            ratio = (sample_pixels / reference_pixels) * 100
            if ratio >= 90:
                st.info("Traffic is very high. Green signal time: 60 seconds")
            elif ratio > 85:
                st.info("Traffic is high. Green signal time: 50 seconds")
            elif ratio > 75:
                st.info("Traffic is moderate. Green signal time: 40 seconds")
            elif ratio > 50:
                st.info("Traffic is low. Green signal time: 30 seconds")
            else:
                st.info("Traffic is very low. Green signal time: 20 seconds")
        else:
            st.warning("Upload and preprocess both images first.")

    if st.button("🚑 Detect Ambulance"):
        result, found = detect_vehicles(image)
        st.image(result, caption="Ambulance Detection Result", use_column_width=True)
        if found:
            st.success("🚑 Ambulance-like vehicle detected!")
        else:
            st.warning("❌ No ambulance-like vehicle found.")











?????????????????/PyQt5 Version
import sys
import os
import cv2
import numpy as np
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication, QVBoxLayout, QLabel, QPushButton, QFileDialog,
    QWidget, QTextEdit, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage
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

weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
model.eval()

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def apply_canny(image_np):
    gray_img = rgb2gray(image_np)
    edge = CannyEdgeDetector([gray_img], sigma=1.4, kernel_size=5, lowthreshold=0.09, highthreshold=0.20, weak_pixel=100)
    processed = edge.detect()[0]
    return (processed / processed.max() * 255).astype(np.uint8)

def count_white_pixels(img):
    return np.sum(img == 255)

def detect_vehicles(image_pil):
    img_rgb = np.array(image_pil.convert("RGB"))
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

class TrafficApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🚦 Smart Traffic Control")
        self.resize(800, 600)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.label = QLabel("Upload a Traffic Image")
        self.layout.addWidget(self.label)

        self.image_label = QLabel()
        self.layout.addWidget(self.image_label)

        self.upload_btn = QPushButton("Upload Image")
        self.upload_btn.clicked.connect(self.upload_image)
        self.layout.addWidget(self.upload_btn)

        self.process_btn = QPushButton("Apply Canny Edge Detection")
        self.process_btn.clicked.connect(self.process_image)
        self.layout.addWidget(self.process_btn)

        self.count_btn = QPushButton("Count White Pixels & Allocate Time")
        self.count_btn.clicked.connect(self.count_pixels)
        self.layout.addWidget(self.count_btn)

        self.detect_btn = QPushButton("Detect Ambulance")
        self.detect_btn.clicked.connect(self.detect_ambulance)
        self.layout.addWidget(self.detect_btn)

        self.output = QTextEdit()
        self.layout.addWidget(self.output)

        self.sample_img = None
        self.reference_img = self.load_reference()

    def load_reference(self):
        ref_path = "gray/refrence.png"
        if not os.path.exists(ref_path):
            QMessageBox.critical(self, "Error", f"Reference image not found: {ref_path}")
            return None
        ref = Image.open(ref_path).convert("L")
        return np.array(ref)

    def upload_image(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
        if file:
            self.image = Image.open(file)
            qimg = QImage(file)
            self.image_label.setPixmap(QPixmap.fromImage(qimg).scaledToWidth(600))
            self.output.append("✅ Image uploaded.")

    def process_image(self):
        if not hasattr(self, 'image'):
            self.output.append("❌ Please upload an image first.")
            return
        np_image = np.array(self.image)
        self.sample_img = apply_canny(np_image)
        self.output.append("🧪 Canny edge detection applied.")

    def count_pixels(self):
        if self.sample_img is None or self.reference_img is None:
            self.output.append("❌ Missing image data.")
            return

        sample_count = count_white_pixels(self.sample_img)
        ref_count = count_white_pixels(self.reference_img)

        ratio = (sample_count / ref_count) * 100
        msg = f"📊 Sample Pixels: {sample_count}, Reference Pixels: {ref_count}\n"

        if ratio >= 90:
            msg += "🚦 Traffic is very high. Green signal time: 60 seconds"
        elif ratio > 85:
            msg += "🚦 Traffic is high. Green signal time: 50 seconds"
        elif ratio > 75:
            msg += "🚦 Traffic is moderate. Green signal time: 40 seconds"
        elif ratio > 50:
            msg += "🚦 Traffic is low. Green signal time: 30 seconds"
        else:
            msg += "🚦 Traffic is very low. Green signal time: 20 seconds"

        self.output.append(msg)

    def detect_ambulance(self):
        if not hasattr(self, 'image'):
            self.output.append("❌ Upload image first.")
            return
        result_img, found = detect_vehicles(self.image)
        self.output.append("🚑 Ambulance detection complete.")

        if found:
            self.output.append("✅ Ambulance-like vehicle detected.")
        else:
            self.output.append("❌ No ambulance-like vehicle found.")

        result_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        height, width, _ = result_bgr.shape
        qimg = QImage(result_bgr.data, width, height, 3 * width, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg).scaledToWidth(600))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TrafficApp()
    window.show()
    sys.exit(app.exec_())

























?????????????????????????????????????????????????final okok??????/
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from CannyEdgeDetector import *
import io

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

from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights

weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
model = ssdlite320_mobilenet_v3_large(weights=weights)
model.eval()


st.title("🚦 Density-Based Smart Traffic Control System")

uploaded_file = st.file_uploader("Upload a Traffic Image", type=["jpg", "jpeg", "png"])

sample_pixels = None
reference_pixels = None

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

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Traffic Image", use_column_width=True)
    image = Image.open(uploaded_file)

    if st.button("🧪 Preprocess Image (Canny Edge Detection)"):
        result_img = apply_canny(image)
        st.image(result_img, caption="Canny Edge Output", use_column_width=True, clamp=True)
        st.session_state['sample'] = result_img

   # Load reference image from file once (you must have this file in your repo)
reference_path = "gray/refrence.png"
if not os.path.exists(reference_path):
    st.error("Reference image not found. Please ensure gray/refrence.png exists.")
else:
    reference = Image.open(reference_path).convert("L")
    ref_array = np.array(reference)
    st.session_state['reference'] = ref_array


    if st.button("📊 Count White Pixels & Allocate Time"):
        if 'sample' in st.session_state and 'reference' in st.session_state:
            sample_pixels = count_white_pixels(st.session_state['sample'])
            img_height, img_width = st.session_state['sample'].shape
            image_area = img_height * img_width
            white_density = sample_pixels / image_area  # value between 0 and 1

            st.success(f"Sample White Pixels: {sample_pixels}\nImage Area: {image_area}\nWhite Pixel Density: {white_density:.4f}")

            # Allocate green time based on white pixel density thresholds
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


        else:
            st.warning("Upload and preprocess both images first.")


    if st.button("🚑 Detect Ambulance"):
        result, found = detect_vehicles(image)
        st.image(result, caption="Ambulance Detection Result", use_column_width=True)
        if found:
            st.success("🚑 Ambulance-like vehicle detected!")
        else:
            st.warning("❌ No ambulance-like vehicle found.")









