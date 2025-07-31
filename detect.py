import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_for_ocr(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Adaptive thresholding instead of global OTSU
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 8
    )
    
    # Morphological opening to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Invert back so text is black on white
    processed = cv2.bitwise_not(opening)
    
    # Save processed image for debugging
    cv2.imwrite('processed_for_ocr.png', processed)
    return processed

def test_ocr_on_crop(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load image at {img_path}")
        return
    
    processed = preprocess_for_ocr(img)
    
    # Use config to limit OCR for uppercase letters only (common for AMBULANCE)
    custom_config = r'--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    text = pytesseract.image_to_string(processed, config=custom_config).strip()
    print("OCR EasyOCR Text:\n")
    print(text)

if __name__ == "__main__":
    test_ocr_on_crop('C:/Users/Slnrockstones/Desktop/test/ambulance_crop.jpeg')
