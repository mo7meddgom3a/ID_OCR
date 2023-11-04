
from ultralytics import YOLO
import cv2
import easyocr
from crop_card import crop_card_front, crop_card_back


reader = easyocr.Reader(['ar'], verbose=True)
# reader = easyocr.Reader(['fa', 'ar'], verbose=True)

# Load a pretrained YOLOv8l model
ocr_front_model = YOLO('./models/ocr_front_g2.pt')
# Load a pretrained YOLOv8l model
ocr_back_model = YOLO("./models/ocr_back_g2.pt")



def ocr_reader(img):
    """
    This function used to detect arabic letters on images.

    Parameters:
        src (MatLike)

    Returns:
        data (str)
    """

    results = reader.readtext(img,text_threshold=0.7, low_text=0.3,paragraph = True)  # Set the allowlist to Arabic script
    detected_texts = [item[1] for item in results]
    data = ' '.join(detected_texts)

    return data


def read_text_front(img):
    """
    This function take front card image and get the name and address on the card.

    Parameters:
        src (MatLike)
    
    Returns:
        data (dict): {"name": , "address":}
    """
    
    masks = {
        "Name": img[110:250, 260:],
        "Address": img[250:380, 260:]
        # ,"ID": img[380:500, 335:]
    }

    Data ={}

    for key, mask in (list(masks.items())):
        data = ocr_reader(mask)
        Data[f"{key}"] = data

    return Data


def read_text_back(img):
    """
    This function take front card image and get data on the card.
    
    Parameters:
        src (MatLike)
    
    Returns:
        data (dict): {"Job": , "Gender": ,"Religion": ,"Status": }
    """

    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (1080, 640))

    masks = {
        "Job": img[75:190, 400:],
        "Gender": img[175:260, 790:900],
        "Religion": img[175:260, 550:790],
        "Status":img[175:260, 350:550]
    }

    Data ={}
    
    for key, mask in (list(masks.items())):

        data = ocr_reader(mask)
        Data[f"{key}"] = data

    return Data           # Job,Gender,Religion,status


def read_id_front(img):
    """
    This function take front card image and get national number on the card.
    
    Parameters:
        src (MatLike)
    
    Returns:
        data (str)
    """

    # Run inference on 'image33.jpg'
    results = ocr_front_model(img)  # results list

    # Extract the bounding boxes and their corresponding classes
    boxes = results[0].boxes.xywh  # Assuming the boxes are in xywh format
    classes = results[0].boxes.cls
    
    # Create a list of tuples, where each tuple contains (x-coordinate, class_name)
    x_coords_with_classes = [(box[0], ocr_front_model.names[int(c)]) for box, c in zip(boxes, classes)]

    # Sort this list based on the x-coordinate
    sorted_classes = [cls for _, cls in sorted(x_coords_with_classes, key=lambda x: x[0])]
    result_string = ''.join(sorted_classes)
    return result_string

def read_id_back(img):
    """
    This function take back card image and get national number on the card.
    
    Parameters:
        src (MatLike)
    
    Returns:
        data (str)
    """
    # get gray image
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Crop the image
    cropped = gray[0:65, 375:690]

    # Set the confidence threshold for the model to 0.25
    ocr_back_model.conf = 0.25  # Adjust this line according to the actual API

    # Convert cropped image to color (YOLO typically requires three channels)
    cropped_color = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)

    # Run inference on the cropped image
    results = ocr_back_model(cropped_color)

    # Extract the bounding boxes and their corresponding classes
    boxes = results[0].boxes.xywh  # Assuming the boxes are in xywh format
    classes = results[0].boxes.cls

    # Create a list of tuples, where each tuple contains (x-coordinate, class_name)
    x_coords_with_classes = [(box[0], ocr_back_model.names[int(c)]) for box, c in zip(boxes, classes)]

    # Sort this list based on the x-coordinate
    sorted_classes = [cls for _, cls in sorted(x_coords_with_classes, key=lambda x: x[0])]

    # Concatenate the sorted class names into a single string
    result_string = ''.join(sorted_classes)

    return result_string


def extract_id_and_average_confidence(img):
    """
    This function take front card image and get national number
    and average precentages of number recognition accurac on the card.
    
    Parameters:
        src (MatLike)
    
    Returns:
        id (str)
        confidence (float)
    """   
    # Run inference on the image
    results = ocr_front_model(img)

    # Extract the bounding boxes, classes, and confidence scores
    boxes = results[0].boxes.xywh  # Assuming the boxes are in xywh format
    classes = results[0].boxes.cls
    confidences = results[0].boxes.conf  # Assuming confidence scores are stored in results[0].boxes.conf

    # Create a list of tuples, where each tuple contains (x-coordinate, class_name, confidence_score)
    x_coords_classes_confidences = [
        (box[0], ocr_front_model.names[int(cls)], conf) for box, cls, conf in zip(boxes, classes, confidences)
    ]

    # Sort this list based on the x-coordinate
    sorted_info = sorted(x_coords_classes_confidences, key=lambda x: x[0])
    
    # Concatenate class names to form the result string
    result_string = ''.join([info[1] for info in sorted_info])
    
    # Calculate the average confidence score
    average_confidence = sum([info[2] for info in sorted_info]) / len(sorted_info) if sorted_info else 0
    avg = float(average_confidence)
    return result_string, avg

def extract_id_back_and_average_confidence(img):
    """
    This function take back card image and get national number
    and average percentages of number recognition accuracy on the card.
    
    Parameters:
        src (MatLike)
    
    Returns:
        id (str)
        confidence (float)
    """   
    
    # Run infer
    a, b,_= img.shape
    cropped = img[0:65, 375:690]
    
    # Run inference on the cropped image
    results = ocr_back_model(cropped)  # Ensure the model accepts numpy array inputs directly

    # Extract the bounding boxes and their corresponding classes
    # The following lines assume results is a list-like object with attributes, adjust according to actual returned object
    boxes = results[0].boxes.xywh  # Assuming the boxes are in xywh format
    classes = results[0].boxes.cls
    confidences = results[0].boxes.conf  # Assuming confidence scores are stored in results[0].boxes.conf

    # Create a list of tuples, where each tuple contains (x-coordinate, class_name, confidence_score)
    x_coords_classes_confidences = [
        (box[0], ocr_back_model.names[int(cls)], conf) for box, cls, conf in zip(boxes, classes, confidences)
    ]

# Sort this list based on the x-coordinate
    sorted_info = sorted(x_coords_classes_confidences, key=lambda x: x[0])
    
    # Concatenate class names to form the result string
    result_string = ''.join([info[1] for info in sorted_info])
    
    # Calculate the average confidence score
    average_confidence = sum([info[2] for info in sorted_info]) / len(sorted_info) if sorted_info else 0
    avg = float(average_confidence)
    return result_string , avg



######## user functions #######

def process_ocr_front(front_image):
    """
    This function take front card image and get the name and address and Id on the card.

    Parameters:
        src (MatLike)
    
    Returns:
        data (dict): {"name": , "address": , "Id"}
    """

    card_img = crop_card_front (front_image)
    Data = read_text_front(card_img)
    Data["Id"] = read_id_front(card_img)
    
    return Data


def process_ocr_back(back_img):
    """
    This function take front card image and get data on the card.
    
    Parameters:
        src (MatLike)
    
    Returns:
        data (dict): {"Job": , "Gender": ,"Religion": ,"Status": , "national_id":}
    """

    card_img = crop_card_back(back_img)
    Data = read_text_back(card_img)
    Data["Id"] = read_id_back(card_img)
    
    return Data


def process_ocr_two_side(front_img, back_img):

    # front image
    card_front = crop_card_front (front_img)
    data_front = read_text_front(card_front)
    id_front, avg_confidence_front = extract_id_and_average_confidence(card_front)

    # back image
    card_back = crop_card_back(back_img)
    data_back = read_text_back(card_back)
    id_back, avg_confidence_back = extract_id_back_and_average_confidence(card_back)

    # detect the best id
    id = id_front
    if id_front == id_back and len(id_front) == 14:
        id = id_front

    elif len(id_front) == 14 and len(id_back) != 14:
        id = id_front
    
    elif len(id_front) != 14 and len(id_back) == 14:
        id = id_back
    
    elif len(id_front) == len(id_back) and len(id_front) == 14:
        if avg_confidence_front > avg_confidence_back:
            id = id_front

        else:
            id = id_back

    else:
        id = None

    # collect all data
    Data = {}
    Data.update(data_front)
    Data.update(data_back)
    Data["Id"] = id

    return Data
