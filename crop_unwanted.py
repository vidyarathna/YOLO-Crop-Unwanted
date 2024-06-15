import numpy as np
import argparse
import time
import cv2
import os

def crop_unwanted_regions(image, objects_to_keep):
    cropped_images = []
    for obj in objects_to_keep:
        x, y, w, h = obj
        cropped_images.append(image[y:y+h, x:x+w])
    return cropped_images

def main():
    # Construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to input image")
    ap.add_argument("-y", "--yolo", required=True, help="base path to YOLO directory")
    ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
    ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applying non-maxima suppression")
    args = vars(ap.parse_args())

    # Load the COCO class labels our YOLO model was trained on
    labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    # Derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
    configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

    # Load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # Load our input image and grab its spatial dimensions
    image = cv2.imread(args["image"])
    (H, W) = image.shape[:2]

    # Determine only the output layer names that we need from YOLO
    ln = net.getLayerNames()
    unconnected_layers = net.getUnconnectedOutLayers()
    print("Unconnected layers:", unconnected_layers)
    layerNames = [ln[i - 1] for i in unconnected_layers]



    # Construct a blob from the input image and then perform a forward pass of the YOLO object detector
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(layerNames)
    end = time.time()

    # Show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # Initialize lists for detected bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    classIDs = []

    # Loop over each of the layer outputs
    for output in layerOutputs:
        for detection in output:
            # Extract the class ID and confidence (i.e., probability) of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # Filter out weak predictions by ensuring the detected probability is greater than the minimum probability
            if confidence > args["confidence"]:
                # Scale the bounding box coordinates back relative to the size of the image
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # Use the center (x, y)-coordinates to derive the top and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # Update the list of bounding box coordinates, confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

    # Ensure at least one detection exists
    if len(idxs) > 0:
        print("Indices of boxes to keep:", idxs)
        objects_to_keep = [boxes[i] for i in idxs.flatten()]


        cropped_images = crop_unwanted_regions(image, objects_to_keep)

        # Display the cropped images
        for i, cropped_img in enumerate(cropped_images):
            cv2.imshow(f"Cropped Image {i+1}", cropped_img)
        cv2.waitKey(0)

if __name__ == "__main__":
    main()

