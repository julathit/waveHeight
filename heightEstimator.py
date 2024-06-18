import torch
import cv2
import numpy as np
from fastsam import FastSAM, FastSAMPrompt
import os
import csv
import shutil

maxScale = 94
ruleW = 16.64

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
    print("GPU Device Name:", torch.cuda.get_device_name(0))  # Print the GPU device name
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.") 

# Global variable to store the coordinates
clicked_point = None

# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        cv2.destroyAllWindows()

def mouseClicker(set_mouse_call, name, mouse_call, image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Display the image
    cv2.imshow("Select Image", image)
    set_mouse_call.setMouseCallback(name, mouse_call)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def calculate_height(IMAGE_PATH):

    model = FastSAM('FastSAM-s.pt')
    DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    everything_results = model(
        IMAGE_PATH,
        device=DEVICE,
        retina_masks=True,
        imgsz=1024,
        conf=0.4,
        iou=0.9,
    )
    prompt_process = FastSAMPrompt(IMAGE_PATH, everything_results, device=DEVICE)

    # Point prompt
    ann = prompt_process.point_prompt(points=[clicked_point], pointlabel=[1])

    # Find the bounding box of the non-transparent regions
    x, y, w, h = cv2.boundingRect(ann[0].astype(np.uint8))

    height_cm = maxScale - h * ruleW / w
    print(height_cm,w,h)
    return height_cm


def extract_frames(video_path, output_folder, timestamps):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Couldn't open the video file.")
        return

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Extract frames at specified timestamps
    for timestamp in timestamps:
        # Set the frame position to the timestamp (in milliseconds)
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        
        # Read the frame
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Couldn't read frame at timestamp {timestamp} seconds.")
            continue
        
        # Save the frame as an image with timestamp in the filename
        output_filename = os.path.join(output_folder, f"{timestamp}.jpg")
        cv2.imwrite(output_filename, frame)
        print(f"Frame extracted at timestamp {timestamp} seconds and saved as {output_filename}")

    # Release the video capture object
    cap.release()

# Path to the video file
video_path = "data1.mp4"

# Output folder to save the extracted frames
output_folder = "extracted_frames"

# amout of frame to have
timestamps = np.arange(1, 60, 1)

# Call the function to extract frames
extract_frames(video_path, output_folder, timestamps)

surfaceHeight = []

mouseClicker(cv2, "Select Image", mouse_callback, "./extracted_frames/2.jpg")

for _ in timestamps:
    height = calculate_height(f"./extracted_frames/{_}.jpg")
    surfaceHeight.append([_,height])
    
print(surfaceHeight)

# Save the surfaceHeight array to a CSV file
output_csv_file = "surface_height.csv"
with open(output_csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Timestamp', 'Height'])  # Write the header
    for item in surfaceHeight:
        writer.writerow(item)

# shutil.rmtree(output_folder)

print("Surface height data saved to:", output_csv_file)
