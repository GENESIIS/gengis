import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import ImageTk
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2DTranspose
import cv2
import numpy as np
#from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from patchify import patchify, unpatchify
from PIL import Image
import segmentation_models as sm
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
import keras
import tensorflow as tf
from keras.models import load_model
from ultralytics import YOLO 
import sys 
import altair as alt 

tf.compat.v1.losses.sparse_softmax_cross_entropy
tf.estimator.SessionRunHook

# Defining the function to get the absolute paths of resources
def resource_path(relative_path):
    
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

def open_file():
    print("File -> Open")

def save_file():
    print("File -> Save")

def exit_app():
    root.quit()

def predict_teak():
    # Creating a new window for teak finder
    map_window = tk.Toplevel(root)
    map_window.title("Teak Area Finder")
    map_window.geometry("800x600")
    map_window.configure(bg="white")

    def browse_image2():
        filename2 = filedialog.askopenfilename(title="Select Image File", filetypes=(("Image files", "*.jpg;*.jpeg;*.png;*.tif"), ("All files", "*.*")))
        if filename2:
            img_path2.set(filename2)
            # displaying the browesed image
            img2 = cv2.imread(filename2)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            img2 = Image.fromarray(img2)
            img2.thumbnail((200, 200))  # Resize image for display
            img2 = ImageTk.PhotoImage(img2)
            image_label2.config(image=img2)
            image_label2.image = img2

    def predict_teak():
        img_path_str2 = img_path2.get()
        if img_path_str2:
            img = cv2.imread(img_path_str2)

            # Disabling predict_button and updating label text
            teak_seg_button.config(state=tk.DISABLED, text="Searching...", bg="yellow")
            map_window.update()

           
            # Loadding the model
            teak_model_path=resource_path("models/teak_last.pt")
            model = YOLO(teak_model_path)

            results = model.predict(img, conf=0.2)  # Passing the OpenCV image directly to the model

            for result in results:
          
                result.save(filename='result.jpg') 

                result_img = cv2.imread('result.jpg')

                plt.figure(figsize=(12, 6))
                plt.title('Predicted teak areas')
                plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.show()


            # Re-enabling predict_button and updating label text
            teak_seg_button.config(state=tk.NORMAL, text="Find teak trees",bg="green", fg="white")
    #tile lable for teak window
    label2 = tk.Label(map_window, text="Teak Area Finder", font=("Calibri", 20, "bold"))
    label2.configure(bg="white")
    label2.pack()

    # separator widget for teak window
    separator2 = ttk.Separator(map_window, orient="horizontal")
    separator2.pack(fill="x")

    label_text2 = """Teak area finder stands as an automated Python application designed to automatically identify and mask teak trees through the power of AI technology. Leveraging a trained YOLOV8 model, this tool automatically detect teak trees on a provided satellite image. To begin, simply select your image in jpg or tiff format by clicking 'Browse Image', then proceed by clicking 'Find teak trees'."""
    
    #description Lable for teak window
    Description_label2 = tk.Label(map_window, text=label_text2, wraplength=400)
    Description_label2.configure(bg="white")
    Description_label2.pack()
    
    # Browse button for teak window
    browse_button2 = tk.Button(map_window, text="Browse Image", command=browse_image2, font=("Arial", 12, "bold"), bg="#23395d", fg="white")
    browse_button2.pack(pady=15)

    # Label to display the selected image path in the teak window
    img_path2 = tk.StringVar()
    selected_image_label2 = tk.Label(map_window, textvariable=img_path2, bg="white")
    selected_image_label2.pack()

    # Label to display the selected image on the teak window
    image_label2 = tk.Label(map_window)
    image_label2.pack()

    # teak Predict button (yolov8 nodel) for teak window
    teak_seg_button = tk.Button(map_window, text="Find teak trees", command=predict_teak, font=("Arial", 12, "bold"), bg="green", fg="white")
    teak_seg_button.pack(pady=10)
    


    map_window.mainloop()



# Creating the main Tkinter window for LUMG
root = tk.Tk()
root.title("Land Use Map Generator")
root.geometry("800x700")  # Set width and height of the window
root.configure(bg="white")

# Creating a menu bar
menu_bar = tk.Menu(root)

# Creating a "File" menu
file_menu = tk.Menu(menu_bar, tearoff=0)
file_menu.add_command(label="Open", command=open_file)
file_menu.add_command(label="Save", command=save_file)
file_menu.add_separator()
file_menu.add_command(label="Exit", command=exit_app)

# Creating a "Option" menu
option_menu = tk.Menu(menu_bar, tearoff=0)
option_menu.add_command(label="Teak area finder", command=predict_teak)

# Adding the "File" menu to the menu bar
menu_bar.add_cascade(label="File", menu=file_menu)  
menu_bar.add_cascade(label="Options", menu=option_menu)

# Configure the root window to use the menu bar
root.config(menu=menu_bar)


# Loading the background banner image
background_image = resource_path("resources/app_banner-01.jpg")
background_image = Image.open(background_image)
background_image = background_image.resize((800, 77), Image.LANCZOS)  # Resize the image to fit the window
background_photo = ImageTk.PhotoImage(background_image)

# Creating a Label widget to display the background banner image
background_label = tk.Label(root, image=background_photo)
background_label.place(x=0, y=0, relwidth=1, relheight=0.07)  # Place the label at the top center

#titile lable for the main window 
label = tk.Label(root, text="Land-Use Map Generator (LUMG 0.1)", font=("Calibri", 20, "bold"))
label.configure(bg="white")
label.pack(pady=60)

# Creating a separator widget for the main window
separator = ttk.Separator(root, orient="horizontal")
separator.pack(fill="x")

# Defining the text for the label
label_text = """LUMG 0.1 stands as an automated Python application designed to effortlessly create basic land use maps through the power of AI technology. Leveraging a trained Keras U-Net model, this tool predicts six distinct land use classes based on a provided satellite image. To begin, simply select your satellite image in jpg format by clicking 'Browse Image', then proceed by clicking 'Generate your AI Land-use Map'."""


#create the 3rd lable widget
label = tk.Label(root, text=label_text, wraplength=400)
label.configure(bg="white")
label.pack()

# Function to browse and select an image
def browse_image():
    filename = filedialog.askopenfilename(title="Select Image File", filetypes=(("Image files", "*.jpg;*.jpeg;*.png"), ("All files", "*.*")))
    if filename:
        img_path.set(filename)
        # displaying the browsed image
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = Image.fromarray(img)
        img.thumbnail((200, 200))  # Resize image for display
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img

# Function to predict and display the result
def predict_and_display():
    img_path_str = img_path.get()
    if img_path_str:
        img = cv2.imread(img_path_str)

        # Disabling the predict_button and updating the label text
        predict_button.config(state=tk.DISABLED, text="Your Map is being processed...", bg="yellow")
        root.update()
        luse_model_path= resource_path("models/satellite_standard_unet_100epochs_9May2024.hdf5")
        model = load_model(luse_model_path, compile=False)

        # Size of patches
        patch_size = 256

        # Predicting patch by patch without smooth blending
        SIZE_X = (img.shape[1]//patch_size)*patch_size  # Nearest size divisible by the patch size
        SIZE_Y = (img.shape[0]//patch_size)*patch_size  # Nearest size divisible by the patch size
        large_img = Image.fromarray(img)
        large_img = large_img.crop((0 ,0, SIZE_X, SIZE_Y))  # Crop from top left corner
        large_img = np.array(large_img)     

        patches_img = patchify(large_img, (patch_size, patch_size, 3), step=patch_size)  # Step=patchsize means no overlap
        patches_img = patches_img[:,:,0,:,:,:]

        patched_prediction = []
        for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):
        
                single_patch_img = patches_img[i,j,:,:,:]
        
                # Using minmaxscaler instead of just dividing by the patch size. 
                single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
                single_patch_img = np.expand_dims(single_patch_img, axis=0)
                pred = model.predict(single_patch_img)
                pred = np.argmax(pred, axis=3)
                pred = pred[0, :,:]
                                 
                patched_prediction.append(pred)

        patched_prediction = np.array(patched_prediction)
        patched_prediction = np.reshape(patched_prediction, [patches_img.shape[0], patches_img.shape[1], 
                                            patches_img.shape[2], patches_img.shape[3]])

        unpatched_prediction = unpatchify(patched_prediction, (large_img.shape[0], large_img.shape[1]))

        # Displaying the prediction with a legend
        display_prediction_with_legend(unpatched_prediction)




def label_to_rgb(predicted_image):
    
    Building = '#3C1098'.lstrip('#')
    Building = np.array(tuple(int(Building[i:i+2], 16) for i in (0, 2, 4))) # 60, 16, 152
    
    Land = '#8429F6'.lstrip('#')
    Land = np.array(tuple(int(Land[i:i+2], 16) for i in (0, 2, 4))) #132, 41, 246
    
    Road = '#6EC1E4'.lstrip('#') 
    Road = np.array(tuple(int(Road[i:i+2], 16) for i in (0, 2, 4))) #110, 193, 228
    
    Vegetation =  'FEDD3A'.lstrip('#') 
    Vegetation = np.array(tuple(int(Vegetation[i:i+2], 16) for i in (0, 2, 4))) #254, 221, 58
    
    Water = 'E2A929'.lstrip('#') 
    Water = np.array(tuple(int(Water[i:i+2], 16) for i in (0, 2, 4))) #226, 169, 41
    
    Unlabeled = '#9B9B9B'.lstrip('#') 
    Unlabeled = np.array(tuple(int(Unlabeled[i:i+2], 16) for i in (0, 2, 4))) #155, 155, 155
    
    
    
    segmented_img = np.empty((predicted_image.shape[0], predicted_image.shape[1], 3))
    
    segmented_img[(predicted_image == 0)] = Building
    segmented_img[(predicted_image == 1)] = Land
    segmented_img[(predicted_image == 2)] = Road
    segmented_img[(predicted_image == 3)] = Vegetation
    segmented_img[(predicted_image == 4)] = Water
    segmented_img[(predicted_image == 5)] = Unlabeled
    
    segmented_img = segmented_img.astype(np.uint8)
    return(segmented_img)

def predict_buildings():
    img_path_str = img_path.get()
    if img_path_str:
        img = cv2.imread(img_path_str)

        # Disabling predict_button and updating label text
        building_seg_button.config(state=tk.DISABLED, text="Searching...", bg="yellow")
        root.update()

        building_model_path = resource_path("models/building_last.pt")
        model = YOLO(building_model_path)

        results = model.predict(img, conf=0.2)  # Passing the OpenCV image directly to the model

        for result in results:

            result.save(filename='result.jpg') 

            result_img = cv2.imread('result.jpg')

            plt.figure(figsize=(12, 6))
            plt.title('The Predicted Buildings')
            plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()


        # Re-enabling predict_button and update label text
        building_seg_button.config(state=tk.NORMAL, text="Find buildings",bg="green", fg="white")
# Image Browse button for the main window
browse_button = tk.Button(root, text="Browse Image", command=browse_image, font=("Arial", 12, "bold"), bg="#23395d", fg="white")
browse_button.pack(pady=15)

# Label to display selected image path in the main window
img_path = tk.StringVar()
selected_image_label = tk.Label(root, textvariable=img_path, bg="white")
selected_image_label.pack()

# Label to display the selected image in the main window
image_label = tk.Label(root)
image_label.pack()

# Predict button (keras unet model) to generate the land use map
predict_button = tk.Button(root, text="Generate your AI Land-use Map", command=predict_and_display, font=("Arial", 12, "bold"), bg="green", fg="white")
predict_button.pack(pady=10)

# Predict button (yolov8 nodel) to detect buildings 
building_seg_button = tk.Button(root, text="Find buildings", command=predict_buildings, font=("Arial", 12, "bold"), bg="green", fg="white")
building_seg_button.pack(pady=10)

# Function to display the prediction with a legend
def display_prediction_with_legend(predicted_image):
    prediction_without_smooth_blending = label_to_rgb(predicted_image)
    
    # Creating a custom legend
    legend_elements = [
        mpatches.Patch(color='#3C1098', label='Building'),
        mpatches.Patch(color='#8429F6', label='Land'),
        mpatches.Patch(color='#6EC1E4', label='Road'),
        mpatches.Patch(color='#FEDD3A', label='Vegetation'),
        mpatches.Patch(color='#E2A929', label='Water'),
        mpatches.Patch(color='#9B9B9B', label='Unlabeled')
    ]
    
    # Displaying the prediction with a legend
    plt.figure(figsize=(12, 6))
    plt.title('The Predicted Land Use Map')
    plt.imshow(prediction_without_smooth_blending)
    plt.axis('off')
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.show()

    # Re-enabling the predict_button and updating the label text
    predict_button.config(state=tk.NORMAL, text="Generate your AI Land-use Map",bg="green", fg="white")

root.mainloop()