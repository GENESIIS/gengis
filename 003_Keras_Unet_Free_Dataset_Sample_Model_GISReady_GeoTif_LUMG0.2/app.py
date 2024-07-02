import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import os
import tensorflow as tf
from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from patchify import patchify, unpatchify
from sklearn.preprocessing import MinMaxScaler
import rasterio
from rasterio.features import shapes
import fiona
from fiona.crs import from_epsg
from shapely.geometry import shape, Polygon, mapping

# Disable TensorFlow optimizations for oneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# MinMaxScaler for image normalization
scaler = MinMaxScaler()

# Functions
def browse_image():
    filename = filedialog.askopenfilename(title="Select Image File", filetypes=(("Image files", "*.tif"), ("All files", "*.*")))
    if filename:
        img_path.set(filename)
        with rasterio.open(filename) as dataset:
            imgr = dataset.read([1, 2, 3]).transpose((1, 2, 0))
            global meta, transform
            meta = dataset.meta
            transform = dataset.transform
        imgr = cv2.cvtColor(imgr, cv2.COLOR_BGR2RGB)
        imgr = Image.fromarray(imgr)
        imgr.thumbnail((400, 400))
        imgr = ImageTk.PhotoImage(imgr)
        image_label.config(image=imgr)
        image_label.image = imgr

def predict_and_display():
    img_path_str = img_path.get()
    if img_path_str:
        with rasterio.open(img_path_str) as dataset:
            imgr = dataset.read([1, 2, 3]).transpose((1, 2, 0))

        predict_button.config(state=tk.DISABLED, text="Processing...", bg="yellow")
        root.update()

        model = load_model(r"C:\practice\Python_Projects\Keras Unet\Dubai_dataset\satellite_standard_unet_100epochs_9May2024.hdf5", compile=False)

        patch_size = 256
        SIZE_X = (imgr.shape[1] // patch_size) * patch_size
        SIZE_Y = (imgr.shape[0] // patch_size) * patch_size
        large_img = Image.fromarray(imgr).crop((0, 0, SIZE_X, SIZE_Y))
        large_img = np.array(large_img)

        patches_img = patchify(large_img, (patch_size, patch_size, 3), step=patch_size)
        patches_img = patches_img[:, :, 0, :, :, :]

        patched_prediction = []
        for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):
                single_patch_img = patches_img[i, j, :, :, :]
                single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
                single_patch_img = np.expand_dims(single_patch_img, axis=0)
                pred = model.predict(single_patch_img)
                pred = np.argmax(pred, axis=3)[0, :, :]
                patched_prediction.append(pred)

        patched_prediction = np.array(patched_prediction)
        patched_prediction = np.reshape(patched_prediction, (patches_img.shape[0], patches_img.shape[1], patch_size, patch_size))
        unpatched_prediction = unpatchify(patched_prediction, (large_img.shape[0], large_img.shape[1]))

        unpatched_prediction = unpatched_prediction.astype(np.int32)
        display_prediction_with_legend(unpatched_prediction)

def label_to_rgb(predicted_image):
    color_map = {
        0: '#3C1098',  # Building
        1: '#8429F6',  # Land
        2: '#6EC1E4',  # Road
        3: '#FEDD3A',  # Vegetation
        4: '#E2A929',  # Water
        5: '#9B9B9B'   # Unlabeled
    }
    segmented_img = np.zeros((predicted_image.shape[0], predicted_image.shape[1], 3), dtype=np.uint8)
    for label, color in color_map.items():
        color = color.lstrip('#')
        color = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
        segmented_img[predicted_image == label] = color
    return segmented_img

def save_as_geotiff(meta, transform, img, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    meta.update({
        "driver": "GTiff",
        "height": img.shape[0],
        "width": img.shape[1],
        "count": 3,
        "dtype": "uint8",
        "transform": transform,
        "crs": meta['crs']  # Ensure CRS is included
    })
    with rasterio.open(output_path, "w", **meta) as dest:
        for i in range(3):
            dest.write(img[:, :, i], i + 1)

def save_as_shapefile(predicted_image, transform, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    shapes_gen = shapes(predicted_image, transform=transform)
    schema = {'geometry': 'Polygon', 'properties': {'class': 'int'}}
    
    # Ensure the correct CRS is used
    with fiona.open(output_path, 'w', driver='ESRI Shapefile', crs=meta['crs'], schema=schema) as shp:
        for geom, value in shapes_gen:
            shp.write({'geometry': mapping(shape(geom)), 'properties': {'class': int(value)}})

def display_prediction_with_legend(predicted_image):
    segmented_img_rgb = label_to_rgb(predicted_image)
    legend_elements = [
        mpatches.Patch(color='#3C1098', label='Building'),
        mpatches.Patch(color='#8429F6', label='Land'),
        mpatches.Patch(color='#6EC1E4', label='Road'),
        mpatches.Patch(color='#FEDD3A', label='Vegetation'),
        mpatches.Patch(color='#E2A929', label='Water'),
        mpatches.Patch(color='#9B9B9B', label='Unlabeled')
    ]
    plt.figure(figsize=(12, 6))
    plt.title('The Predicted Land Use Map')
    plt.imshow(segmented_img_rgb)
    plt.axis('off')
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.show()

    output_geotiff_path = r"C:\practice\Python_Projects\Keras Unet\Dubai_dataset\lumg_result.tif"
    output_shapefile_path = r"C:\practice\Python_Projects\Keras Unet\Dubai_dataset\lumg_result.shp"

    save_as_geotiff(meta, transform, segmented_img_rgb, output_geotiff_path)
    save_as_shapefile(predicted_image, transform, output_shapefile_path)

    predict_button.config(state=tk.NORMAL, text="Generate your AI Land-use Map", bg="green", fg="white")

# GUI Setup
root = tk.Tk()
root.title("Land Use Map Generator")
root.geometry("800x700")
root.configure(bg="white")

menu_bar = tk.Menu(root)
file_menu = tk.Menu(menu_bar, tearoff=0)
file_menu.add_command(label="Open", command=browse_image)
file_menu.add_separator()
file_menu.add_command(label="Exit", command=root.quit)
menu_bar.add_cascade(label="File", menu=file_menu)
root.config(menu=menu_bar)

background_image = Image.open(r"C:\practice\Python_Projects\Keras Unet\Dubai_dataset\app_banner-01.jpg")
background_image = background_image.resize((800, 77), Image.LANCZOS)
background_photo = ImageTk.PhotoImage(background_image)
background_label = tk.Label(root, image=background_photo)
background_label.place(x=0, y=0, relwidth=1, relheight=0.07)

label = tk.Label(root, text="Land-Use Map Generator (LUMG 0.2)", font=("Calibri", 20, "bold"), bg="white")
label.pack(pady=60)

separator = ttk.Separator(root, orient="horizontal")
separator.pack(fill="x")

label_text = """LUMG 0.2 is the higher version of LUMG 0.1 which is especially developed to handle GeoTif datasets. The new version is capable of reading GeoTif satellite images and provides GIS-Ready resuluts for given satellite images. LUMG is originally desinged as an automated Python application to effortlessly create basic land use maps through the power of AI technology. Leveraging a trained Keras U-Net model, this tool predicts six distinct land use classes based on a provided satellite image. To begin, simply select your satellite image in jpg format by clicking 'Browse Image', then proceed by clicking 'Generate your AI Land-use Map'."""
description_label = tk.Label(root, text=label_text, wraplength=400, bg="white")
description_label.pack()

browse_button = tk.Button(root, text="Browse Image", command=browse_image, font=("Arial", 12, "bold"), bg="#23395d", fg="white")
browse_button.pack(pady=15)

img_path = tk.StringVar()
selected_image_label = tk.Label(root, textvariable=img_path, bg="white")
selected_image_label.pack()

image_label = tk.Label(root)
image_label.pack()

predict_button = tk.Button(root, text="Generate your AI Land-use Map", command=predict_and_display, font=("Arial", 14, "bold"), bg="green", fg="white")
predict_button.pack(pady=30)

footer_label = tk.Label(root, text="Developed by GENESIIS Software (pvt)Ltd. | Powered by Python", font=("Arial", 10), bg="white")
footer_label.pack(side="bottom", pady=5)

root.mainloop()
