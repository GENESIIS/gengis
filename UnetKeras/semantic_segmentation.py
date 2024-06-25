import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2DTranspose
import cv2
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify, unpatchify
from PIL import Image
import segmentation_models as sm
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
import keras
import tensorflow as tf

#print("Keras version:", keras.__version__)
#print("TensorFlow version:", tf.__version__)


#from simple_multi_unet_model import jacard_coef  


#img = cv2.imread("test_data/image_part_008.jpg", 1)
#original_mask = cv2.imread("test_data/mask_part_008.png", 1)
#original_mask = cv2.cvtColor(original_mask,cv2.COLOR_BGR2RGB)

img = cv2.imread("C:/Users/Dilshani/Documents/Genesiis.Py/Python new/python/segmentation/UNet_Keras/data/text_data/image_part_006.jpg", 1)
original_mask = cv2.imread("C:/Users/Dilshani/Documents/Genesiis.Py/Python new/python/segmentation/UNet_Keras/data/text_data/image_part_006.png", 1)
original_mask = cv2.cvtColor(original_mask,cv2.COLOR_BGR2RGB)


from keras.models import load_model


# Replace deprecated function call
tf.compat.v1.executing_eagerly_outside_functions()


model = load_model("C:/Users/Dilshani/Documents/Genesiis.Py/Python new/python/segmentation/UNet_Keras/model/standard_unet_model_100epochs_9May2024.hdf5", compile=False)
                  
# size of patches
patch_size = 256

# Number of classes 
n_classes = 6

         
#################################################################################
#Predict patch by patch with no smooth blending
###########################################

SIZE_X = (img.shape[1]//patch_size)*patch_size #Nearest size divisible by our patch size
SIZE_Y = (img.shape[0]//patch_size)*patch_size #Nearest size divisible by our patch size
large_img = Image.fromarray(img)
large_img = large_img.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
#image = image.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
large_img = np.array(large_img)     


patches_img = patchify(large_img, (patch_size, patch_size, 3), step=patch_size)  #Step=256 for 256 patches means no overlap
patches_img = patches_img[:,:,0,:,:,:]

patched_prediction = []
for i in range(patches_img.shape[0]):
    for j in range(patches_img.shape[1]):
        
        single_patch_img = patches_img[i,j,:,:,:]
        
        #Use minmaxscaler instead of just dividing by 255. 
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

plt.imshow(unpatched_prediction)
plt.axis('on')
###################################################################################
#Predict using smooth blending

input_img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)

####################################################
def predict_img_with_smooth_windowing(input_img, window_size, subdivisions, nb_classes, pred_func):
  import numpy as np
import scipy.signal
from tqdm import tqdm

def predict_img_with_smooth_windowing(input_img, window_size, subdivisions, nb_classes, pred_func):
    """
    Apply the `pred_func` function to square patches of the image, and overlap
    the predictions to merge them smoothly.
    """
    def _spline_window(window_size, power=2):
        """
        Squared spline (power=2) window function.
        """
        intersection = int(window_size/4)
        wind_outer = (abs(2*(scipy.signal.triang(window_size))) ** power)/2
        wind_outer[intersection:-intersection] = 0

        wind_inner = 1 - (abs(2*(scipy.signal.triang(window_size) - 1)) ** power)/2
        wind_inner[:intersection] = 0
        wind_inner[-intersection:] = 0

        wind = wind_inner + wind_outer
        wind = wind / np.average(wind)
        return wind

    def _window_2D(window_size, power=2):
        """
        Make a 1D window function, then infer and return a 2D window function.
        """
        wind = _spline_window(window_size, power)
        wind = np.expand_dims(np.expand_dims(wind, 1), 1)
        wind = wind * wind.transpose(1, 0, 2)
        return wind

    def _pad_img(img, window_size, subdivisions):
        """
        Add borders to img for a "valid" border pattern according to "window_size" and
        "subdivisions".
        """
        aug = int(round(window_size * (1 - 1.0/subdivisions)))
        more_borders = ((aug, aug), (aug, aug), (0, 0))
        ret = np.pad(img, pad_width=more_borders, mode='reflect')
        return ret

    def _unpad_img(padded_img, window_size, subdivisions):
        """
        Undo what's done in the `_pad_img` function.
        """
        aug = int(round(window_size * (1 - 1.0/subdivisions)))
        ret = padded_img[
            aug:-aug,
            aug:-aug,
            :
        ]
        return ret

    def _rotate_mirror_do(im):
        """
        Duplicate an np array (image) of shape (x, y, nb_channels) 8 times.
        """
        mirrs = []
        mirrs.append(np.array(im))
        mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=1))
        mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=2))
        mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=3))
        im = np.array(im)[:, ::-1]
        mirrs.append(np.array(im))
        mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=1))
        mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=2))
        mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=3))
        return mirrs

    def _rotate_mirror_undo(im_mirrs):
        """
        Merge a list of 8 np arrays (images) generated from the `_rotate_mirror_do` function.
        """
        origs = []
        origs.append(np.array(im_mirrs[0]))
        origs.append(np.rot90(np.array(im_mirrs[1]), axes=(0, 1), k=3))
        origs.append(np.rot90(np.array(im_mirrs[2]), axes=(0, 1), k=2))
        origs.append(np.rot90(np.array(im_mirrs[3]), axes=(0, 1), k=1))
        origs.append(np.array(im_mirrs[4])[:, ::-1])
        origs.append(np.rot90(np.array(im_mirrs[5]), axes=(0, 1), k=3)[:, ::-1])
        origs.append(np.rot90(np.array(im_mirrs[6]), axes=(0, 1), k=2)[:, ::-1])
        origs.append(np.rot90(np.array(im_mirrs[7]), axes=(0, 1), k=1)[:, ::-1])
        return np.mean(origs, axis=0)

    def _windowed_subdivs(padded_img, window_size, subdivisions, nb_classes, pred_func):
        """
        Create tiled overlapping patches.
        """
        WINDOW_SPLINE_2D = _window_2D(window_size=window_size, power=2)

        step = int(window_size/subdivisions)
        padx_len = padded_img.shape[0]
        pady_len = padded_img.shape[1]
        subdivs = []

        for i in range(0, padx_len-window_size+1, step):
            subdivs.append([])
            for j in range(0, pady_len-window_size+1, step):
                patch = padded_img[i:i+window_size, j:j+window_size, :]
                subdivs[-1].append(patch)

        subdivs = np.array(subdivs)
        a, b, c, d, e = subdivs.shape
        subdivs = subdivs.reshape(a * b, c, d, e)
        subdivs = pred_func(subdivs)
        subdivs = np.array([patch * WINDOW_SPLINE_2D for patch in subdivs])
        subdivs = subdivs.reshape(a, b, c, d, nb_classes)
        return subdivs

    def _recreate_from_subdivs(subdivs, window_size, subdivisions, padded_out_shape):
        """
        Merge tiled overlapping patches smoothly.
        """
        step = int(window_size/subdivisions)
        padx_len = padded_out_shape[0]
        pady_len = padded_out_shape[1]

        y = np.zeros(padded_out_shape)

        a = 0
        for i in range(0, padx_len-window_size+1, step):
            b = 0
            for j in range(0, pady_len-window_size+1, step):
                windowed_patch = subdivs[a, b]
                y[i:i+window_size, j:j+window_size] = y[i:i+window_size, j:j+window_size] + windowed_patch
                b += 1
            a += 1
        return y / (subdivisions ** 2)

    pad = _pad_img(input_img, window_size, subdivisions)
    pads = _rotate_mirror_do(pad)

    res = []
    for pad in tqdm(pads):
        sd = _windowed_subdivs(pad, window_size, subdivisions, nb_classes, pred_func)
        one_padded_result = _recreate_from_subdivs(
            sd, window_size, subdivisions,
            padded_out_shape=list(pad.shape[:-1])+[nb_classes])

        res.append(one_padded_result)

    padded_results = _rotate_mirror_undo(res)

    prd = _unpad_img(padded_results, window_size, subdivisions)

    prd = prd[:input_img.shape[0], :input_img.shape[1], :]

    return prd


#########################################################

# Use the algorithm. The `pred_func` is passed and will process all the image 8-fold by tiling small patches with overlap, called once with all those image as a batch outer dimension.
# Note that model.predict(...) accepts a 4D tensor of shape (batch, x, y, nb_channels), such as a Keras model.
predictions_smooth = predict_img_with_smooth_windowing(
    input_img,
    window_size=patch_size,
    subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
    nb_classes=n_classes,
    pred_func=(
        lambda img_batch_subdiv: model.predict(img_batch_subdiv)  # Remove extra parentheses around img_batch_subdiv
    )
)



final_prediction = np.argmax(predictions_smooth, axis=2)

###################
#Convert labeled images back to original RGB colored masks. 

def label_to_rgb(predicted_image):
    Building = '#3C1098'.lstrip('#')
    Building = np.array(tuple(int(Building[i:i+2], 16) for i in (0, 2, 4)))
    Land = '#8429F6'.lstrip('#')
    Land = np.array(tuple(int(Land[i:i+2], 16) for i in (0, 2, 4)))
    Road = '#6EC1E4'.lstrip('#')
    Road = np.array(tuple(int(Road[i:i+2], 16) for i in (0, 2, 4)))
    Vegetation =  'FEDD3A'.lstrip('#')
    Vegetation = np.array(tuple(int(Vegetation[i:i+2], 16) for i in (0, 2, 4)))
    Water = 'E2A929'.lstrip('#')
    Water = np.array(tuple(int(Water[i:i+2], 16) for i in (0, 2, 4)))
    Unlabeled = '#9B9B9B'.lstrip('#')
    Unlabeled = np.array(tuple(int(Unlabeled[i:i+2], 16) for i in (0, 2, 4)))
    segmented_img = np.empty((predicted_image.shape[0], predicted_image.shape[1], 3))
    segmented_img[(predicted_image == 0)] = Building
    segmented_img[(predicted_image == 1)] = Land
    segmented_img[(predicted_image == 2)] = Road
    segmented_img[(predicted_image == 3)] = Vegetation
    segmented_img[(predicted_image == 4)] = Water
    segmented_img[(predicted_image == 5)] = Unlabeled
    segmented_img = segmented_img.astype(np.uint8)
    return(segmented_img)

prediction_with_smooth_blending = label_to_rgb(final_prediction)
prediction_without_smooth_blending = label_to_rgb(unpatched_prediction)

# Define the legend
import matplotlib.patches as mpatches

legend_elements = [
    mpatches.Patch(color='#3C1098', label='Building'),
    mpatches.Patch(color='#8429F6', label='Land'),
    mpatches.Patch(color='#6EC1E4', label='Road'),
    mpatches.Patch(color='#FEDD3A', label='Vegetation'),
    mpatches.Patch(color='#E2A929', label='Water'),
    mpatches.Patch(color='#9B9B9B', label='Unlabeled')
]

plt.figure(figsize=(12, 12))
plt.subplot(221)
plt.title('Testing Image')
plt.imshow(img)
plt.subplot(222)
plt.title('Testing Label')
plt.imshow(original_mask)
plt.subplot(223)
plt.title('Prediction without smooth blending')
plt.imshow(prediction_without_smooth_blending)
plt.subplot(224)
plt.title('Prediction with smooth blending')
plt.imshow(prediction_with_smooth_blending)

# Add the legend
plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.5, 1))
plt.show()