# Stride-and-Slice-Images - 2D & 3D images

This is an extension of the original code to account for 3D volumetric images. The original code by AnmolChachra
(https://github.com/AnmolChachra/Stride-and-Slice-Images) uses OpenCV to read and write images. This code uses Nibabel
and tifffile packages in addition to OpenCV.

---

Below is the original documentation on Stride-and-Slice-Images (with some minor modifications):

This blog will explain the code and equations mentioned in the code. Give it a read before going ahead.
Blog - https://codeitplease.wordpress.com/2018/02/26/striding-and-slicing-images/

Slice an image into smaller images (overlapping or non-overlapping). This is a custom built script without the use of any predefined function. The algorithm that will help an image analysis enthusiast to understand how striding, padding and convolution works.

The script contains a 'transform' function that takes 'source_dir', 'size', 'strides', 'PADDING' as variables and then returns a dictionary with key - string of count starting from 1 e.g '1','2' and value - list of np.ndarray types of output images e.g. [np.ndarray(size),]

'source_dir' - Can be a directory path to a single image or a directory path to a directory containg multiple images.

'size' - tuple of desired height and width e.g. (100,100) in case of 2D and (100, 100, 100) in case of 3D

'strides' - tuple of desired stride along height and stride along width, e.g. (100,100) in case of 2D and (100, 100, 100) in case of 3D


'PADDING' - (default False) If set True will calculate appropriate padding that will give you complete images with respect to the given strides.

To save the output images in a file, the script also provides a 'save_images' method that will take the result of 'transform' method as input and save all the images according to their respective main images with respective serial number as folder name.

How to run the script?
-->Run the scipt using idle-
  - Run the script in idle.
    - transform()
-->Import the script in python shell-
  - Run python shell in the directory where the script is stored
    - import strideslice as ss
    - ss.transform()
 
---
# Example data:
1) A volumetric CT data from The Stanford volume data archive (https://graphics.stanford.edu/data/voldata/)
in TIFF format.

Other 2D and 3D example data will be added in near future.