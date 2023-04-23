import pydicom as dicom
import numpy as np
import os
import skimage
import matplotlib.pyplot as plt
from stl import mesh
from skimage import measure
from sklearn.cluster import KMeans 

def AddImage(path, output_shape):
  ds = dicom.dcmread(path)

  image = ds.pixel_array
  image = image.astype(np.int16)
  image[image == -2000] = 0
  
  # Convert to Hounsfield units (HU)
  intercept = ds.RescaleIntercept
  slope = ds.RescaleSlope
  
  if slope != 1:
      image = slope * image.astype(np.float64)
      image = image.astype(np.int16)
      
  image += np.int16(intercept)
  image =  np.array(image, dtype=np.float32)
  #resize for more faster nn learning
  image = skimage.transform.resize(image, output_shape)
  return image

def AddImageToArray(x,folderPath, output_shape = (512, 512)):
  patchs = os.listdir(folderPath)
  for filename in sorted(patchs, key=len):
    f = os.path.join(folderPath, filename)
    if os.path.isfile(f):
      x.append(AddImage(os.path.join(folderPath, filename), output_shape))
    else:
      AddImageToArray(x,f)

def DrawImages(images, count = 4, grid_size = (2,2), offset = 0):
  k = offset
  for i in range(count):
    plt.subplot(grid_size[0], grid_size[1], i+1)
    plt.imshow(images[k], cmap='gray')
    k+=1
  plt.show()

def make_mesh(image, threshold=-300, step_size=1):
    p = image.transpose(2,1,0)
    verts, faces, norm, val = measure.marching_cubes(p, threshold, step_size=step_size, allow_degenerate=True) 
    return verts, faces

def plt_3d(verts, faces, filename = 'skull.stl'):
    cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            cube.vectors[i][j] = verts[f[j],:]
    
    cube.save(filename)

def segmentation(images, left_offset, right_offset):
    width = images.shape[1]
    height = images.shape[2]
    x_transforms = np.where(images > 100, 1, 0)
    x_transforms = np.array(x_transforms, np.float32)
    for img in x_transforms:
        for i in range(0, height):
            for j in range(0, left_offset):
                img[i,j] = 0
        for i in range(0, height):
            for j in range(width - right_offset, width):
                img[i,j] = 0
    return np.array(x_transforms)