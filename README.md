<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
  
</head>
<body>

<h1>Face Reconstruction with PCA</h1>

<h2>Introduction</h2>

This project demonstrates face reconstruction using Principal Component Analysis (PCA). It includes methods for preprocessing images, performing PCA, reconstructing faces, projecting faces onto the PCA space, and generating random faces.

<h2>Usage</h2>

1. **Preprocessing Dataset:**
   - `preprocess_image(image_path, target_size=TARGET_SIZE)`: Preprocesses a single image.
   - `preprocess_dataset(dataset_dir)`: Preprocesses a directory of images.

2. **Perform PCA:**
   - `perform_pca(data, n_components=None)`: Performs PCA on the provided dataset.

3. **Reconstructing Faces:**
   - `reconstruct_faces(pca, mean_face, data, n_components)`: Reconstructs faces using PCA components.

4. **Visualizing Faces:**
   - `visualize_faces(original_faces, reconstructed_faces, n_components)`: Visualizes original and reconstructed faces.

5. **Projecting Your Face:**
   - `project_your_face(pca, mean_face, your_face)`: Projects a new face onto the PCA space.

6. **Finding Closest Face:**
   - `find_closest_face(pca, celeba_data, your_face_coefficients, metric='euclidean')`: Finds the closest face in the dataset to a given face.

7. **Generating Random Faces:**
   - `generate_random_face_vectors(n_components, n_samples=5, scale=2)`: Generates random faces using PCA components.

8. **Evaluation:**
   - `evaluate_reconstruction(original_faces, reconstructed_faces)`: Evaluates face reconstruction using Mean Squared Error (MSE).

9. **Visualizing Original and Reconstructed Faces:**
   - `visualize_original_and_reconstructed(original_faces, reconstructed_faces, n_samples)`: Visualizes original and reconstructed faces side by side.

<h2>Example Usage</h2>

```python
# Perform PCA on CelebA dataset
pca, mean_face = perform_pca(celeba_data)

# Reconstruct faces with 50 components
n_components = 50
reconstructed_faces = reconstruct_faces(pca, mean_face, celeba_data, n_components)

# Visualize original and reconstructed faces
visualize_original_and_reconstructed(celeba_data, reconstructed_faces, n_samples=5)

```
<h2>Dependencies</h2>
OpenCV (cv2)
NumPy (numpy)
Matplotlib (matplotlib)
Scikit-learn (sklearn)
