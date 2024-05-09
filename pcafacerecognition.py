import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
import os

TARGET_SIZE = (64, 64)
N_VALUES = [20, 30, 40, 50]

def preprocess_image(image_path, target_size=TARGET_SIZE):

    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flattened_image = gray_image.flatten()
    return flattened_image

def preprocess_dataset(dataset_dir):

    all_images = []
    for file in os.listdir(dataset_dir):
        if file.endswith(".jpg") or file.endswith(".png"):
            image_path = os.path.join(dataset_dir, file)
            flattened_image = preprocess_image(image_path)
            all_images.append(flattened_image)
    all_images = np.array(all_images)
    return all_images

def perform_pca(data, n_components=None):

    mean_face = np.mean(data, axis=0)
    centered_data = data - mean_face
    covariance_matrix = np.cov(centered_data, rowvar=False)
    pca = PCA(n_components=n_components)
    pca.fit(centered_data)
    return pca, mean_face

dataset_dir = "path_to_celeba_dataset_directory"
celeba_data = preprocess_dataset(dataset_dir)

pca, mean_face = perform_pca(celeba_data)

def reconstruct_faces(pca, mean_face, data, n_components):
    projected_data = pca.transform(data)[:, :n_components]
    reconstructed_faces = np.dot(projected_data, pca.components_[:n_components, :])
    reconstructed_faces += mean_face
    return reconstructed_faces

def visualize_faces(original_faces, reconstructed_faces, n_components):
    num_samples = min(5, original_faces.shape[0])
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 10))
    fig.suptitle(f"Face Reconstruction with {n_components} Components", fontsize=16)
    for i in range(num_samples):
        axes[i, 0].imshow(original_faces[i].reshape(64, 64), cmap='gray')
        axes[i, 0].set_title("Original Face")
        axes[i, 0].axis('off')
        axes[i, 1].imshow(reconstructed_faces[i].reshape(64, 64), cmap='gray')
        axes[i, 1].set_title("Reconstructed Face")
        axes[i, 1].axis('off')
    plt.tight_layout()
    plt.show()

n_values = [20, 30, 40, 50]
for n in n_values:
    reconstructed_faces = reconstruct_faces(pca, mean_face, celeba_data, n)
    visualize_faces(celeba_data, reconstructed_faces, n)


def preprocess_your_face(image_path, target_size=(64, 64)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flattened_image = gray_image.flatten()
    return flattened_image

your_face_path = "path_to_your_face_image"
your_face = preprocess_your_face(your_face_path)

def project_your_face(pca, mean_face, your_face):
    centered_your_face = your_face - mean_face
    your_face_coefficients = pca.transform(centered_your_face.reshape(1, -1))
    return your_face_coefficients

your_face_coefficients = project_your_face(pca, mean_face, your_face)

def find_closest_face(pca, celeba_data, your_face_coefficients, metric='euclidean'):
    celeba_coefficients = pca.transform(celeba_data)
    if metric == 'euclidean':
        distances = np.linalg.norm(celeba_coefficients - your_face_coefficients, axis=1)
    elif metric == 'cosine':
        distances = 1 - cosine_similarity(celeba_coefficients, your_face_coefficients)
    closest_index = np.argmin(distances)
    return closest_index

closest_index = find_closest_face(pca, celeba_data, your_face_coefficients)
print("Index of closest celebrity face:", closest_index)

def generate_random_face_vectors(n_components, n_samples=5, scale=2):
    random_coefficients = np.random.normal(0, scale, size=(n_samples, n_components))
    generated_faces = np.dot(random_coefficients, pca.components_[:n_components, :])
    generated_faces += mean_face
    return generated_faces

def visualize_generated_faces(generated_faces, n_samples):
    fig, axes = plt.subplots(1, n_samples, figsize=(12, 6))
    fig.suptitle("Randomly Generated Faces", fontsize=16)
    for i in range(n_samples):
        axes[i].imshow(generated_faces[i].reshape(64, 64), cmap='gray')
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

n_components = 50
generated_faces = generate_random_face_vectors(n_components)

visualize_generated_faces(generated_faces, n_samples=5)

def evaluate_reconstruction(original_faces, reconstructed_faces):
    mse = mean_squared_error(original_faces, reconstructed_faces)
    return mse

def visualize_original_and_reconstructed(original_faces, reconstructed_faces, n_samples):
    fig, axes = plt.subplots(2, n_samples, figsize=(12, 6))
    fig.suptitle("Original and Reconstructed Faces", fontsize=16)
    for i in range(n_samples):
        axes[0, i].imshow(original_faces[i].reshape(64, 64), cmap='gray')
        axes[0, i].set_title("Original")
        axes[0, i].axis('off')
        axes[1, i].imshow(reconstructed_faces[i].reshape(64, 64), cmap='gray')
        axes[1, i].set_title("Reconstructed")
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.show()

reconstructed_faces = reconstruct_faces(pca, mean_face, celeba_data, n_components)
mse = evaluate_reconstruction(celeba_data, reconstructed_faces)
print("Mean Squared Error (MSE) for reconstruction:", mse)


visualize_original_and_reconstructed(celeba_data, reconstructed_faces, n_samples=5)