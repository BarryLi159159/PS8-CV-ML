import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import re
import random
from os import listdir


"""
TODO 0: Find out the image shape as a tuple and include it in your report.
"""
def find_image_shape(data_dir):
    # Get the list of files in the directory
    file_list = [fname for fname in listdir(data_dir) if fname.endswith('.pgm')]
    if not file_list:
        raise ValueError("No images found in the directory.")

    # shape of the first image
    sample_image = cv2.imread(data_dir + file_list[0], 0)  # Read in grayscale
    if sample_image is None:
        raise ValueError("Unable to read the sample image.")
    
    return sample_image.shape


data_directory = '/Users/li/Desktop/cmu文件/cv/ps8/lfw_crop/' 
IMG_SHAPE = find_image_shape(data_directory)
print("Image Shape:", IMG_SHAPE)



def load_data(data_dir, top_n=10):
    """
    Load the data and return a list of images and their labels.

    :param data_dir: The directory where the data is located
    :param top_n: The number of people with the most images to use

    Suggested return values, feel free to change as you see fit
    :return data_top_n: A list of images of only people with top n number of images
    :return target_top_n: Corresponding labels of the images
    :return target_names: A list of all labels(names)
    :return target_count: A dictionary of the number of images per person
    """
    # read and randomize list of file names
    file_list = [fname for fname in listdir(data_dir) if fname.endswith('.pgm')]
    random.shuffle(file_list)
    name_list = [re.sub(r'_\d{4}.pgm', '', name).replace('_', ' ') for name in file_list]

    # get a list of all labels
    target_names = sorted(list(set(name_list)))

    # get labels for each image
    target = np.array([target_names.index(name) for name in name_list])

    # read in all images
    data = np.array([cv2.imread(data_dir + fname, 0) for fname in file_list])
    """
    TODO 1: Only preserve images of 10 people with the highest occurence, then plot 
            a histogram of the number of images per person in the preserved dataset.
            Include the histogram in your report.
    """
    # YOUR CODE HERE
    # target_count is a dictionary of the number of images per person
    # where the key is an index to label ('target'), and the value is the number of images
    # Try to use sorted() to sort the dictionary by value, then only keep the first 10 items of the output list.
    target_count = {}

    # data_top_n is a list of labels of only people with top n number of images
    target_top_n = []
    data_top_n = []
    # Count the number of images per person
    target_count = {name: name_list.count(name) for name in target_names}

    # Sort by count and get the top N people
    sorted_count = sorted(target_count.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_n_names = [item[0] for item in sorted_count]

    # Filter the dataset for the top N people
    
    for i, name in enumerate(name_list):
        if name in top_n_names:
            image = cv2.imread(data_dir + file_list[i], 0)  # Read in grayscale
            data_top_n.append(image)
            target_top_n.append(target_names.index(name))

    data_top_n = np.array(data_top_n)
    # Plot the histogram for top N people
    plt.figure(figsize=(10, 6))
    plt.bar(
        [name.replace(' ', '\n') for name in top_n_names],
        [count for _, count in sorted_count],
    )
    plt.xlabel("Person")
    plt.ylabel("Number of Images")
    plt.title(f"Number of Images for Top {top_n} People")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return data_top_n, target_top_n, target_names, target_count
    
def load_data_nonface(data_dir):
    """
    Your can write your functin comments here.
    """
    
    """
    TODO 2: Load the nonface data and return a list of images.
    """
    # YOUR CODE HERE
    # Take a look at the load_data() function for reference

    # Get the list of files in the directory
    file_list = [fname for fname in listdir(data_dir) if fname.endswith('.png')]
    if not file_list:
        raise ValueError("No non-face images found in the directory.")
    
    # Load images as a NumPy array
    data = np.array([cv2.imread(data_dir + fname, 0) for fname in file_list])
    print(f"Loaded {len(data)} non-face images from {data_dir}")
    return data



nonface_data_dir = "/Users/li/Desktop/cmu文件/cv/ps8/imagenet_val1000_downsampled/"  # Replace with your actual directory path
data_nonface = load_data_nonface(nonface_data_dir)
print("Non-face data shape:", data_nonface.shape)


def perform_pca(data_train, data_test, data_noneface, n_components, plot_PCA=False):
    """
    Your can write your functin comments here.
    """

    """
    TODO 3: Perform PCA on the training data, then transform the training, testing, 
            and nonface data. Return the transformed data. This includes:
            a) Flatten the images if you haven't done so already
            b) Standardize the data (0 mean, unit variance)
            c) Perform PCA on the standardized training data
            d) Transform the standardized training, testing, and nonface data
            e) Plot the transformed training and nonface data using the first three
               principal components if plot_PCA is True. Include the plots in your report.
            f) Return the principal components and transformed training, testing, and nonface data
    """
    # YOUR CODE HERE
    # You can use the StandardScaler() function to standardize the data
    #data_train_centered = None
    #data_test_centered = None
    #data_noneface_centered = None

    # You can use the decomposition.PCA() and function to perform PCA
    # You can check the example code in the documentation using the links below
    # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    #pca = None

    # You can use the pca.transform() function to transform the data
    #data_train_pca = None
    #data_test_pca = None
    #data_noneface_pca = None

    # You can use the scatter3D() function to plot the transformed data
    # Please not that 3 principal components may not be enough to separate the data
    # So your plot of face and nonface data may not be clearly separated
    # if plot_PCA:
    # Flatten the images (samples × features)
    data_train_flat = data_train.reshape(data_train.shape[0], -1)
    data_test_flat = data_test.reshape(data_test.shape[0], -1)
    data_nonface_flat = data_nonface.reshape(data_nonface.shape[0], -1)

    # Standardize the data
    scaler = StandardScaler()
    data_train_scaled = scaler.fit_transform(data_train_flat)
    data_test_scaled = scaler.transform(data_test_flat)
    data_nonface_scaled = scaler.transform(data_nonface_flat)

    # Perform PCA
    pca = PCA(n_components=n_components)
    data_train_pca = pca.fit_transform(data_train_scaled)
    data_test_pca = pca.transform(data_test_scaled)
    data_nonface_pca = pca.transform(data_nonface_scaled)

    # Plot PCA results 
    if plot_PCA and n_components >= 8:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(
            data_train_pca[:, 0], data_train_pca[:, 1], data_train_pca[:, 2],
            c='blue', label='Training Data', alpha=0.5
        )
        ax.scatter(
            data_nonface_pca[:, 0], data_nonface_pca[:, 1], data_nonface_pca[:, 2],
            c='red', label='Non-Face Data', alpha=0.5
        )
        ax.set_title("3D Scatter Plot of PCA Results")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_zlabel("Principal Component 3")
        ax.legend()
        plt.show()

    return pca, data_train_pca, data_test_pca, data_nonface_pca


def plot_eigenfaces(pca):
    """
    TODO 4: Plot the first 8 eigenfaces. Include the plot in your report.
    """
    n_row = 2
    n_col = 4
    fig, axes = plt.subplots(n_row, n_col, figsize=(12, 6))
    for i in range(n_row * n_col):
        # YOUR CODE HERE
        # The eigenfaces are the principal components of the training data
        # Since we have flattened the images, you can use reshape() to reshape to the original image shape
        eigenface = pca.components_[i].reshape(IMG_SHAPE)  # Reshape to the original image dimensions
        ax = axes[i // n_col, i % n_col]
        ax.imshow(eigenface, cmap='gray')
        ax.set_title(f"Eigenface {i + 1}")
        ax.axis('off')
        pass
    plt.show()


def train_classifier(data_train_pca, target_train):
    """
    TODO 5: OPTIONAL: Train a classifier on the training data.
            SVM is recommended, but feel free to use any classifier you want.
            Also try using the RandomizedSearchCV to find the best hyperparameters.
            Include the classifier you used as well as the parameters in your report.
            Feel free to look up sklearn documentation and examples on usage of classifiers.
    """
    # YOUR CODE HERE
    # You can read the documents from sklearn to learn about the classifiers provided by sklearn
    # https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
    # If you are using SVM, you can also check the example below
    # https://scikit-learn.org/stable/modules/svm.html
    # Also, you can use the RandomizedSearchCV to find the best hyperparameters
    # Define the SVM classifier
    clf = SVC(kernel='rbf', random_state=42)

    # Define the hyperparameter search space
    param_distributions = {
        'C': loguniform(1e-3, 1e3),  # Regularization parameter
        'gamma': loguniform(1e-4, 1e0),  # Kernel coefficient
    }

    # Use RandomizedSearchCV for hyperparameter tuning
    random_search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_distributions,
        n_iter=100,  # Number of parameter settings sampled
        cv=5,  # 5-fold cross-validation
        scoring='accuracy',
        random_state=42,
        n_jobs=-1,  # Use all available cores
    )

    # Fit the classifier to the training data
    random_search.fit(data_train_pca, target_train)

    # Print the best parameters and score
    print("Best parameters found:", random_search.best_params_)
    print("Best cross-validation accuracy:", random_search.best_score_)

    # Return the trained classifier
    return random_search.best_estimator_

def plot_confusion_matrix(classifier, data_test_pca, target_test):

# Predict on test data
    pred = classifier.predict(data_test_pca)

    # Compute the confusion matrix
    cm = confusion_matrix(target_test, pred)

    # Plot the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.show()

def evaluate_pca_components(data_train, data_test, target_train, target_test, n_components_list):

    accuracies = []

    for n_components in n_components_list:
        print(f"Evaluating PCA with {n_components} components...")
        
        # Perform PCA with n_components
        pca, data_train_pca, data_test_pca, _ = perform_pca(
            data_train, data_test, np.zeros_like(data_test), n_components=n_components, plot_PCA=True
        )
        
        # Train the classifier
        classifier = train_classifier(data_train_pca, target_train)
        
        # Evaluate the classifier
        pred = classifier.predict(data_test_pca)
        accuracy = np.mean(pred == target_test)
        accuracies.append(accuracy)

        print(f"Accuracy with {n_components} components: {accuracy:.4f}")
    
    # Plot accuracy vs. number of PCA components
    plt.figure(figsize=(10, 6))
    plt.plot(n_components_list, accuracies, marker='o')
    plt.xlabel("Number of PCA Components")
    plt.ylabel("Classification Accuracy")
    plt.title("Accuracy vs. Number of PCA Components")
    plt.grid(True)
    plt.show()

    return accuracies

if __name__ == '__main__':
    """
    Load the data
    Face Dataset from https://conradsanderson.id.au/lfwcrop/
    Modified from original dataset http://vis-www.cs.umass.edu/lfw/
    Noneface Dataset modified from http://image-net.org/download-images
    All modified datasets are available in the Box folder
    """
    data, target, target_names, target_count = load_data('lfw_crop/', top_n=10)
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.25, random_state=42)
    data_noneface = load_data_nonface('imagenet_val1000_downsampled/')
    print("Total dataset size:", data.shape[0])
    print("Training dataset size:", data_train.shape[0])
    print("Test dataset size:", data_test.shape[0])
    print("Nonface dataset size:", data_noneface.shape[0])

    # Perform PCA, you can change the number of components as you wish
    pca, data_train_pca, data_test_pca, data_noneface_pca = perform_pca(
        data_train, data_test, data_noneface, n_components=20, plot_PCA=False
    )

    # Plot the first 8 eigenfaces. To do this, make sure n_components is at least 8
    plot_eigenfaces(pca)

    """
    Start of PS 8-2
    This part is optional. You will get extra credits if you complete this part.
    """
    
    # Train a classifier on the transformed training data
    classifier = train_classifier(data_train_pca, target_train)

    # Evaluate the classifier
    pred = classifier.predict(data_test_pca)
    # Use a simple percentage of correct predictions as the metric
    accuracy = np.count_nonzero(np.where(pred == target_test)) / pred.shape[0]
    print("Accuracy:", accuracy)
    """
    TODO 6: OPTIONAL: Plot the confusion matrix of the classifier.
            Include the plot and accuracy in your report.
            You can use the sklearn.metrics.ConfusionMatrixDisplay function.
    """
    plot_confusion_matrix(classifier, data_test_pca, target_test)


    """
    TODO 7: OPTIONAL: Plot the accuracy with different number of principal components.
            This might take a while to run. Feel free to decrease training iterations if
            you want to speed up the process. We won't set a hard threshold on the accuracy.
            Include the plot in your report.
    """
    n_components_list = [3, 5, 10, 20, 40, 60, 80, 100, 120, 130]
    # YOUR CODE HERE
    accuracies = evaluate_pca_components(data_train, data_test, target_train, target_test, n_components_list)
