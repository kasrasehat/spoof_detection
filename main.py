import numpy as np
from skimage import feature
import cv2
from skimage.feature import local_binary_pattern
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks
from skimage.feature import hog
from scipy.ndimage import convolve
from skimage.feature import local_binary_pattern
import pywt


def lbp_histogram_features(image, P=8, R=1):
    """
    Compute mean, variance, skewness, and kurtosis of the LBP histogram of an image.

    Parameters:
    image (2D array): Grayscale image.
    P (int): Number of circularly symmetric neighbor set points.
    R (float): Radius of circle.

    Returns:
    tuple: Mean, variance, skewness, and kurtosis of the LBP histogram.
    """
    lbp = local_binary_pattern(image, P, R, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))

    # Normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)

    # Calculate statistical measures
    mean = np.mean(hist)
    variance = np.var(hist)
    skewness = skew(hist)
    kurt = kurtosis(hist)

    return mean, variance, skewness, kurt


def lbp_uniformity(image, P=8, R=1):
    """
    Compute the LBP uniformity score for an image.

    Parameters:
    image (2D array): Grayscale image.
    P (int): Number of circularly symmetric neighbor set points.
    R (float): Radius of circle.

    Returns:
    float: LBP uniformity score.
    """
    lbp = feature.local_binary_pattern(image, P, R, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))

    # Normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)

    # Calculate the uniformity
    uniformity = np.sum(hist**2)
    return uniformity


def lbp_histogram_top_peaks(image, P=8, R=1, num_peaks=3):
    """
    Compute the indices of the top peaks in the LBP histogram of an image.

    Parameters:
    image (2D array): Grayscale image.
    P (int): Number of circularly symmetric neighbor set points.
    R (float): Radius of circle.
    num_peaks (int): Number of top peaks to return.

    Returns:
    array: Indices of the top histogram peaks.
    """
    lbp = local_binary_pattern(image, P, R, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))

    # Find peaks in the histogram
    peaks, _ = find_peaks(hist)

    # Select top peaks based on their height
    top_peaks = peaks[np.argsort(hist[peaks])[-num_peaks:]][::-1]
    sorted_top_peaks = np.sort(top_peaks)

    # If less than num_peaks found, fill with -1 or some default value
    if len(sorted_top_peaks) < num_peaks:
        sorted_top_peaks = np.pad(sorted_top_peaks, (0, num_peaks - len(sorted_top_peaks)), 'constant', constant_values=-1)

    return sorted_top_peaks


def color_histogram_features(image):
    """
    Compute the mean and variance of color histograms for the Red, Green, and Blue channels.

    Parameters:
    image (3D array): Color (BGR) image.

    Returns:
    tuple: Mean and variance for Red, Green, and Blue channel histograms.
    """
    # Convert image to RGB from BGR
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize lists to store means and variances
    means = []
    variances = []

    # Iterate through each channel (Red, Green, Blue)
    for i in range(3):
        channel = image_rgb[:, :, i]

        # Calculate histogram for the current channel
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])

        # Normalize the histogram
        hist = hist / hist.sum()

        # Calculate and store the mean and variance
        mean_val = np.mean(hist)
        var_val = np.var(hist)

        means.append(mean_val)
        variances.append(var_val)

    return (means[0], variances[0], means[1], variances[1], means[2], variances[2])


def grayscale_histogram_features(image):
    """
    Compute the mean and variance of the grayscale histogram for an image.

    Parameters:
    image_path (str): Path to the image.

    Returns:
    tuple: Mean and variance of the grayscale histogram.
    """

    # Check if the image is loaded correctly
    if image is None:
        print(f"Error: Image could not be loaded.")
        return None

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate histogram for the grayscale image
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    # Normalize the histogram
    hist = hist / hist.sum()

    # Calculate the mean and variance of the histogram
    mean_val = np.mean(hist)
    var_val = np.var(hist)

    return mean_val, var_val


def hog_variance(image):
    """
    Compute the variance of the Histogram of Oriented Gradients (HOG) for an image.

    Parameters:
    image (2D array): Grayscale image.

    Returns:
    float: Variance of the HOG descriptor.
    """
    # Compute HOG for the image
    hog_features, hog_image = feature.hog(image, visualize=True)

    # Calculate the variance of HOG features
    variance = np.var(hog_features)
    mean = np.mean(hog_features)
    return mean, variance


def hog_histogram_peak_count(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    """
    Compute the number of significant peaks in the HOG histogram of an image.

    Parameters:
    image (2D array): Grayscale image.
    orientations (int): Number of orientation bins.
    pixels_per_cell (tuple): Size (in pixels) of a cell.
    cells_per_block (tuple): Number of cells in each block.

    Returns:
    int: Number of significant peaks in the HOG histogram.
    """
    # Compute HOG for the image
    hog_features = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, feature_vector=True)

    # Calculate histogram of HOG features
    hist, bins = np.histogram(hog_features, bins=orientations)

    # Find peaks in the histogram
    peaks, _ = find_peaks(hist)

    # Count the number of significant peaks
    peak_count = len(peaks)

    return peak_count


def edge_density_sobel(image):
    """
    Compute the edge density using the Sobel filter.

    Parameters:
    image (2D array): Grayscale image.

    Returns:
    float: Edge density of the image.
    """
    # Apply Sobel filter to find edges
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate the magnitude of gradients
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)

    # Calculate edge density
    edge_density = np.mean(gradient_magnitude)
    return edge_density


def edge_density_canny(image, low_threshold=100, high_threshold=200):
    """
    Compute the edge density using the Canny filter.

    Parameters:
    image (2D array): Grayscale image.
    low_threshold (int): Lower threshold for the hysteresis procedure.
    high_threshold (int): Higher threshold for the hysteresis procedure.

    Returns:
    float: Edge density of the image.
    """
    # Apply Canny edge detector
    edges = cv2.Canny(image, low_threshold, high_threshold)

    # Calculate edge density
    edge_density = np.sum(edges > 0) / edges.size
    return edge_density


def gabor_filter_response_variance(image, frequencies=[0.4, 0.5, 0.6], thetas=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """
    Compute the variance of the Gabor filter responses for multiple orientations.

    Parameters:
    image (2D array): Grayscale image.
    frequencies (list): Frequencies of the sinusoidal wave.
    thetas (list): Angles of the sinusoidal wave.

    Returns:
    list: Variances of the Gabor filter responses.
    """
    variances = []
    sigma = 5.0  # Standard deviation of the Gaussian envelope

    for frequency in frequencies:
        for theta in thetas:
            lambda_ = 1 / frequency  # Wavelength of the sinusoidal wave

            # Create Gabor filter
            gabor_kernel = cv2.getGaborKernel((21, 21), sigma, theta, lambda_, 0.5, 0, ktype=cv2.CV_64F)

            # Apply filter to image
            filtered_image = convolve(image, gabor_kernel.real)

            # Calculate and append variance of the filter response
            variances.append(np.var(filtered_image))

    return variances


def gabor_filter_mean_response(image, frequencies=[0.4, 0.5, 0.6], thetas=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """
    Compute the mean of the Gabor filter responses for multiple orientations.

    Parameters:
    image (2D array): Grayscale image.
    frequencies (list): Frequencies of the sinusoidal wave.
    thetas (list): Angles of the sinusoidal wave.

    Returns:
    list: Means of the Gabor filter responses.
    """
    means = []
    sigma = 5.0  # Standard deviation of the Gaussian envelope

    for frequency in frequencies:
        for theta in thetas:
            lambda_ = 1 / frequency  # Wavelength of the sinusoidal wave

            # Create Gabor filter
            gabor_kernel = cv2.getGaborKernel((21, 21), sigma, theta, lambda_, 0.5, 0, ktype=cv2.CV_64F)

            # Apply filter to image
            filtered_image = convolve(image, gabor_kernel.real)

            # Calculate and append mean of the filter response
            means.append(np.mean(filtered_image))

    return means


def detect_specular_highlights(image, eye_cascade_path):
    """
    Detect specular highlights in the eyes.

    Parameters:
    image (3D array): Color (BGR) image.
    eye_cascade_path (str): Path to the Haar cascade file for eye detection.

    Returns:
    bool: True if specular highlights are detected in the eyes, False otherwise.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load Haar cascade for eye detection
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    # Detect eyes
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

    # Check for specular highlights in each detected eye
    for (ex, ey, ew, eh) in eyes:
        eye_region = gray[ey:ey+eh, ex:ex+ew]
        # Apply a threshold to find bright regions
        _, thresholded_eye = cv2.threshold(eye_region, 220, 255, cv2.THRESH_BINARY)

    return np.sum(thresholded_eye)


def iris_texture_uniformity(iris_region, P=8, R=1):
    """
    Compute the texture uniformity of the iris region.

    Parameters:
    iris_region (2D array): Grayscale image of the iris.
    P (int): Number of circularly symmetric neighbor set points in LBP.
    R (float): Radius of circle for LBP.

    Returns:
    float: Texture uniformity of the iris.
    """
    # Compute Local Binary Pattern (LBP) for the iris region
    lbp = local_binary_pattern(iris_region, P, R, method="uniform")

    # Calculate the histogram of the LBP
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))

    # Normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)

    # Calculate uniformity
    uniformity = np.sum(hist**2)
    return uniformity


def face_contour_sharpness(image):
    """
    Compute the sharpness of face contours.

    Parameters:
    image (2D array): Grayscale image of the face.

    Returns:
    float: Sharpness of the face contours.
    """
    # Apply Sobel filter for edge detection
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobelx**2 + sobely**2)

    # Compute the sharpness as the mean of edge magnitudes
    sharpness = np.mean(edge_magnitude)
    return sharpness


def shadow_distribution_variance(image, threshold=100):
    """
    Compute the variance of shadow distribution in an image.

    Parameters:
    image (2D array): Grayscale image.
    threshold (int): Threshold value to identify shadow regions.

    Returns:
    float: Variance of shadow distribution.
    """
    # Identify shadow regions (darker regions)
    shadow_regions = image < threshold

    # Calculate the variance of intensities in shadow regions
    shadow_variance = np.var(image[shadow_regions])
    shadow_mean = np.mean(image[shadow_regions])
    return shadow_mean, shadow_variance


def reflection_symmetry_score(image):
    """
    Compute the reflection symmetry score of an image.

    Parameters:
    image (2D array): Grayscale image of a face.

    Returns:
    float: Reflection symmetry score.
    """
    # Get the center column index
    center_col = image.shape[1] // 2

    # Split the image into left and right halves
    left_half = image[:, :center_col]
    right_half = image[:, center_col:]

    # Flip the right half for comparison
    flipped_right_half = np.fliplr(right_half)

    # Calculate the symmetry score
    # Using mean squared difference here; other metrics can also be used
    symmetry_score = np.mean((left_half - flipped_right_half) ** 2)

    return symmetry_score


def wavelet_transform_low_freq_features(image):
    """
    Compute the mean and variance of the low-frequency components of the wavelet transform.

    Parameters:
    image (2D array): Grayscale image.

    Returns:
    tuple: Mean and variance of the low-frequency wavelet components.
    """
    # Apply wavelet decomposition
    coeffs = pywt.dwt2(image, 'haar')
    cA, (cH, cV, cD) = coeffs

    # Calculate mean and variance of the approximation coefficients (low frequency)
    mean_low_freq = np.mean(cA)
    variance_low_freq = np.var(cA)

    return mean_low_freq, variance_low_freq


def wavelet_transform_high_freq_features(image):
    """
    Compute the mean and variance of the high-frequency components of the wavelet transform.

    Parameters:
    image (2D array): Grayscale image.

    Returns:
    tuple: Mean and variance of the high-frequency wavelet components.
    """
    # Apply wavelet decomposition
    coeffs = pywt.dwt2(image, 'haar')
    cA, (cH, cV, cD) = coeffs

    # Combine high-frequency components (details)
    high_freq_components = np.hstack((np.hstack((cH, cV)), cD))

    # Calculate mean and variance of the high-frequency components
    mean_high_freq = np.mean(high_freq_components)
    variance_high_freq = np.var(high_freq_components)

    return mean_high_freq, variance_high_freq


image = cv2.imread('/home/kasra/spoof_data/pre_processed_cropped_faces/train/Iman_HajMostafaee/spoof/ID_card_1.png', cv2.IMREAD_GRAYSCALE)
uniformity_score = lbp_uniformity(image)
print("LBP Uniformity Score:", uniformity_score)

descriptor = lbp_histogram_features(image)
print("Combined LBP Descriptor:", descriptor)

top_peaks = lbp_histogram_top_peaks(image)
print("Top LBP Histogram Peaks:", top_peaks)

image_BGR = cv2.imread('/home/kasra/spoof_data/pre_processed_cropped_faces/train/Iman_HajMostafaee/spoof/ID_card_1.png')
red_mean, red_variance, green_mean, green_variance, blue_mean, blue_variance = color_histogram_features(image_BGR)
print("Red Mean:", red_mean, "Red Variance:", red_variance)
print("Green Mean:", green_mean, "Green Variance:", green_variance)
print("Blue Mean:", blue_mean, "Blue Variance:", blue_variance)

mean, variance = grayscale_histogram_features(image_BGR)
print("Grayscale Histogram Mean:", mean, "Grayscale Histogram Variance:", variance)

hog_mean, hog_var = hog_variance(image)
print("HOG Mean:", hog_mean, "HOG Variance:", hog_var)

peak_count = hog_histogram_peak_count(image)
print("Number of Significant HOG Histogram Peaks:", peak_count)

edge_density_value = edge_density_sobel(image)
print("Edge Density (Sobel Filter):", edge_density_value)

edge_density_value = edge_density_canny(image)
print("Edge Density (Canny Filter):", edge_density_value)

variances = gabor_filter_response_variance(image)
print("Gabor Filter Response Variances:", variances)

means = gabor_filter_mean_response(image)
print("Gabor Filter Mean Responses:", means)

eye_cascade_path = '/home/kasra/PycharmProjects/spoof_detection/haarcascade_eye.xml'
has_highlights = detect_specular_highlights(image_BGR, eye_cascade_path)
print("Specular Highlights Detected:", has_highlights)

contour_sharpness = face_contour_sharpness(image)
print("Face Contour Sharpness:", contour_sharpness)

shadow_mean, shadow_var = shadow_distribution_variance(image)
print("Shadow Distribution Mean:", shadow_mean, "Shadow Distribution Variance:", shadow_var)

score = reflection_symmetry_score(image)
print("Reflection Symmetry Score:", score)

mean_low_freq, variance_low_freq = wavelet_transform_low_freq_features(image)
print("Wavelet Transform Mean (Low Frequency):", mean_low_freq)
print("Wavelet Transform Variance (Low Frequency):", variance_low_freq)
mean_high_freq, variance_high_freq = wavelet_transform_high_freq_features(image)
print("Wavelet Transform Mean (High Frequency):", mean_high_freq)
print("Wavelet Transform Variance (High Frequency):", variance_high_freq)

