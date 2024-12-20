import numpy as np
from scipy.linalg import sqrtm


def calculate_fid(real_features, generated_features):
    """
    Calculate the Fréchet Inception Distance (FID) between real and generated features.
    :param real_features: Features extracted from real images.
    :param generated_features: Features extracted from generated images.
    :return: FID score.
    """
    # Calculate the mean and covariance of both datasets
    mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(
        real_features, rowvar=False
    )
    mu_generated, sigma_generated = np.mean(generated_features, axis=0), np.cov(
        generated_features, rowvar=False
    )

    # Compute the squared difference of means
    mean_diff = np.sum((mu_real - mu_generated) ** 2)

    # Compute the product of covariances
    covmean, _ = sqrtm(sigma_real @ sigma_generated, disp=False)

    # Handle numerical instability
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Compute the FID score
    fid = mean_diff + np.trace(sigma_real + sigma_generated - 2 * covmean)

    return fid
