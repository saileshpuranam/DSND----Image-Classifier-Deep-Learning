from PIL import Image
from data_utils import get_test_transforms


def process_image(image_path):
    """ Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    """
    with Image.open(image_path) as image:
        transform = get_test_transforms()
        image = transform(image).numpy()

    return image