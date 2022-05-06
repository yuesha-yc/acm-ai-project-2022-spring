from PIL import Image

def resize_image(df, size):
    """
    Resizes all images in a dataframe to a specified size.
    """
    for i, id in enumerate(df['Image']):
        if (i % 1000 == 0):
            print(i, id)
        image = Image.open(data_path + id)
        image = image.convert('RGB')
        image = image.resize(size)
        