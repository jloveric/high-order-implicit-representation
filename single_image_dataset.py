#from PIL import Image
from matplotlib import image

def image_to_dataset(filename: str) :
    # Open the image form working directory
    img = image.imread(filename)
    # summarize some details about the image
    print('image', img.shape)
    #print(img.format)
    #print(img.size)
    #print(img.mode)
    # show the image
    return img

def other():
    pyplot.imshow(image)
    pyplot.show()

if __name__ == "__main__":
    image_to_dataset("images/newt.jpg")