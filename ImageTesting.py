from PIL import Image
import numpy as np

# image = Image.open('test.jpg')
# image_array = np.array(image)

# import Ax_b
#
# import MatrixTransformations
#
# m=image_array.shape[0]
# b=np.ones((m,1))
# print(b)
#
# x,conv = Ax_b.SolveLarge_Ax_b(MatrixTransformations.toSquareMatrix(image_array),b)
#
# print(x)

def squareImage(stringOfImage):
    from PIL import Image
    image = Image.open(stringOfImage)
    image_array = np.array(image)
    square_image=image.resize((image_array.shape[0], image_array.shape[0]))
    square_image.save('square_image.jpg')

    # return square_image matrix

    return np.array(square_image)

squareImage("test.jpg")

