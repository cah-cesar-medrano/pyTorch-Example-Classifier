import matplotlib.pyplot as plt
import numpy as np

# Lets show some of the training images for fun

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def peek(trainloader, torchvision, classes):
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show image
    imshow(torchvision.utils.make_grid(images))

    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
