import matplotlib.pyplot as plt


def show_image(image_set, idx):
    image=image_set[idx, :, :, :]
    plt.axis("off")
    plt.imshow(image)
    plt.show()

