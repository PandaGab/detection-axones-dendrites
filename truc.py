import matplotlib.pyplot as plt

def plot(data, dout):
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(data['actine'],cmap='gray')
    plt.title('Actine')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(data['mask'],cmap='gray')
    plt.title('Dendrite Mask')
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(dout['actine'],cmap='gray')
    plt.title('Dendrite Crop')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(dout['mask'],cmap='gray')
    plt.axis('off')