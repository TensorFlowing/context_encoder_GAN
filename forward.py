# Basile Van Hoorick, Jan 2020
'''
Hallucinates beyond all four edges of an image, increasing both dimensions by 50%.
The outpainting process interally converts 128x128 to 192x192, after which the generated output is upscaled.
Then, the original input is blended onto the result for optimal fidelity.
Example usage:
python forward.py input.jpg output.jpg
'''

if __name__ == '__main__':

    import sys
    from outpainting import *

    print("PyTorch version: ", torch.__version__)
    print("Torchvision version: ", torchvision.__version__)

    # src_file = sys.argv[1]
    # dst_file = sys.argv[2]
    # src_file = "/mnt/Data/yangbo/data/Places365_mini/val_large/Places365_val_00001099.jpg"
    src_file = "/home/us000123/image-outpainting/ceiling.png"
    dst_file = "/home/us000123/image-outpainting/inpainted5.jpg"

    gen_model = load_model('/home/us000123/outpaint_models/G_3.pt')
    print('Source file: ' + src_file + '...')
    input_img = plt.imread(src_file)[:, :, :3]
    # output_img, blended_img = perform_outpaint(gen_model, input_img)

    # input_img, masked_img, output_img = perform_inpaint(gen_model, input_img)

    input_size = 128
    mask_size = 64
    i = (input_size - mask_size) // 2
    # input_img = input_img[0:800, 300:1100, :]
    input_img = skimage.transform.resize(input_img, (input_size, input_size), anti_aliasing=True)
    masked_img = np.copy(input_img)
    masked_img[i:i+mask_size, i:i+mask_size, :] = 1
    output_img = inference_inpaint(gen_model, masked_img)
    plt.imsave(dst_file, np.hstack((input_img, masked_img, output_img)))
    print('Destination file: ' + dst_file + ' written')

    # plt.subplot(311)
    # plt.imshow(input_img)
    # plt.title("input_img")
    # plt.subplot(312)
    # plt.imshow(output_img)
    # plt.title("output_img")
    # plt.savefig("input_img, masked_img, output_img")