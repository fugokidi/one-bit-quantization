"""
The code is extracted from https://github.com/trimailov/qwer
I modified a bit to host the different bitdepths


Note: The alogorithm traverse the value from x = 1 and y = 1,
I want to have one bit image at the end, so I traverse all pixels.
"""

from math import floor

import torch
import torchvision.transforms as transforms

pil2tensor = transforms.ToTensor()
tensor2pil = transforms.ToPILImage()


def quantize(X, bitdepth):
    return (X * 255 / 2**(8 - bitdepth)).floor() / (2**bitdepth - 1)

def dequantize(X, bitdepth):
    return (X * (2**bitdepth - 1) * 2**(8 - bitdepth)).floor() / 255


def apply_threshold(value, bitdepth):
    return floor(floor(value / 2**(8 - bitdepth)) / (2**bitdepth - 1) * 255)


def floyd_steinberg_dither(image, bitdepth):
    """
    https://en.wikipedia.org/wiki/Floyd–Steinberg_dithering
    Pseudocode:
    for each y from top to bottom
       for each x from left to right
          oldpixel  := pixel[x][y]
          newpixel  := find_closest_palette_color(oldpixel)
          pixel[x][y]  := newpixel
          quant_error  := oldpixel - newpixel
          pixel[x+1][y  ] := pixel[x+1][y  ] + quant_error * 7/16
          pixel[x-1][y+1] := pixel[x-1][y+1] + quant_error * 3/16
          pixel[x  ][y+1] := pixel[x  ][y+1] + quant_error * 5/16
          pixel[x+1][y+1] := pixel[x+1][y+1] + quant_error * 1/16
    find_closest_palette_color(oldpixel) = floor(oldpixel / 256)
    """
    new_image = image.copy()
    pixel = new_image.load()

    x_lim, y_lim = image.size

    for y in range(y_lim):
        for x in range(x_lim):
            red_oldpixel, green_oldpixel, blue_oldpixel = pixel[x, y]

            red_newpixel = apply_threshold(red_oldpixel, bitdepth)
            green_newpixel = apply_threshold(green_oldpixel, bitdepth)
            blue_newpixel = apply_threshold(blue_oldpixel, bitdepth)

            # print(red_oldpixel, green_oldpixel, blue_oldpixel)
            pixel[x, y] = red_newpixel, green_newpixel, blue_newpixel

            red_error = red_oldpixel - red_newpixel
            blue_error = blue_oldpixel - blue_newpixel
            green_error = green_oldpixel - green_newpixel
            # print(red_error, green_error, blue_error)

            if x < x_lim - 1:
                red = pixel[x+1, y][0] + round(red_error * 7/16)
                green = pixel[x+1, y][1] + round(green_error * 7/16)
                blue = pixel[x+1, y][2] + round(blue_error * 7/16)

                pixel[x+1, y] = (red, green, blue)

            if x > 1 and y < y_lim - 1:
                red = pixel[x-1, y+1][0] + round(red_error * 3/16)
                green = pixel[x-1, y+1][1] + round(green_error * 3/16)
                blue = pixel[x-1, y+1][2] + round(blue_error * 3/16)

                pixel[x-1, y+1] = (red, green, blue)

            if y < y_lim - 1:
                red = pixel[x, y+1][0] + round(red_error * 5/16)
                green = pixel[x, y+1][1] + round(green_error * 5/16)
                blue = pixel[x, y+1][2] + round(blue_error * 5/16)

                pixel[x, y+1] = (red, green, blue)

            if x < x_lim - 1 and y < y_lim - 1:
                red = pixel[x+1, y+1][0] + round(red_error * 1/16)
                green = pixel[x+1, y+1][1] + round(green_error * 1/16)
                blue = pixel[x+1, y+1][2] + round(blue_error * 1/16)

                pixel[x+1, y+1] = (red, green, blue)

    return new_image


def batch_dither(data_batch, bitdepth):
    dither_tensors = []
    for data in data_batch:
        pilimg = tensor2pil(data.cpu())
        dithered = floyd_steinberg_dither(pilimg, bitdepth)
        tensor = pil2tensor(dithered).unsqueeze(0)
        dither_tensors.append(tensor)
    return torch.cat(dither_tensors, dim=0)
