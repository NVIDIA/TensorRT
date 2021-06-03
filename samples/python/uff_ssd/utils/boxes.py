#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Utility functions for drawing bounding boxes on PIL images
import numpy as np
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont


def draw_bounding_boxes_on_image(image,
                                 boxes,
                                 color=(255, 0, 0),
                                 thickness=4,
                                 display_str_list=()):
    """Draws bounding boxes on image.

    Args:
        image (PIL.Image): PIL.Image object
        boxes (np.array): a 2 dimensional numpy array
            of [N, 4]: (ymin, xmin, ymax, xmax)
            The coordinates are in normalized format between [0, 1]
        color (int, int, int): RGB tuple describing color to draw bounding box
        thickness (int): bounding box line thickness
        display_str_list [str]: list of strings.
            Contains one string for each bounding box.
    Raises:
        ValueError: if boxes is not a [N, 4] array
    """
    boxes_shape = boxes.shape
    if not boxes_shape:
        return
    if len(boxes_shape) != 2 or boxes_shape[1] != 4:
        raise ValueError('boxes must be of size [N, 4]')
    for i in range(boxes_shape[0]):
        draw_bounding_box_on_image(image, boxes[i, 0], boxes[i, 1], boxes[i, 2],
            boxes[i, 3], color, thickness, display_str_list[i])

def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color=(255, 0, 0),
                               thickness=4,
                               display_str='',
                               use_normalized_coordinates=True):
    """Adds a bounding box to an image.

    Bounding box coordinates can be specified in either absolute (pixel) or
    normalized coordinates by setting the use_normalized_coordinates argument.

    The string passed in display_str is displayed above the
    bounding box in black text on a rectangle filled with the input 'color'.
    If the top of the bounding box extends to the edge of the image, the string
    is displayed below the bounding box.

    Args:
        image (PIL.Image): PIL.Image object
        ymin (float): ymin of bounding box
        xmin (float): xmin of bounding box
        ymax (float): ymax of bounding box
        xmax (float): xmax of bounding box
        color (int, int, int): RGB tuple describing color to draw bounding box
        thickness (int): line thickness
        display_str (str): string to display in box
        use_normalized_coordinates (bool): If True, treat coordinates
            ymin, xmin, ymax, xmax as relative to the image. Otherwise treat
            coordinates as absolute
    """
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom),
               (right, top), (left, top)], width=thickness, fill=tuple(color))
    try:
        font = ImageFont.truetype('arial.ttf', 24)
    except IOError:
        font = ImageFont.load_default()
    
    # If the total height of the display string added to the top of the bounding
    # box exceeds the top of the image, move the string below the bounding box
    # instead of above
    display_str_height = font.getsize(display_str)[1]
    # Each display_str has a top and bottom margin of 0.05x
    total_display_str_height = (1 + 2 * 0.05) * display_str_height

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height
    
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle(
        [(left, text_bottom - text_height - 2 * margin), (left + text_width,
            text_bottom)],
        fill=tuple(color))
    draw.text(
        (left + margin, text_bottom - text_height - margin),
        display_str,
        fill='black',
        font=font)
    text_bottom -= text_height - 2 * margin
