import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

from steps.utils import matchmap_generate
from models import VGG19, LSTMBranch

import sys

path_to_dataloader = '../'
sys.path.append(path_to_dataloader)

from dataloader import *


def caption_list_modify(caption):
    """
    Due to the parsing done by pytorch's dataloader, captions generated there
    is a list of positional tuples. This function creates a list of tokenized captions.
    :param caption: list of positional tuples
    :return caption_list: list of tokenized captions
    """
    caption_list = []
    for i in range(len(caption[0])):
        new_caption = []
        for j in range(len(caption)):
            new_caption.append(caption[j][i])
        caption_list.append(new_caption)
    return caption_list


def genome_load_data(batch_size, transform, mode='train'):
    genome_loader = get_loader_genome(transform=transform,
                                      mode='train',
                                      batch_size=batch_size)

    for batch in genome_loader:
        (image_tensor, caption_glove, ann_id) = batch

    return image_tensor, caption_glove, ann_id


def coco_load_data(batch_size, transform, mode='val'):
    """
    Loads data from the coco dataset using fold mode
    :return  image_tensor, caption_glove tensor, parsed list of caption:
    """
    coco_loader = get_loader_coco(transform=transform,
                                  mode='val',
                                  batch_size=batch_size)
    for batch in coco_loader:
        image_tensor, caption_glove, captions = batch[0], batch[1], batch[2]
    caption_list = caption_list_modify(captions)

    return image_tensor, caption_glove, caption_list


def flickr_load_data(batch_size, parse_mode, transform, mode='test', eval_mode=True):
    """
    Loads data from test fold of flickr dataset using flickr_loader
    :param parse_mode: decides whether the captions should be parsed 
    :return: image tensor, caption glove tensor and tuple of caption ids
    """
    flickr_loader = get_loader_flickr(transform=transform,
                                      batch_size=batch_size,
                                      mode=mode, parse_mode=parse_mode)

    for batch in flickr_loader:
        image, caption_glove, caption, ids = batch[0], batch[1], batch[2], batch[3]

    if eval_mode:
        print ("Hello")
        return flickr_loader

    return image, caption_glove, ids


def get_models(model_path):
    '''
    Load pre-trained model
    :return: pretrained image and caption model
    '''
    image_model = VGG19(pretrained=True)
    caption_model = LSTMBranch()

    if not os.path.exists(model_path):
        print("Not using trained models")
        return image_model, caption_model

    checkpoint = torch.load(model_path, map_location='cpu')
    image_model.load_state_dict(checkpoint['image_model'])
    caption_model.load_state_dict(checkpoint['caption_model'])
    print('Loaded pretrained models')
    return image_model, caption_model


def gen_coloc_maps_image(image_model,
                   image_tensor):
    """
    Generate a list of numpy match-maps from a batch of data
    :param models and tensors:Caption glove and image tensors required 
    generate a list of co_localization maps.
    :return: A list of co-localization maps. Each map is a numpy ndarray
    """
    with torch.no_grad():
        image_op = image_model(image_tensor)
    #batch_size = image_tensor.size(0)
    #coloc_maps = list()
    #vgg_op = image_op.mean(1)       # Mean across depth dimension

    #for i in np.arange(batch_size):
    #    coloc = matchmap_generate(image_op[i], caption_op[i])
    #    mm = coloc.detach().numpy()
    #    coloc_maps.append(mm)

    #return coloc_maps, vgg_op
    return image_op

def gen_coloc_maps_caption(caption_model,caption_glove):
    """
    Generate a list of numpy match-maps from a batch of data
    :param models and tensors:Caption glove and image tensors required 
    generate a list of co_localization maps.
    :return: A list of co-localization maps. Each map is a numpy ndarray
    """
    with torch.no_grad():
        caption_op = caption_model(caption_glove)
    #batch_size = image_tensor.size(0)
    #coloc_maps = list()
    #vgg_op = image_op.mean(1)       # Mean across depth dimension

    #for i in np.arange(batch_size):
    #    coloc = matchmap_generate(image_op[i], caption_op[i])
    #    mm = coloc.detach().numpy()
    #    coloc_maps.append(mm)

    #return coloc_maps, vgg_op
    return caption_op


def fetch_data(index, coloc_maps, vgg_op, image_tensor, captions):
    """
    Parse all necessary data based on index into a dictionary
    :return element: Dictionary containing three items:
    1. image tensor at index-th index
    2. caption/Caption ID depending on dataset being used
    3. coloc_map: untrimmed co_loc map at index position.
    """
    element = dict()
    element['coloc_map'] = coloc_maps[index]
    element['image'] = image_tensor[index]
    element['caption'] = captions[index] # In case of flickr, AnnID
    element['name'] = str(index)
    element['vgg_op'] = vgg_op[index]

    return element


def rgb2gray(rgb):
    """
    Convert color numpy image to grayscale
    """
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def tensor2img(tensor_image):
    """
    Convert each tensor image to numpy image
    :return image: Dictionary of two elements
    :return image['col']: Numpy 3 channel color image
    :return image['bw']: Numpy single channel BW image
    """
    img = tensor_image.permute(1, 2, 0)
    color_img = img.numpy()
    bw_img = rgb2gray(color_img)
    image = {"color": color_img, "bw": bw_img}

    return image

def vgg_op_processor(vgg_op):
    """
    Convert each vgg_op to a mask
    """
    mask = vgg_op.detach().numpy()
    mask = cv2.resize(mask, dsize=(224,224))

    return mask


def coloc_map_processor(coloc_map):
    """
    Process a list of coloc maps
    :param coloc_map: List of numpy matrices. Clipped coloc map.
    :return mask_list: numpy stack of resized masks to be visualized
    """
    mask_list = list()
    for frame in coloc_map:
        mask = cv2.resize(frame, dsize=(224, 224))
        mask_list.append(mask)
    mask_list = np.stack(mask_list, axis=0)
    return mask_list


def caption_parser(tok_caption, coloc_maps, index_list):
    """
    When parse_mode for flickr is not phrase, tokenized caption is passed into data
    Use this tokenized data to generate a parsed caption and average of the 
    associated positional colocalization maps
    :param tok_caption: Tokenized caption without end and start words
    :param coloc_maps: clipped list of colocalization maps. 
    :param index_list: list of positional indices to convert tok_caption to tok_phrase
    :return parsed_caption: list of phrases. each phrase is a list
    :return parsed_coloc_maps: list of averaged colocalization maps. 
    """
    parsed_caption = list()
    parsed_coloc_maps = list()

    for position in index_list:
        new_word_token = list()
        new_coloc_map = list()
        for index in position:
            new_word_token.append(tok_caption[index])
            new_coloc_map.append(coloc_maps[index])
        new_coloc_map = np.array(new_coloc_map)
        new_map = new_coloc_map.mean(axis=0)
        parsed_caption.append(new_word_token)
        parsed_coloc_maps.append(new_map)

    return parsed_caption, parsed_coloc_maps


def flickr_element_processor(element, parse_mode, data):
    """
    Clips co_loc maps and captions for FLICKR data points
    :param parse_mode: Decides which caption to fetch using ann_id
    :param data: dictionary containing data of all captions
    :return: return dict element with clipped & resized co_loc masks
    clipped co_loc maps and captions have no '<start>' and '<end>'
    Processed image has been converted to np images ready to be plotted
    """
    caption_data = data[element['caption']]
    bboxes = caption_data['boxes']
    image_size = caption_data['image_size']
    index_list = caption_data['caption_index']
    coloc_map = list(element['coloc_map'])
    if parse_mode == 'phrase':
        caption = caption_data['parsed_caption']
    else:
        caption = caption_data['tok_sent']
    actual_length = len(caption)
    del coloc_map[(actual_length+1):]
    del coloc_map[0]

    if parse_mode != "phrase":
        caption, coloc_map = caption_parser(caption, coloc_map, index_list)

    coloc_map = coloc_map_processor(coloc_map)

    element['coloc_map'] = coloc_map
    element['caption'] = caption
    element['image'] = tensor2img(element['image'])
    element['boxes'] = bboxes
    element['image_size'] = image_size

    vgg_op = element['vgg_op']
    element['vgg_op'] = vgg_op_processor(vgg_op)
    
    return element



def coco_element_processor(element):
    """
    Clips co_loc maps and captions for COCO data points
    :return: return dict element with clipped & resized co_loc masks
    clipped co_loc maps and captions have no '<start>' and '<end>'
    Processed image has been converted to np images ready to be plotted
    """
    caption = element['caption']

    coloc_map = list(element['coloc_map'])
    coloc_map = coloc_map_processor(coloc_map)
    element['coloc_map'] = coloc_map

    vgg_op = element['vgg_op']
    element['vgg_op'] = vgg_op_processor(vgg_op)

    element['caption'] = caption

    element['image'] = tensor2img(element['image'])

    return element


def list2string(query_list):
    """
    Convert a list of strings to a space separated string
    :param query_list: a list of strings
    :return final_string: a string where each token is joined by " "
    """
    assert type(query_list) is list, "Lists accepted only"
    separator = ' '
    final_string = separator.join(query_list)
    return final_string


def phrase_detokenizer(caption):
    """
    Converts a list of tokenized phrases into a list of phrases
    :param caption: List of tokenized phrases. Each phrase is a list of tokens
    :return new_caption: List of strings. Each string is a phrase
    """
    new_caption = list()
    for phrase in caption:
        new_caption.append(list2string(phrase))
    return new_caption


def mask_viz(mask_list, caption, bw_img, boxes, save_flag=False, save_name=''):
    """
    Generate heatmap for each entity
    """
    fig = plt.figure(figsize=(100,30), facecolor="white")
    columns = len(mask_list) + 1
    rows = 1
    for id in range(len(mask_list)-1):
        cap_phrase = caption[id]
        mask = cv2.resize(mask_list[id], dsize=(224,224))
        ax = fig.add_subplot(rows, columns, id + 1)
        ax.imshow(bw_img)
        ax.imshow(mask, cmap='jet', alpha=0.5)
        if type(boxes[id]) is list:
            for box in boxes[id]:
                rect = patches.Rectangle((box[0], box[1]), (box[2]-box[0]), (box[3]-box[1]), linewidth=10, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
        plt.title(cap_phrase, fontdict={'fontsize': 70})
        plt.axis('off')
    plt.show()

    if save_flag:
        fig.savefig(save_name)


def mask_viz_coco(mask_list, caption, bw_img, save_flag=False, save_name=''):
    """
    Generate heatmap for each entity
    """
    fig = plt.figure(figsize=(100,30), facecolor="white")
    columns = len(mask_list) + 1
    rows = 1
    for id in range(len(mask_list)-1):
        cap_phrase = caption[id]
        mask = cv2.resize(mask_list[id], dsize=(224,224))
        if not cap_phrase in ['<start>','<end>',',','.','<unk>']:
            fig.add_subplot(rows, columns, id + 1)
            plt.imshow(bw_img)
            plt.imshow(mask, cmap='jet', alpha=0.5)
            plt.title(cap_phrase, fontdict={'fontsize': 25})
            plt.axis('off')
    plt.show()

    if save_flag:
        fig.savefig(save_name)


def seg_viz(mask_list, caption, color_img, boxes, thresh, save_flag=False, save_name=''):
    """
    Generate localization masks for each entity phrase
    """
    fig = plt.figure(figsize=(100,30), facecolor="white")
    columns = len(mask_list) + 1
    rows = 1

    for id in range(len(mask_list)-1):
        cap_phrase = caption[id]
        mask = cv2.resize(mask_list[id], dsize=(224, 224))
        mask2 = np.where((mask < thresh * np.mean(mask)), 0, 1).astype('uint8')
        ax = fig.add_subplot(rows, columns, id + 1)
        img = color_img * mask2[:, :, np.newaxis]

        ax.imshow(img)
        if type(boxes[id]) is list:
            for box in boxes[id]:
                rect = patches.Rectangle((box[0], box[1]), (box[2]-box[0]), (box[3]-box[1]), linewidth=10, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
        plt.title(cap_phrase, fontdict={'fontsize': 70})
        plt.axis('off')

    plt.show()

    if save_flag:
        fig.savefig(save_name)


def seg_viz_coco(mask_list, caption, color_img, thresh, save_flag=False, save_name=''):
    """
    Generate localization masks for each entity phrase
    """
    fig = plt.figure(figsize=(100,30), facecolor="white")
    columns = len(mask_list) + 1
    rows = 1

    for id in range(len(mask_list)-1):
        cap_phrase = caption[id]
        mask = cv2.resize(mask_list[id], dsize=(224, 224))
        mask2 = np.where((mask < thresh * np.mean(mask)), 0, 1).astype('uint8')
        # ax = 
        if not cap_phrase in ['<start>', '<end>',',','.','<unk>']:
            fig.add_subplot(rows, columns, id + 1)
            img = color_img * mask2[:, :, np.newaxis]
            plt.imshow(img)
            plt.title(cap_phrase, fontdict={'fontsize': 25})
            plt.axis('off')

    plt.show()

    if save_flag:
        fig.savefig(save_name)


def find_mask_max(mask_list):
    """
    Find the coordinates of the maximum point in each mask
    :param mask_list: list of co localization masks
    :return coords_max: list of tuples. Each tuple is max_coordinate
    """
    coords_max = list()
    for id in range(len(mask_list)):
        mask = cv2.resize(mask_list[id], dsize=(224, 224))
        ind = np.unravel_index(np.argmax(mask, axis=None), mask.shape)
        coords_max.append(ind)
    return coords_max


def flickr_box_converter(boxes, image_size):
    """
    Bounding box coordinates change because image is of other size now
    :param boxes: original bounding box list from element
    :param image_size: original image size from element
    :return new_boxes: transformed boxes with coordinates
    """
    width_multiplier = 224 / image_size['width']
    height_multiplier = 224 / image_size['height']

    new_boxes = list()
    for frame in boxes:
        if type(frame) is not list:
            new_boxes.append('<none>')
            continue
        else:
            new_frame = list()
            for box in frame:
                new_box = list()
                new_box.extend([(box[0] * width_multiplier),
                                (box[1] * height_multiplier),
                                (box[2] * width_multiplier),
                                (box[3] * height_multiplier)])
                new_frame.append(new_box)
            new_boxes.append(new_frame)

    return new_boxes


def hit_condition(tup,coord):
    """
    check condition whether tuple lies in coordinates
    """
    return int(coord[0] <= tup[0] <= coord[2]) and (coord[1] <= tup[1] <= coord[3])


def single_image_score(boxes, coordinates_max):
    """
    Find out localization score for single image
    :param boxes: ground truth bounding boxes
    :param coordinates_max: predicted location for each entity in caption
    :return score: localization score
    """
    hits = 0
    total = 0

    for frame_index in range(len(boxes)):
        if type(boxes[frame_index]) is not list:
            continue
        else:
            for box in boxes[frame_index]:
                total += 1
                hits += hit_condition(coordinates_max[frame_index], box)

    score = hits / total
    return score


def element_score(element):
    """
    Find out localization score for each element
    :param element: element is a dictionary
    :return score: score for that data element
    """
    boxes = element['boxes']
    image_size = element['image_size']

    coords_max = find_mask_max(element['coloc_map'])
    boxes = flickr_box_converter(boxes, image_size)

    return single_image_score(boxes, coords_max)