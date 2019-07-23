from .visualize_utils import *

from torchvision import transforms
import json
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean
from tqdm import tqdm
import collections
from collections import defaultdict

from .utils import *

# from steps.utils import *

transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])


class FlickrViz():

    def __init__(self, batch_size, parse_mode, model_path, 
                 mode='test', transform=transform, eval_mode=False):
        """
        If eval_mode is true, evaluate localization score for entities present in the dataset.
        Otherwise, load models, batch-size data, compute colocalization maps. 
        """
        self.eval_mode = eval_mode
        self.batch_size = batch_size
        self.parse_mode = parse_mode
        self.json_path = 'data/flickr_30kentities/annotations_flickr/Sentences/test/data.json'
        self.data = json.load(open(self.json_path, encoding='utf-8', mode='r'))
        self.model_path = model_path
        self.mode = mode
        self.transform = transform
        self.image_model, self.caption_model = get_models(self.model_path)
        
        if self.eval_mode:
            loader = flickr_load_data(1, self.parse_mode, self.transform,
                                      self.mode, self.eval_mode)
            self.dataset = loader.dataset

        else:
            self.image_tensor, self.caption_glove, self.ids = flickr_load_data(self.batch_size,
                                                                           self.parse_mode,
                                                                           self.transform)
            self.coloc_maps, self.vgg_op = gen_coloc_maps(self.image_model, self.caption_model,
                                    self.image_tensor, self.caption_glove)



    def loc_eval(self, last,num_images):
        """
        When in eval mode, this will load entire dataset and iteratively find
        localization score for each image first and then average it to find 
        localization score for the entire dataset. Only works when eval_mode is True.
        :param last: number of images to evaluate. For full dataset, use len(data_loader.dataset)
        :return score_list: last - length list of all scores
        :return mean(score_list): mean localization score for dataset. 
        """
        img_cap_dict = defaultdict(list)
        img_cap_len_dict={}        
        score_list = list()
        all_img_op_tensor=[]
        all_cap_op_tensor=[]
        include_in_list=True
        img_tensor_id=0
        allimages=[]
        allcapgloves=[]
        
        for index in range(0,last,5):
            startid=index
            endid=startid+5
            
            if include_in_list==True:
               for caption_id in range(startid,endid,1):
                  image_tensor, caption_glove, caption, cap_id = self.dataset[caption_id]                 
                  if (int(caption_id)%int(5)) == 0:
                     allimages.append(image_tensor)
                  
                  if len(caption)==22:
                     allcapgloves.append(caption_glove)
                     img_cap_dict[img_tensor_id].append(caption)
                  
               img_tensor_id=img_tensor_id+1
               
        allimages_tensor=torch.stack(allimages,dim=0)
        total_caption_gloves=torch.stack(allcapgloves,dim=0)
        
        print ("ALLIMAGESTENSOR SHAPE",allimages_tensor.size(),total_caption_gloves.size())
        
      
        cur_img_tensor_op=[]
        for i in range(0,allimages_tensor.shape[0],150):
           print ("**********",i)
           startid=i
           endid=startid+150
           cur_img_tensor=allimages_tensor[startid:endid,:,:,:]
           cur_img_tensor_op.append(gen_coloc_maps_image(self.image_model,cur_img_tensor))
        
        caption_gloves_output=gen_coloc_maps_caption(self.caption_model,total_caption_gloves)
        
        print ("Printing size of Output Tensors",caption_gloves_output.size())

        
        for i in range(len(img_cap_dict)):
            img_cap_len_dict[i] = len(img_cap_dict[i])
        
        img_cap_corr = dict()
        total_count = 0
        for key in img_cap_len_dict.keys():
            img_cap_corr[key] = list(range(img_cap_len_dict[key]))
            img_cap_corr[key] = [item + total_count for item in img_cap_corr[key]]
            total_count += img_cap_len_dict[key]
        cap_img_corr = dict( (v,k) for k in img_cap_corr for v in img_cap_corr[k] )
        img_cap_corr = list(img_cap_corr.values())


        cur_cap_img_sim_mat=[]
        score_type='Avg_Both'
        for k in range(len(cur_img_tensor_op)):
           cur_img_output=cur_img_tensor_op[k]
           cur_sim_val = compute_matchmap_similarity_matrix(cur_img_output,caption_gloves_output,score_type)                   
           print ("shape",cur_sim_val.shape)
           cur_cap_img_sim_mat.append(cur_sim_val)
        cur_cap_img_sim_mat_tensor=torch.cat(cur_cap_img_sim_mat,dim=0)
        print ("SIM MAT",cur_cap_img_sim_mat_tensor.shape)
        
        cur_recall_values=calc_recalls_uneven_testing(cur_cap_img_sim_mat_tensor,score_type,img_cap_corr,cap_img_corr)       
        print ("CUR RECALL VALUES")
        print (cur_recall_values)