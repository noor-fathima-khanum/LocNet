from retrieval_scores import *


model_path='/home/noor/wacv19/LocNet/retrieval_scores/wacv2019_models/flickr_100epochs.tar'

flickr_processor = FlickrViz(batch_size=1, parse_mode='phrase', model_path=model_path, eval_mode=True, mode='test')
length_dataset = len(flickr_processor.dataset)

print ("Length of dataset",length_dataset)

num_images=3
flickr_processor.loc_eval(5000,num_images)
