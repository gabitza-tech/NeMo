import os
import json
from nemo.core.config import hydra_runner
from omegaconf import OmegaConf
from scripts.utils import majority_element, load_pickle

def common_member(a, b):
    a_set = set(a)
    b_set = set(b)
    if len(a_set.intersection(b_set)) > 0:
        inters = list(a_set.intersection(b_set))
        return inters 
    return [] 

@hydra_runner(config_path="../conf", config_name="speaker_identification_fewshot")
def main(cfg):

    enroll_dict = load_pickle(cfg.data.enrollment_embs)
    test_dict = load_pickle(cfg.data.test_embs)

    inters_list = common_member(enroll_dict['concat_slices'], test_dict['concat_slices'])
    if len(inters_list)>0:
        
        inters_ids = {}
        for id1 in inters_list:
            if id1 not in inters_ids.keys():
                inters_ids[id1] = {}
                inters_ids[id1]["enroll"] = [enroll_dict['concat_labels'][index] for index,id in enumerate(enroll_dict['concat_slices']) if id1 == id]
                inters_ids[id1]["test"] = [test_dict['concat_labels'][index] for index,id in enumerate(test_dict['concat_slices']) if id1 == id]

                id_inters = common_member(inters_ids[id1]['enroll'],inters_ids[id1]['test'])

            if len(id_inters) > 0:
                print(id_inters)
                print("There is an error in the data preprocessing THE QUERY and THE SUPPORT have COMMON ELEMENTS!")
                exit(0)

    print("Everything looks good")

if __name__ == '__main__':
    main()