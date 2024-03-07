from fastNLP.io import Loader, DataBundle
from fastNLP.io.pipe.utils import iob2bioes
from fastNLP import DataSet, Instance
from transformers import AutoTokenizer
import numpy as np
import os
import torch
import torchvision
from itertools import chain
from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


from fastNLP.io import ConllLoader, Loader
from fastNLP.io.loader.conll import _read_conll
from fastNLP.io.pipe.utils import iob2, iob2bioes
from fastNLP import DataSet, Instance
from fastNLP.io import Pipe
from transformers import AutoTokenizer



# Define the function to convert BIO tags to spans
def _bio_tag_to_spans(tags, ignore_labels=None):
    """
    Given a list of tags, this function extracts the spans (start, end, label).
    :param tags: List of tags, e.g., ['B-PER', 'I-PER', 'O', 'B-LOC']
    :param ignore_labels: Labels to ignore, e.g., ['O']
    :return: A list of spans, e.g., [('PER', (0, 1)), ('LOC', (3, 3))]
    """
    spans = []
    prev_tag = 'O'
    prev_type = ''
    start = -1
    for i, tag in enumerate(tags):
        if tag == 'O':
            if prev_tag != 'O':
                spans.append((prev_type, (start, i - 1)))
            prev_tag = tag
            continue
        tag_type = tag.split('-')[-1]
        if tag.startswith('B-') or prev_type != tag_type:
            if prev_tag != 'O':
                spans.append((prev_type, (start, i - 1)))
            start = i
        prev_tag = tag
        prev_type = tag_type
    if prev_tag != 'O':
        spans.append((prev_type, (start, len(tags) - 1)))
    if ignore_labels is not None:
        spans = [span for span in spans if span[0] not in ignore_labels]
    return spans

class BartNERPipe(Pipe):
    def __init__(self, image_feature_path=None, image_annotation_path=None, max_bbox=16, normalize=False, tokenizer='facebook/bart-base', target_type='word'):
        super().__init__()
        self.image_feature_path = image_feature_path
        self.image_annotation_path = image_annotation_path
        self.max_bbox = max_bbox
        self.normalize = normalize
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        assert target_type in ('word')
        self.target_type = target_type
        self.not_cover = 0
        self.mapping2id = {}  # Initialize the attribute here
        self.mapping = {}  # Initialize the attribute here

    def add_tags_to_special_tokens(self, data_bundle):
    # def add_tags_to_special_tokens(self):
        mapping = {
            '0': '<<which region>>',
            '1': '<<no region>>',
            'loc': '<<location>>',
            'per': '<<person>>',
            'other': '<<others>>',
            'org': '<<organization>>'
        }
        self.mapping = mapping
        sorted_add_tokens = list(mapping.values())
        self.tokenizer.add_special_tokens({'additional_special_tokens': sorted_add_tokens})
        for key, value in mapping.items():
            key_id = self.tokenizer.convert_tokens_to_ids([value])[0]
            self.mapping2id[value] = key_id
            self.mapping2targetid[key] = len(self.mapping2targetid)

    def read_image_label(self,img_id):
        import xml.etree.ElementTree as ET
        fn=os.path.join(self.image_annotation_path,img_id+'.xml')
        tree=ET.parse(fn)
        root=tree.getroot()
        aspects = []
        boxes = []
        for object_container in root.findall('object'):
            for names in object_container.findall('name'):
                box_name=names.text
                box_container = object_container.findall('bndbox')
                if len(box_container) > 0:
                    xmin = int(box_container[0].findall('xmin')[0].text) 
                    ymin = int(box_container[0].findall('ymin')[0].text) 
                    xmax = int(box_container[0].findall('xmax')[0].text) 
                    ymax = int(box_container[0].findall('ymax')[0].text) 
                aspects.append(box_name)
                boxes.append([xmin,ymin,xmax,ymax])
        return aspects, boxes

    def visualize_detections(self, image_id, image_path, detection_results):
        # Load the image using PIL (Python Imaging Library)
        image = Image.open(image_path)

        # Create a figure and axis to plot the image
        fig, ax = plt.subplots(1)
        ax.imshow(image)

        # Plot bounding boxes and class labels
        for result in detection_results:
            x_min, y_min, x_max, y_max, class_label = result
            width, height = x_max - x_min, y_max - y_min
            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x_min, y_min - 5, class_label, color='red', fontsize=12)

        # Set the title of the plot to the image ID
        ax.set_title(f'Image ID: {image_id}')

        # Show the plot
        plt.show()

    def process(self, data_bundle):
        
        # self.add_tags_to_special_tokens(data_bundle)  

        # 转换tag
        target_shift = len(self.mapping) + 2  

        def prepare_target(ins):
            img_id = ins['img_id']
            
            image_num = 0
            image_tag =''
            image_boxes = np.zeros((self.max_bbox,4),dtype= np.float32)
            image_feature = np.zeros((self.max_bbox, self.region_dim),dtype= np.float32)
            
            if self.image_feature_path:
                ########### Vinvl image feature
                try:
                    img=np.load(os.path.join(self.image_feature_path,str(img_id)+'.jpg.npz'))
                    image_num = img['num_boxes']
                    image_feature_ = img['box_features']  
                    if self.normalize:
                        image_feature_ = (image_feature_/np.sqrt((image_feature_**2).sum()))  
                    final_num = min(image_num,self.max_bbox)
                    image_feature[:final_num] = image_feature_[:final_num]
                    image_boxes[:final_num] = img['bounding_boxes'][:final_num]
                except:
                    print("no image feature"+str(img_id)) 
            else:
                print("image feature error!")


            raw_words = ins['raw_words']
            word_bpes = [[self.tokenizer.bos_token_id]]
            first = []  # 用来取每个word第一个bpe
            cur_bpe_len = 1
            for word in raw_words:
                bpes = self.tokenizer.tokenize(word, add_prefix_space=True)
                bpes = self.tokenizer.convert_tokens_to_ids(bpes)
                first.append(cur_bpe_len)
                cur_bpe_len += len(bpes)
                word_bpes.append(bpes)
            assert first[-1] + len(bpes) == sum(map(len, word_bpes))
            word_bpes.append([self.tokenizer.eos_token_id])
            assert len(first) == len(raw_words) == len(word_bpes) - 2   ## raw_word

            lens = list(map(len, word_bpes))
            cum_lens = np.cumsum(lens).tolist()
            first = list(range(cum_lens[-1]))  ## first 只掩码sentence内部的

            

            ###### image_label #######
            aspect_ious_dic ={}   ## {aspect:[iou1,iou2,...]}
            if os.path.exists(os.path.join(self.image_annotation_path,img_id+'.xml')):
                names, gt_boxes = self.read_image_label(img_id)
                assert len(names) > 0, img_id
                IoUs=(torchvision.ops.box_iou(torch.tensor(gt_boxes),torch.tensor(image_boxes))).numpy() #[x,4],[16,4]  ->[x,16]
                
                for i,nn in enumerate(names):  ## 对于每个标注框
                    cur_iou = IoUs[i]
                    if max(cur_iou) < 0.5: ## object detector 没有检测到
                        self.not_cover +=1  ## ps: not_cover 这个数量是针对每个标注框的，不是每个实体的。
                        if nn not in aspect_ious_dic: ## 首次出现这个name，赋 -1；如果之前已经有这个name的记录，此处都不再关注
                            aspect_ious_dic[nn] = np.array([-1])  
                    else:
                        if nn in aspect_ious_dic:  ## 如果一个aspect对应多个标注框，更新iou
                            last_iou = aspect_ious_dic[nn]
                            if last_iou[0] == -1:  
                                aspect_ious_dic[nn] = cur_iou ## 直接赋当前iou
                            else: ## 该aspect有多个标注框，且多个框被检测到
                                final_iou = np.array([max(last_iou[i],cur_iou[i]) for i in range(len(last_iou))])
                                aspect_ious_dic[nn] = final_iou
                        else:
                            aspect_ious_dic[nn] = cur_iou
            
            
            region_label = []
            cover_flag = []  ## 0:entity-region 相关，但 detector 没有检测到 ；1: entity-region 相关，且检测到；2:entity-region 不相关
            entities = ins['entities']  # [[ent1, ent2,], [ent1, ent2]]
            for i,e in enumerate(entities):
                e = ' '.join(e)
                if e in aspect_ious_dic:
                    ori_ious = aspect_ious_dic[e]
                    ### 处理notcover
                    if ori_ious[0] == -1:
                        average_iou = 0.
                        region_label.append(np.array([average_iou]*self.max_bbox + [1.]))   ## 按照不相关训练
                        cover_flag.append(np.array([0])) ## 按照相关评估 ## 
                    else:
                        keeped_ious = np.array([iou if iou >0.5 else 0 for iou in ori_ious])
                        norm_iou = keeped_ious / float(sum(keeped_ious))
                        region_label.append(np.append(norm_iou,[0.]))
                        cover_flag.append(np.array([1]))
                    
                else:
                    average_iou = 0.  # 0. # 1 / self.max_bbox
                    region_label.append(np.array([average_iou]*self.max_bbox + [1.]))  
                    cover_flag.append(np.array([2]))
            
            ## 全 O 是 [], 会报错，先pad一个
            if len(region_label) ==0:
                region_label.append(np.array([0.]*(self.max_bbox +1)))
            if len(cover_flag) ==0:
                cover_flag.append(np.array([2]))
            
            
            entity_spans = ins['entity_spans']  # [(s1, e1, s2, e2), ()]
            entity_tags = ins['entity_tags']  # [tag1, tag2...]
            target = [0]  
            pairs = []
            
            assert len(entity_spans) == len(entity_tags)
            _word_bpes = list(chain(*word_bpes))
            for idx, (entity, tag) in enumerate(zip(entity_spans, entity_tags)):
                cur_pair = []
                num_ent = len(entity) // 2
                for i in range(num_ent):
                    start = entity[2 * i]
                    end = entity[2 * i + 1]
                    cur_pair_ = []
                    if self.target_type == 'word':
                        cur_pair_.extend([cum_lens[k] for k in list(range(start, end))])
                    elif self.target_type == 'span':
                        cur_pair_.append(cum_lens[start])
                        cur_pair_.append(cum_lens[end]-1)  # it is more reasonable to use ``cur_pair_.append(cum_lens[end-1])``
                    elif self.target_type == 'span_bpe':
                        cur_pair_.extend(
                            list(range(cum_lens[start], cum_lens[start + 1])))  # 由于cum_lens是[1, 3...]即第0位其实就是cls之后的了
                        cur_pair_.extend(
                            list(range(cum_lens[end - 1], cum_lens[end])))  # 由于cum_lens是[1, 3...]即第0位其实就是cls之后的了
                    elif self.target_type == 'bpe':
                        cur_pair_.extend(list(range(cum_lens[start], cum_lens[end])))
                    else:
                        raise RuntimeError("Not support other tagging")
                    cur_pair.extend([p + target_shift for p in cur_pair_])
                for _, (j, word_idx) in enumerate(zip((cur_pair[0], cur_pair[-1]), (0, -1))):
                    j = j - target_shift
                    if 'word' == self.target_type or word_idx != -1:
                        assert _word_bpes[j] == \
                               self.tokenizer.convert_tokens_to_ids(
                                   self.tokenizer.tokenize(entities[idx][word_idx], add_prefix_space=True)[:1])[0]
                    else:
                        assert _word_bpes[j] == \
                               self.tokenizer.convert_tokens_to_ids(
                                   self.tokenizer.tokenize(entities[idx][word_idx], add_prefix_space=True)[-1:])[0]
                assert all([cur_pair[i] < cum_lens[-1] + target_shift for i in range(len(cur_pair))])
                               
                cur_pair.append(self.mapping2targetid[str(int(region_label[idx][-1]))] +2)  ##  entity-region relation
                cur_pair.append(self.mapping2targetid[tag] + 2)  # 加2是由于有shift 
                ### ↑ [span, <<which region>>(+Linear), type] / [span, <<no region>>, type]                
                
                pairs.append([p for p in cur_pair])
                
            target.extend(list(chain(*pairs)))
            target.append(1)  # 特殊的eos

            word_bpes = list(chain(*word_bpes))
            
            
            dict  = {'tgt_tokens': target, 'target_span': pairs, 'src_tokens': word_bpes,
                    'first': first,'image_tag':image_tag, 'image_feature':image_feature,'region_label':region_label,'cover_flag':cover_flag}
            return dict

        # data_bundle.apply_more(prepare_target, use_tqdm=False, tqdm_desc='pre. tgt.')  
        data_bundle.apply_more(prepare_target)



        # data_bundle.set_ignore_type('target_span', 'entities') 
        data_bundle.set_ignore('target_span', 'entities')

        data_bundle.set_ignore('image_tag')
        data_bundle.set_pad('tgt_tokens', 1)  # Set padding value for 'tgt_tokens' field
        data_bundle.set_pad('src_tokens', self.tokenizer.pad_token_id)

        data_bundle.apply_field(lambda x: len(x), field_name='src_tokens', new_field_name='src_seq_len')
        data_bundle.apply_field(lambda x: len(x), field_name='tgt_tokens', new_field_name='tgt_seq_len')

        # Set input fields for each dataset
        for name, dataset in data_bundle.iter_datasets():
            dataset.set_input('tgt_tokens', 'src_tokens', 'src_seq_len', 'tgt_seq_len', 'first', 'image_feature', 'image_tag')

        # Set target fields for each dataset
        for name, dataset in data_bundle.iter_datasets():
            dataset.set_target('tgt_tokens', 'tgt_seq_len', 'target_span', 'entities', 'region_label', 'cover_flag')


        print("not_cover: %d"%(self.not_cover))



        return data_bundle

    def process_from_file(self, paths, demo=False) -> DataBundle:
        
        # 读取数据
        if isinstance(paths, str):
            path = paths
        else:
            path = paths['train']
        
        data_bundle = TwitterNer(demo=demo).load(paths)
        
        data_bundle = self.process(data_bundle)
        
        return data_bundle
    




class TwitterNer(Loader):
    def __init__(self, demo=False):
        super().__init__()
        self.demo = demo

    def _load(self, path):
        ds = DataSet()
        with open(path, 'r', encoding='utf-8') as f:
            raw_data = []
            target = []
            for line in f:
                if line.strip() == '':
                    if raw_data:
                        ds.append(Instance(raw_words=raw_data, target=target))
                        raw_data = []
                        target = []
                else:
                    parts = line.strip().split()
                    raw_data.append(parts[0])
                    target.append(parts[-1])
            if raw_data:
                ds.append(Instance(raw_words=raw_data, target=target))
        return ds

if __name__ == '__main__':
    # Load the data
    loader = TwitterNer()
    data_bundle = loader.load('Twitter10000')

    # Process the data
    pipe = BartNERPipe(target_type='word')
    data_bundle = pipe.process(data_bundle)


    image_id = '16_05_01_7_region'
    image_path = '16_05_01_7.jpg'
    detection_results = [
        [241, 422, 569, 587, 'Mike Garciaparra']
]
    pipe.visualize_detections(image_id, image_path, detection_results)








# if __name__ == '__main__':
#     data_bundle = TwitterNer(demo=False).load('data/twitter')
#     BartNERPipe(target_type='word', dataset_name='twitter').process(data_bundle)