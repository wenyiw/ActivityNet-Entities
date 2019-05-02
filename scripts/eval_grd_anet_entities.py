# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Evaluation script for object grounding over generated sentences

import json
import argparse
import sys, os

os.system('pip install https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp27-cp27mu-linux_x86_64.whl')
os.system('pip install stanfordcorenlp')

import torch
import itertools
import numpy as np
from collections import defaultdict
from utils import bbox_overlaps_batch, get_frm_mask

from stanfordcorenlp import StanfordCoreNLP

import logging

class ANetGrdEval(object):

    def __init__(self, reference_file=None, submission_file=None,
                 split_file=None, val_split=None, iou_thresh=0.5, verbose=False):

        if not reference_file:
            raise IOError('Please input a valid reference file!')
        if not submission_file:
            raise IOError('Please input a valid submission file!')

        self.iou_thresh = iou_thresh
        self.verbose = verbose
        self.val_split = val_split

        self.import_ref(reference_file, split_file)
        self.import_sub(submission_file)


    def import_ref(self, reference_file=None, split_file=None):

        with open(split_file) as f:
            split_dict = json.load(f)
        split = {}
        for s in self.val_split:
            split.update({i:i for i in split_dict[s]})

        with open(reference_file) as f:
            ref = json.load(f)['annotations']
        ref = {k:v for k,v in ref.items() if k in split}
        self.ref = ref


    def import_sub(self, submission_file=None):
        with open(submission_file) as f:
            pred = json.load(f)['results']
        self.pred = pred


    def gt_grd_eval(self):

        ref = self.ref
        pred = self.pred

        results = defaultdict(list)
        for vid, anns in ref.items():
            for seg, ann in anns['segments'].items():
                if len(ann['frame_ind']) == 0:
                    continue # annotation not available

                ref_bbox_all = torch.cat((torch.Tensor(ann['process_bnd_box']), \
                    torch.Tensor(ann['frame_ind']).unsqueeze(-1)), dim=1) # 5-D coordinates
                sent_idx = set(itertools.chain.from_iterable(ann['process_idx'])) # index of word in sentence to evaluate
                for idx in sent_idx:
                    sel_idx = [ind for ind, i in enumerate(ann['process_idx']) if idx in i]
                    ref_bbox = ref_bbox_all[sel_idx] # select matched boxes
                    # Note that despite discouraged, a single word could be annotated across multiple boxes/frames
                    assert(ref_bbox.size(0) > 0)

                    class_name = ann['process_clss'][sel_idx[0]][ann['process_idx'][sel_idx[0]].index(idx)]
                    if vid not in pred:
                        results[class_name].append(0) # video not grounded
                    elif seg not in pred[vid]:
                        results[class_name].append(0) # segment not grounded
                    elif idx not in pred[vid][seg]['idx_in_sent']:
                        results[class_name].append(0) # object not grounded
                    else:
                        pred_ind = pred[vid][seg]['idx_in_sent'].index(idx)
                        pred_bbox = torch.cat((torch.Tensor(pred[vid][seg]['bbox_for_all_frames'][pred_ind])[:,:4], \
                            torch.Tensor(range(10)).unsqueeze(-1)), dim=1)

                        frm_mask = torch.from_numpy(get_frm_mask(pred_bbox[:, 4].numpy(), \
                            ref_bbox[:, 4].numpy()).astype('uint8'))
                        overlap = bbox_overlaps_batch(pred_bbox[:, :5].unsqueeze(0), \
                            ref_bbox[:, :5].unsqueeze(0), frm_mask.unsqueeze(0))
                        results[class_name].append(1 if torch.max(overlap) > self.iou_thresh else 0)

        grd_accu = np.mean([sum(hm)*1./len(hm) for i,hm in results.items()])

        print('localization_accuracy: {:.4f}\n'.format(grd_accu))

        return grd_accu


    def grd_eval(self, mode='all'):

        dirpath = os.getcwd()
        tools = os.path.join(dirpath, 'program/stanford-corenlp-full-2018-02-27')

        ref = self.ref
        pred = self.pred

        print('before stanfordnlp')

        nlp = StanfordCoreNLP(tools)
        props={'annotators': 'lemma','pipelineLanguage':'en', 'outputFormat':'json'}
        vocab_in_split = set()

        print('here')

        # precision
        prec = defaultdict(list)
        for vid, anns in ref.items():
            for seg, ann in anns['segments'].items():
                if len(ann['frame_ind']) == 0 or vid not in pred or seg not in pred[vid]:
                    continue # do not penalize if sentence not annotated

                ref_bbox_all = torch.cat((torch.Tensor(ann['process_bnd_box']), \
                    torch.Tensor(ann['frame_ind']).unsqueeze(-1)), dim=1) # 5-D coordinates

                idx_in_sent = {}
                for box_idx, cls_lst in enumerate(ann['process_clss']):
                    vocab_in_split.update(set(cls_lst))
                    for cls_idx, cls in enumerate(cls_lst):
                        idx_in_sent[cls] = idx_in_sent.get(cls, []) + [ann['process_idx'][box_idx][cls_idx]]

                sent_idx = set(itertools.chain.from_iterable(ann['process_idx'])) # index of gt object words
                exclude_obj = {json.loads(nlp.annotate(token.encode('utf-8'), properties=props) \
                    )['sentences'][0]['tokens'][0]['lemma']:1 for token_idx, token in enumerate(ann['tokens'] \
                    ) if (token_idx not in sent_idx and token != '')}

                for pred_idx, class_name in enumerate(pred[vid][seg]['clss']):
                    if class_name in idx_in_sent:
                        gt_idx = min(idx_in_sent[class_name]) # always consider the first match...
                        sel_idx = [idx for idx, i in enumerate(ann['process_idx']) if gt_idx in i]
                        ref_bbox = ref_bbox_all[sel_idx] # select matched boxes
                        assert(ref_bbox.size(0) > 0)

                        pred_bbox = torch.cat((torch.Tensor(pred[vid][seg]['bbox_for_all_frames'][pred_idx])[:,:4], \
                            torch.Tensor(range(10)).unsqueeze(-1)), dim=1)

                        frm_mask = torch.from_numpy(get_frm_mask(pred_bbox[:, 4].numpy(), \
                            ref_bbox[:, 4].numpy()).astype('uint8'))
                        overlap = bbox_overlaps_batch(pred_bbox[:, :5].unsqueeze(0), \
                            ref_bbox[:, :5].unsqueeze(0), frm_mask.unsqueeze(0))
                        prec[class_name].append(1 if torch.max(overlap) > self.iou_thresh else 0)
                    elif json.loads(nlp.annotate(class_name.encode('utf-8'), properties=props))['sentences'][0]['tokens'][0]['lemma'] in exclude_obj:
                        pass # do not penalize if gt object word not annotated (missed)
                    else:
                        if mode == 'all':
                            prec[class_name].append(0) # hallucinated object

        nlp.close()

        # recall
        recall = defaultdict(list)
        for vid, anns in ref.items():
            for seg, ann in anns['segments'].items():
                if len(ann['frame_ind']) == 0:
                    # print('no annotation available')
                    continue

                ref_bbox_all = torch.cat((torch.Tensor(ann['process_bnd_box']), \
                    torch.Tensor(ann['frame_ind']).unsqueeze(-1)), dim=1) # 5-D coordinates
                sent_idx = set(itertools.chain.from_iterable(ann['process_idx'])) # index of gt object words

                for gt_idx in sent_idx:
                    sel_idx = [idx for idx, i in enumerate(ann['process_idx']) if gt_idx in i]
                    ref_bbox = ref_bbox_all[sel_idx] # select matched boxes
                    # Note that despite discouraged, a single word could be annotated across multiple boxes/frames
                    assert(ref_bbox.size(0) > 0)

                    class_name = ann['process_clss'][sel_idx[0]][ann['process_idx'][sel_idx[0]].index(gt_idx)]
                    if vid not in pred:
                        recall[class_name].append(0) # video not grounded
                    elif seg not in pred[vid]:
                        recall[class_name].append(0) # segment not grounded
                    elif class_name in pred[vid][seg]['clss']:
                        pred_idx = pred[vid][seg]['clss'].index(class_name) # always consider the first match...
                        pred_bbox = torch.cat((torch.Tensor(pred[vid][seg]['bbox_for_all_frames'][pred_idx])[:,:4], \
                            torch.Tensor(range(10)).unsqueeze(-1)), dim=1)

                        frm_mask = torch.from_numpy(get_frm_mask(pred_bbox[:, 4].numpy(), \
                            ref_bbox[:, 4].numpy()).astype('uint8'))
                        overlap = bbox_overlaps_batch(pred_bbox[:, :5].unsqueeze(0), \
                            ref_bbox[:, :5].unsqueeze(0), frm_mask.unsqueeze(0))
                        recall[class_name].append(1 if torch.max(overlap) > self.iou_thresh else 0)
                    else:
                        if mode == 'all':
                            recall[class_name].append(0) # object not grounded

        print('3')

        num_vocab = len(vocab_in_split)
        prec_accu = np.sum([sum(hm)*1./len(hm) for i,hm in prec.items()])*1./num_vocab
        recall_accu = np.sum([sum(hm)*1./len(hm) for i,hm in recall.items()])*1./num_vocab
        f1 = 2. * prec_accu * recall_accu / (prec_accu + recall_accu)

        print('precision_{0}: {1:.4f}'.format(mode, prec_accu))
        print('recall_{0}: {1:.4f}'.format(mode, recall_accu))
        print('F1_{0}: {1:.4f}'.format(mode, f1))
        return prec_accu, recall_accu, f1


def main():

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    submit_dir = os.path.join(input_dir, 'res')
    files = os.listdir(submit_dir)
    gen = ''
    gt = ''
    for f in files:
        if 'gen' in f:
            gen = os.path.join(submit_dir, f)
        if 'gt' in f:
            gt = os.path.join(submit_dir, f)

    print(gen, gt)

    truth_dir = os.path.join(input_dir, 'ref')

    if not os.path.isdir(submit_dir):
        print("%s doesn't exist" % submit_dir)

    if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    output_filename = os.path.join(output_dir, 'scores.txt')
    output_file = open(output_filename, 'wb')

    reference_filename = os.path.join(truth_dir, 'ref.json')
    split_filename = os.path.join(truth_dir, 'split.json')

    if gt != '':
        print('gt')
        input_filename = gt
        grd_evaluator = ANetGrdEval(reference_file=reference_filename, submission_file=input_filename, split_file=split_filename, val_split=['validation'])
        with open(input_filename, 'r') as f:
            grd_accu = grd_evaluator.gt_grd_eval()
            output_file.write('localization_accuracy: {:.4f}\n'.format(grd_accu))
    else:
        print('no gt')
        output_file.write('localization_accuracy: 0\n')
    if gen != '':
        print('gen')
        input_filename = gen
        grd_evaluator = ANetGrdEval(reference_file=reference_filename, submission_file=input_filename, split_file=split_filename, val_split=['validation'])
        with open(input_filename, 'r') as f:
            prec_accu, recall_accu, f1 = grd_evaluator.grd_eval(mode='all')
            output_file.write('precision_all: {0:.4f}\n'.format(prec_accu))
            output_file.write('recall_all: {0:.4f}\n'.format(recall_accu))
            output_file.write('F1_all: {0:.4f}\n'.format(f1))
            prec_accu, recall_accu, f1 = grd_evaluator.grd_eval(mode='loc')
            output_file.write('precision_loc: {0:.4f}\n'.format(prec_accu))
            output_file.write('recall_loc: {0:.4f}\n'.format(recall_accu))
            output_file.write('F1_loc: {0:.4f}\n'.format(f1))
    else:
        print('no gen')
        output_file.write('precision_all: 0\n')
        output_file.write('recall_all: 0\n')
        output_file.write('F1_all: 0\n')
        output_file.write('precision_loc: 0\n')
        output_file.write('recall_loc: 0\n')
        output_file.write('F1_loc: 0\n')
    output_file.close()


if __name__=='__main__':
    main()

