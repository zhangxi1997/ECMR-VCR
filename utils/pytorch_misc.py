"""
Question relevance model
"""

# Make stuff
import os
import re
import shutil
import time

import numpy as np
import pandas as pd
import torch
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.nn.util import device_mapping
from allennlp.training.trainer import move_optimizer_to_cuda
from torch.nn import DataParallel
import torch.nn as nn


def time_batch(gen, reset_every=100):
    """
    Gets timing info for a batch
    :param gen:
    :param reset_every: How often we'll reset
    :return:
    """
    start = time.time()
    start_t = 0
    for i, item in enumerate(gen):
        time_per_batch = (time.time() - start) / (i + 1 - start_t)
        yield time_per_batch, item
        if i % reset_every == 0:
            start = time.time()
            start_t = i


class Flattener(torch.nn.Module):
    def __init__(self):
        """
        Flattens last 3 dimensions to make it only batch size, -1
        """
        super(Flattener, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def pad_sequence(sequence, lengths):
    """
    :param sequence: [\sum b, .....] sequence
    :param lengths: [b1, b2, b3...] that sum to \sum b
    :return: [len(lengths), maxlen(b), .....] tensor
    """
    output = sequence.new_zeros(len(lengths), max(lengths), *sequence.shape[1:])
    start = 0
    for i, diff in enumerate(lengths):
        if diff > 0:
            output[i, :diff] = sequence[start:(start + diff)]
        start += diff
    return output


def extra_leading_dim_in_sequence(f, x, mask):
    return f(x.view(-1, *x.shape[2:]), mask.view(-1, mask.shape[2])).view(*x.shape[:3], -1)


def clip_grad_norm(named_parameters, max_norm, clip=True, verbose=False):
    """Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Variable]): an iterable of Variables that will have
            gradients normalized
        max_norm (float or int): max norm of the gradients

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    max_norm = float(max_norm)
    parameters = [(n, p) for n, p in named_parameters if p.grad is not None]
    total_norm = 0
    param_to_norm = {}
    param_to_shape = {}
    for n, p in parameters:

        param_norm = p.grad.data.norm(2)
        total_norm += param_norm ** 2
        param_to_norm[n] = param_norm
        param_to_shape[n] = tuple(p.size())
        if np.isnan(param_norm.item()):
            raise ValueError("the param {} was null.".format(n))

    total_norm = total_norm ** (1. / 2)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef.item() < 1 and clip:
        for n, p in parameters:
            p.grad.data.mul_(clip_coef)

    if verbose:
        print('---Total norm {:.3f} clip coef {:.3f}-----------------'.format(total_norm, clip_coef))
        for name, norm in sorted(param_to_norm.items(), key=lambda x: -x[1]):
            print("{:<60s}: {:.3f}, ({}: {})".format(name, norm, np.prod(param_to_shape[name]), param_to_shape[name]))
        print('-------------------------------', flush=True)

    return pd.Series({name: norm.item() for name, norm in param_to_norm.items()})


def find_latest_checkpoint(serialization_dir):
    """
    Return the location of the latest model and training state files.
    If there isn't a valid checkpoint then return None.
    """
    have_checkpoint = (serialization_dir is not None and
                       any("model_state_epoch_" in x for x in os.listdir(serialization_dir)))

    if not have_checkpoint:
        return None

    serialization_files = os.listdir(serialization_dir)
    model_checkpoints = [x for x in serialization_files if "model_state_epoch" in x]
    # Get the last checkpoint file.  Epochs are specified as either an
    # int (for end of epoch files) or with epoch and timestamp for
    # within epoch checkpoints, e.g. 5.2018-02-02-15-33-42
    found_epochs = [
        # pylint: disable=anomalous-backslash-in-string
        re.search("model_state_epoch_([0-9\.\-]+)\.th", x).group(1)
        for x in model_checkpoints
    ]
    int_epochs = []
    for epoch in found_epochs:
        pieces = epoch.split('.')
        if len(pieces) == 1:
            # Just a single epoch without timestamp
            int_epochs.append([int(pieces[0]), 0])
        else:
            # has a timestamp
            int_epochs.append([int(pieces[0]), pieces[1]])
    last_epoch = sorted(int_epochs, reverse=True)[0]
    if last_epoch[1] == 0:
        epoch_to_load = str(last_epoch[0])
    else:
        epoch_to_load = '{0}.{1}'.format(last_epoch[0], last_epoch[1])

    model_path = os.path.join(serialization_dir,
                              "model_state_epoch_{}.th".format(epoch_to_load))
    training_state_path = os.path.join(serialization_dir,
                                       "training_state_epoch_{}.th".format(epoch_to_load))
    return model_path, training_state_path


def save_checkpoint(model, optimizer, serialization_dir, epoch, val_metric_per_epoch, is_best=None,
                    learning_rate_scheduler=None) -> None:
    """
    Saves a checkpoint of the model to self._serialization_dir.
    Is a no-op if self._serialization_dir is None.
    Parameters
    ----------
    epoch : Union[int, str], required.
        The epoch of training.  If the checkpoint is saved in the middle
        of an epoch, the parameter is a string with the epoch and timestamp.
    is_best: bool, optional (default = None)
        A flag which causes the model weights at the given epoch to
        be copied to a "best.th" file. The value of this flag should
        be based on some validation metric computed by your model.
    """
    if serialization_dir is not None:
        model_path = os.path.join(serialization_dir, "model_state_epoch_{}.th".format(epoch))
        model_state = model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict()
        torch.save(model_state, model_path)

        training_state = {'epoch': epoch,
                          'val_metric_per_epoch': val_metric_per_epoch,
                          'optimizer': optimizer.state_dict()
                          }
        if learning_rate_scheduler is not None:
            training_state["learning_rate_scheduler"] = \
                learning_rate_scheduler.lr_scheduler.state_dict()
        training_path = os.path.join(serialization_dir,
                                     "training_state_epoch_{}.th".format(epoch))
        torch.save(training_state, training_path)
        if is_best:
            print("Best validation performance so far. Copying weights to '{}/best.th'.".format(serialization_dir))
            shutil.copyfile(model_path, os.path.join(serialization_dir, "best.th"))


def restore_best_checkpoint(model, serialization_dir):
    fn = os.path.join(serialization_dir, 'best.th')
    model_state = torch.load(fn, map_location=device_mapping(-1))
    assert os.path.exists(fn)
    if isinstance(model, DataParallel):
        model.module.load_state_dict(model_state)
    else:
        model.load_state_dict(model_state)


def restore_checkpoint(model, optimizer, serialization_dir, learning_rate_scheduler=None):
    """
    Restores a model from a serialization_dir to the last saved checkpoint.
    This includes an epoch count and optimizer state, which is serialized separately
    from  model parameters. This function should only be used to continue training -
    if you wish to load a model for inference/load parts of a model into a new
    computation graph, you should use the native Pytorch functions:
    `` model.load_state_dict(torch.load("/path/to/model/weights.th"))``
    If ``self._serialization_dir`` does not exist or does not contain any checkpointed weights,
    this function will do nothing and return 0.
    Returns
    -------
    epoch: int
        The epoch at which to resume training, which should be one after the epoch
        in the saved training state.
    """
    latest_checkpoint = find_latest_checkpoint(serialization_dir)
    print ('restore check point')

    if latest_checkpoint is None:
        # No checkpoint to restore, start at 0
        return 0, []

    model_path, training_state_path = latest_checkpoint

    # Load the parameters onto CPU, then transfer to GPU.
    # This avoids potential OOM on GPU for large models that
    # load parameters onto GPU then make a new GPU copy into the parameter
    # buffer. The GPU transfer happens implicitly in load_state_dict.
    model_state = torch.load(model_path, map_location=device_mapping(-1))
    training_state = torch.load(training_state_path, map_location=device_mapping(-1))


    #model_dict = model.state_dict()
    #pretrained_dict = {k: v for k,v in model_state.items() if k in model_dict}
    #model_dict.update(pretrained_dict)

    if isinstance(model, DataParallel):
        model.module.load_state_dict(model_state)
    else:
        model.load_state_dict(model_state)

    # idk this is always bad luck for me
    optimizer.load_state_dict(training_state["optimizer"])

    print (training_state.keys())
    print (learning_rate_scheduler is not None)
    print ("learning_rate_scheduler" in training_state)
    if learning_rate_scheduler is not None and "learning_rate_scheduler" in training_state:
        print ('test scheduler')
        learning_rate_scheduler.lr_scheduler.load_state_dict(
            training_state["learning_rate_scheduler"])
        print (learning_rate_scheduler.lr_scheduler.state_dict())
    move_optimizer_to_cuda(optimizer)

    # We didn't used to save `validation_metric_per_epoch`, so we can't assume
    # that it's part of the trainer state. If it's not there, an empty list is all
    # we can do.
    if "val_metric_per_epoch" not in training_state:
        print("trainer state `val_metric_per_epoch` not found, using empty list")
    else:
        val_metric_per_epoch = training_state["val_metric_per_epoch"]

    if isinstance(training_state["epoch"], int):
        epoch_to_return = training_state["epoch"] + 1
    else:
        epoch_to_return = int(training_state["epoch"].split('.')[0]) + 1
    return epoch_to_return, val_metric_per_epoch


def detokenize(array, vocab):
    """
    Given an array of ints, we'll turn this into a string or a list of strings.
    :param array: possibly multidimensional numpy array
    :return:
    """
    if array.ndim > 1:
        return [detokenize(x, vocab) for x in array]
    tokenized = [vocab.get_token_from_index(v) for v in array]
    return ' '.join([x for x in tokenized if x not in (vocab._padding_token, START_SYMBOL, END_SYMBOL)])


def print_para(model):
    """
    Prints parameters of a model
    :param opt:
    :return:
    """
    st = {}
    total_params = 0
    total_params_training = 0
    for p_name, p in model.named_parameters():
        # if not ('bias' in p_name.split('.')[-1] or 'bn' in p_name.split('.')[-1]):
        st[p_name] = ([str(x) for x in p.size()], np.prod(p.size()), p.requires_grad)
        total_params += np.prod(p.size())
        if p.requires_grad:
            total_params_training += np.prod(p.size())
    pd.set_option('display.max_columns', None)
    shapes_df = pd.DataFrame([(p_name, '[{}]'.format(','.join(size)), prod, p_req_grad)
                              for p_name, (size, prod, p_req_grad) in sorted(st.items(), key=lambda x: -x[1][1])],
                             columns=['name', 'shape', 'size', 'requires_grad']).set_index('name')

    print('\n {:.1f}M total parameters. {:.1f}M training \n ----- \n {} \n ----'.format(total_params / 1000000.0,
                                                                                        total_params_training / 1000000.0,
                                                                                        shapes_df.to_string()),
          flush=True)
    return shapes_df


def batch_index_iterator(len_l, batch_size, skip_end=True):
    """
    Provides indices that iterate over a list
    :param len_l: int representing size of thing that we will
        iterate over
    :param batch_size: size of each batch
    :param skip_end: if true, don't iterate over the last batch
    :return: A generator that returns (start, end) tuples
        as it goes through all batches
    """
    iterate_until = len_l
    if skip_end:
        iterate_until = (len_l // batch_size) * batch_size

    for b_start in range(0, iterate_until, batch_size):
        yield (b_start, min(b_start + batch_size, len_l))


def batch_iterator(seq, batch_size, skip_end=True):
    for b_start, b_end in batch_index_iterator(len(seq), batch_size, skip_end=skip_end):
        yield seq[b_start:b_end]

def convert_to_one_hot(indices, num_classes):
    batch_size = indices.size(0)
    indices = indices.unsqueeze(1)
    one_hot = indices.data.new(batch_size, num_classes).zero_().scatter_(1, indices.data, 1)
    return one_hot

def reverse_padded_sequence(inputs, lengths, batch_first=False):
    """Reverses sequences according to their lengths.
    Inputs should have size ``T x B x *`` if ``batch_first`` is False, or
    ``B x T x *`` if True. T is the length of the longest sequence (or larger),
    B is the batch size, and * is any number of dimensions (including 0).
    Arguments:
        inputs (Variable): padded batch of variable length sequences.
        lengths (list[int]): list of sequence lengths
        batch_first (bool, optional): if True, inputs should be B x T x *.
    Returns:
        A Variable with the same size as inputs, but with each sequence
        reversed according to its length.
    """

    if not batch_first:
        inputs = inputs.transpose(0, 1)
    if inputs.size(0) != len(lengths):
        raise ValueError('inputs incompatible with lengths.')
    reversed_indices = [list(range(inputs.size(1)))
                        for _ in range(inputs.size(0))]
    for i, length in enumerate(lengths):
        if length > 0:
            #length = int(length.cpu())
            reversed_indices[i][:length] = reversed_indices[i][length-1::-1]
    reversed_indices = torch.LongTensor(reversed_indices).unsqueeze(2).expand_as(inputs).to(inputs.device)
    reversed_inputs = torch.gather(inputs, 1, reversed_indices)
    if not batch_first:
        reversed_inputs = reversed_inputs.transpose(0, 1)
    return reversed_inputs

class NormalizeScale(nn.Module):
    def __init__(self, dim, init_norm=20):
        super(NormalizeScale, self).__init__()
        self.init_norm = init_norm
        #self.weight = nn.Parameter(torch.ones(1, dim) * init_norm) #trainable
        self.weight = init_norm

    def forward(self, bottom):
        bottom_normalized = nn.functional.normalize(bottom, p=2, dim=1)
        bottom_normalized_scaled = bottom_normalized * self.weight#.cuda()
        return bottom_normalized_scaled


import scipy.sparse as sp

def normalize_adj(adj):
    adj = adj.to_dense().cpu().numpy()
    adj = sp.coo_matrix(adj)     #构建张量
    rowsum = np.array(adj.sum(1))#每行的数加在一起
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()    #输出rowsum ** -1/2
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.            #溢出部分赋值为0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)            #对角化
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()    #转置后相乘
    return torch.FloatTensor(adj.todense()).cuda()

def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes
    得到bbox的坐标
    """
    # Get the coordinates of bounding boxes
    box1 = torch.from_numpy(box1)
    box2 = torch.from_numpy(box2)
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    if torch.cuda.is_available():
        inter_area = torch.max(inter_rect_x2 - inter_rect_x1 + 1, torch.zeros(inter_rect_x2.shape)) * torch.max(
            inter_rect_y2 - inter_rect_y1 + 1, torch.zeros(inter_rect_x2.shape))
    else:
        inter_area = torch.max(inter_rect_x2 - inter_rect_x1 + 1, torch.zeros(inter_rect_x2.shape)) * torch.max(
            inter_rect_y2 - inter_rect_y1 + 1, torch.zeros(inter_rect_x2.shape))

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou

def adjust_learning_rate(optimizer, epoch, rationale):
    init_lr = 0.0002
    lr1 = 0.2
    lr2 = 0.5
    lr = init_lr
    if rationale:
        if epoch >=13:
            lr = init_lr * lr1
        if epoch >=14:
            lr = init_lr * lr1 * lr2
    else:
        if epoch >= 11:
            lr = init_lr * lr1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def parse_check(rep, node_idx, relate_idx):
    if node_idx.shape[2] != rep.shape[2]:
        if node_idx.shape[2] > rep.shape[2]:
            new_node_idx = node_idx[:, :, :rep.shape[2]]
            new_relate_idx = relate_idx[:, :, :rep.shape[2]]
        else:
            question_node_idx_new = torch.zeros(rep.shape).cuda().float()
            question_node_idx_new[:, :, :node_idx.shape[2]] = node_idx
            new_node_idx = question_node_idx_new
            question_relate_idx_new = torch.zeros(rep.shape).cuda().float()
            question_relate_idx_new[:, :, :relate_idx.shape[2]] = relate_idx
            new_relate_idx = question_relate_idx_new
        return new_node_idx, new_relate_idx
    else:
        return node_idx, relate_idx

def parse_check_ans(rep, node_idx, relate_idx):
    if node_idx.shape[3] != rep.shape[2]:
        if node_idx.shape[3] > rep.shape[2]:
            new_node_idx = node_idx[:, :, :, :rep.shape[2]]
            new_relate_idx = relate_idx[:, :, :, :rep.shape[2]]
        else:
            answers_node_idx_new = torch.zeros(rep.shape).cuda().float()
            answers_node_idx_new[:, :, :, :node_idx.shape[3]] = node_idx
            new_node_idx = answers_node_idx_new
            answers_relate_idx_new = torch.zeros(rep.shape).cuda().float()
            answers_relate_idx_new[:, :, :, :relate_idx.shape[3]] = relate_idx
            new_relate_idx = answers_relate_idx_new
        return new_node_idx, new_relate_idx
    else:
        return node_idx, relate_idx

def gene_Adj(batch_size, length, sen_relate_idx, sen_relate_os, sen_node_idx, sen_mask, ans=False):
    adj_matrix = torch.zeros((batch_size, length, length), requires_grad=False)
    for mini in range(batch_size):
        for i in range(sen_relate_idx.shape[1]):
            if sen_relate_os[mini][i][0] != -1:  # locate the triplet
                obj = sen_relate_os[mini][i][0].long()
                sub = sen_relate_os[mini][i][1].long()
                relate_idx = sen_relate_idx[mini][i].long()
                obj_idx = sen_node_idx[mini][obj]
                sub_idx = sen_node_idx[mini][sub]

                relate_adj = torch.nonzero(relate_idx)  # word location in query
                obj_adj = torch.nonzero(obj_idx)
                sub_adj = torch.nonzero(sub_idx)

                for q in relate_adj:
                    for o in obj_adj:
                        adj_matrix[mini][o[0]][q[0]] = 1
                        adj_matrix[mini][q[0]][o[0]] = 1
                    for s in sub_adj:
                        adj_matrix[mini][s[0]][q[0]] = 1  # sub -> relate
                        adj_matrix[mini][q[0]][s[0]] = 1  # relate -> sub
        # self-loop
        for i in range(length):
            if ans:
                adj_matrix[mini][i][i] = sen_mask[mini][i]
            else:
                adj_matrix[mini][i][i] = sen_mask[mini][0][i]
    return adj_matrix
