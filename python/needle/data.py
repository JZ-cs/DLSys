import numpy as np
from .autograd import Tensor
import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
from needle import backend_ndarray as nd
import copy

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if(flip_img):
            h, w, c = img.shape
            for i in range(w//2):
                left = i
                right = w - 1 - left
                # import pdb; pdb.set_trace()
                tmp = copy.deepcopy(img[:, left, :])
                img[:, left, :] = img[:, right, :]
                img[:, right, :] = tmp
        return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(
            low=-self.padding, high=self.padding + 1, size=2
        )
        ### BEGIN YOUR SOLUTION
        h, w, c = img.shape
        new_h = h + self.padding*2
        new_w = w + self.padding*2
        new_img = np.zeros((new_h, new_w, c), dtype=img.dtype)
        new_img[self.padding : (new_h-self.padding), self.padding : (new_w-self.padding), :] = img
        h_left = self.padding + shift_x
        h_right = h_left + h
        w_left = self.padding + shift_y
        w_right = w_left + w
        return new_img[h_left:h_right, w_left:w_right, :]
        ### END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        # if(batch_size is None):
        #     batch_size = 1
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))
        # else:
        #     self.ordering = np.array_split(np.arange(len(dataset)), 
        #                                    range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        self.i_start = 0
        self.i_end = (len(self.dataset) + self.batch_size - 1) // self.batch_size
        if(self.shuffle):
            # np.random.shuffle(self.ordering)
            idx_ordering = np.arange(len(self.dataset))
            np.random.shuffle(idx_ordering)
            self.ordering = np.array_split(idx_ordering, 
                                           range(self.batch_size, len(self.dataset), self.batch_size))
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if(self.i_start < self.i_end):
            minibatch = self.dataset[self.ordering[self.i_start]]
            self.i_start += 1
            # import pdb; pdb.set_trace()
            return tuple([Tensor(minibatch[i]) for i in range(len(minibatch))])
        else:
            raise StopIteration
        ### END YOUR SOLUTION


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        self.base_folder = base_folder
        self.train = train
        self.p = p
        self.transforms = transforms

        if(self.train):
            data_list = []
            label_list = []
            file_names = ['data_batch_'+str(i) for i in range(1, 6)]
            for fname in file_names:
                data_path = self.base_folder + '/' + fname
                with open(data_path, 'rb') as f:
                    raw_data_dict = pickle.load(f, encoding='bytes')
                    data_list.extend(raw_data_dict[b'data'])
                    label_list.extend(raw_data_dict[b'labels'])
            data_set = np.array(data_list)
            labels = np.array(label_list)
        else:
            fname = 'test_batch'
            data_path = self.base_folder + '/' + fname
            with open(data_path, 'rb') as f:
                raw_data_dict = pickle.load(f, encoding='bytes')
                data_set = np.array(raw_data_dict[b'data'])
                labels = np.array(raw_data_dict[b'labels'])
        self.data_set = data_set
        self.labels = labels
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        if(isinstance(index, int)):
            img = np.reshape(self.data_set[index], (3,32,32)) # 1, 3072
            if(self.transforms is not None):
                _img = np.transpose(img, axes=(1,2,0)) # c,h,w -> h,w,c
                _img = self.apply_transforms(_img)
                img = np.transpose(_img, axes=(2,0,1)) # h,w,c -> c,h,w
            return img, self.labels[index]
        else:
            assert(isinstance(index, np.ndarray))
            bsz = len(index)
            imgs = np.reshape(self.data_set[index], (bsz,3,32,32)) # 1, 3072
            if(self.transforms is not None):
                _imgs = np.transpose(imgs, axes=(0,2,3,1)) # n,c,h,w -> n,h,w,c
                for i in range(len(_imgs)):
                    _imgs[i] = self.apply_transforms(_img[i])
                imgs = np.transpose(_imgs, axes=(0,3,1,2)) # n,h,w,c -> n,c,h,w
            return imgs, self.labels[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return len(self.data_set)
        ### END YOUR SOLUTION


class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])






class Dictionary(object):
    """
    Creates a dictionary from a list of words, mapping each word to a
    unique integer.
    Attributes:
    word2idx: dictionary mapping from a word to its unique ID
    idx2word: list of words in the dictionary, in the order they were added
        to the dictionary (i.e. each word only appears once in this list)
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        """
        Input: word of type str
        If the word is not in the dictionary, adds the word to the dictionary
        and appends to the list of words.
        Returns the word's unique ID.
        """
        ### BEGIN YOUR SOLUTION
        if(word not in self.word2idx.keys()):
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
        return self.word2idx[word]
        ### END YOUR SOLUTION

    def __len__(self):
        """
        Returns the number of unique words in the dictionary.
        """
        ### BEGIN YOUR SOLUTION
        return len(self.idx2word)
        ### END YOUR SOLUTION



class Corpus(object):
    """
    Creates corpus from train, and test txt files.
    """
    def __init__(self, base_dir, max_lines=None):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(base_dir, 'train.txt'), max_lines)
        self.test = self.tokenize(os.path.join(base_dir, 'test.txt'), max_lines)

    def tokenize(self, path, max_lines=None):
        """
        Input:
        path - path to text file
        max_lines - maximum number of lines to read in
        Tokenizes a text file, first adding each word in the file to the dictionary,
        and then tokenizing the text file to a list of IDs. When adding words to the
        dictionary (and tokenizing the file content) '<eos>' should be appended to
        the end of each line in order to properly account for the end of the sentence.
        Output:
        ids: List of ids
        """
        ### BEGIN YOUR SOLUTION
        ids = []
        def empty_word(word:str):
            if(word is None):
                return True
            if(len(word) == 0):
                return True
            if(word == '' or word == "" or word == '\n' or word==' ' or word=='\t'):
                return True
            return False
        
        with open(path) as f:
            line_cnt = 0
            for line in f.readlines():
                _line = line + ' <eos>'
                words = _line.split(' ')
                for word in words:
                    if(not empty_word(word)):
                        ids.append(self.dictionary.add_word(word))
                line_cnt += 1
                if(line_cnt == max_lines):
                    break
        return ids
        ### END YOUR SOLUTION


def batchify(data, batch_size, device, dtype):
    """
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' cannot be learned, but allows more efficient
    batch processing.
    If the data cannot be evenly divided by the batch size, trim off the remainder.
    Returns the data as a numpy array of shape (nbatch, batch_size).
    """
    ### BEGIN YOUR SOLUTION
    len_col = len(data)//batch_size
    _len = len_col * batch_size
    batchified_data = np.array(data[:_len])
    batchified_data = np.reshape(batchified_data, (len_col, batch_size))
    return batchified_data
    ### END YOUR SOLUTION


def get_batch(batches, i, bptt, device=None, dtype=None):
    """
    get_batch subdivides the source data into chunks of length bptt.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM or RNN.
    Inputs:
    batches - numpy array returned from batchify function
    i - index
    bptt - Sequence length
    Returns:
    data - Tensor of shape (bptt, bs) with cached data as NDArray
    target - Tensor of shape (bptt*bs,) with cached data as NDArray
    """
    ### BEGIN YOUR SOLUTION
    idx_ed = i + bptt
    data_np = batches[i:idx_ed]
    _, bs = data_np.shape
    data = Tensor(nd.array(data_np, device=device, dtype=dtype), device=device, dtype=dtype)
    target_np = np.reshape(data_np, (bptt*bs, ))
    target = Tensor(nd.array(target_np, device=device, dtype=dtype), device=device, dtype=dtype)
    return data, target
    ### END YOUR SOLUTION