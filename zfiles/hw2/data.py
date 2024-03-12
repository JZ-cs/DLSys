import numpy as np
from .autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import copy
import gzip, struct

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
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
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
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
        images_arrays, self.labels = parse_mnist(image_filename, label_filename)
        bsz, pixlen = images_arrays.shape
        self.images = np.reshape(images_arrays, (bsz, 28, 28, 1))
        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        img = self.images[index]
        # import pdb; pdb.set_trace()
        _shape = []
        for _shp in img.shape[:-3]:
            _shape.append(_shp)
        _shape.append(784)
        transformed_img = self.apply_transforms(img)
        return np.reshape(transformed_img, _shape), self.labels[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.images)
        ### END YOUR SOLUTION

class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0. The normalization should be applied uniformly
                across the whole dataset, _not_ individual images.

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    def normalization(data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range
    with gzip.open(label_filename, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        labels = np.frombuffer(lbpath.read(),dtype=np.uint8)
    with gzip.open(image_filename, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(len(labels), 784)
    # to fp32
    images_normd = normalization(images.astype(np.float32))
    return images_normd, labels
    ### END YOUR CODE