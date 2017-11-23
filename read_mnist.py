import struct
import gzip
import numpy as np

def label_10d(num):
    label      = np.zeros(10, dtype=np.float32)
    label[num] = 1.0
    return label

def read_mnist():
    rootdir            = "data/"
    file_train_label   = rootdir + "train-labels-idx1-ubyte.gz"
    file_train_images  = rootdir + "train-images-idx3-ubyte.gz"
    file_test_label    = rootdir + "t10k-labels-idx1-ubyte.gz"
    file_test_images   = rootdir + "t10k-images-idx3-ubyte.gz"
    
    with gzip.open(file_train_label, "rb") as f:
        data           = struct.unpack_from(">2i60000B", f.read(), offset=0) # training data, 60,000 examples
        magic          = data[0]
        ndata          = data[1]
        train_labels   = np.array(data[2:], dtype=np.int32)

    with gzip.open(file_train_images, "rb") as f:
        data           = struct.unpack_from(">4i47040000B", f.read(), offset=0) # 47040000 = 60000 * 28 * 28
        magic          = data[0]
        ndata          = data[1]
        nrows          = data[2]
        ncols          = data[3]
        train_images   = np.array(data[4:], dtype=np.float32).reshape((ndata, nrows*ncols))

    with gzip.open(file_test_label, "rb") as f:
        data           = struct.unpack_from(">2i10000B", f.read(), offset=0) # test data, 10,000 examples
        magic          = data[0]
        ndata          = data[1]
        test_labels   = np.array(data[2:], dtype=np.int32)

    with gzip.open(file_test_images, "rb") as f:
        data           = struct.unpack_from(">4i7840000B", f.read(), offset=0) # 7840000 = 10000 * 28 * 28
        magic          = data[0]
        ndata          = data[1]
        nrows          = data[2]
        ncols          = data[3]
        test_images   = np.array(data[4:], dtype=np.float32).reshape((ndata, nrows*ncols))

    # the original value is uint8 so 0 - 255, so scale the value to 0.0 - 1.0
    train_images = [x/255.0 for x in train_images]
    test_images  = [x/255.0 for x in test_images]
        
    # split 60,000 training data into two pieces, 50,000 data will be used for actual training,
    # and remaining 10,000 data will be used for validation
    train_images0 = train_images[0:50000]
    train_labels0 = train_labels[0:50000]
    train_images1 = train_images[50000:60000]
    train_labels1 = train_labels[50000:60000]

    train_data      = list(zip([np.reshape(x, (784)) for x in train_images0], [label_10d(x) for x in train_labels0]))
    validation_data = list(zip([np.reshape(x, (784)) for x in train_images1], [label_10d(x) for x in train_labels1]))
    test_data       = list(zip([np.reshape(x, (784)) for x in test_images],   [label_10d(x) for x in test_labels]))

    return (train_data, validation_data, test_data)
