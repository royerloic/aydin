import numpy
from copy import deepcopy


def SimpleDataGenerator(image, batch_size, shuffle=True):
    image = deepcopy(image)
    if shuffle:
        numpy.random.shuffle(image)

    j = 0
    num_cycle = numpy.ceil(image.shape[0] / batch_size)
    while True:
        i = numpy.mod(j, num_cycle).astype(int)
        img_output = image[batch_size * i : batch_size * (i + 1)]
        j += 1
        yield img_output


def DataGenerator_2in2out(input1, input2, batch_size, shuffle=True):
    input1 = deepcopy(input1)
    input2 = deepcopy(input2)
    assert (
        input1.shape[0] == input2.shape[0]
    ), 'Two inputs has to have the same length in dim 0'

    if shuffle:
        ind = numpy.random.randint(0, input1.shape[0], size=batch_size)
        input1 = input1[ind]
        input2 = input2[ind]

    j = 0
    num_cycle = numpy.ceil(input1.shape[0] / batch_size)
    while True:
        i = numpy.mod(j, num_cycle).astype(int)
        img_output1 = input1[batch_size * i : batch_size * (i + 1)]
        img_output2 = input2[batch_size * i : batch_size * (i + 1)]
        j += 1
        yield img_output1, img_output2
