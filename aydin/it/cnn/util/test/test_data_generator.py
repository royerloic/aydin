import numpy
from aydin.it.cnn.util.data_generator import SimpleDataGenerator, DataGenerator_2in2out

data = numpy.ones((5, 10, 10, 10, 1))
data_3d = []
for i in range(10):
    data_3d.append(i * data)
data_3d = numpy.vstack(data_3d)


def test_SimpleDataGenerator():
    datagen = SimpleDataGenerator(data_3d, batch_size=5, shuffle=False)
    for i, d in enumerate(datagen):
        assert int(numpy.unique(d)) == i
        assert d.shape == (5, 10, 10, 10, 1)
        if i == 9:
            break


def test_SimpleDataGenerator_shuffle():
    datagen = SimpleDataGenerator(data_3d, batch_size=5, shuffle=True)
    for i, d in enumerate(datagen):
        assert d.shape == (5, 10, 10, 10, 1)
        if i == 9:
            break


def test_DataGenerator2():
    datagen = DataGenerator_2in2out(data_3d, data_3d, batch_size=5, shuffle=False)
    for i, d in enumerate(datagen):
        assert int(numpy.unique(d[0])) == i
        assert int(numpy.unique(d[0])) == i
        assert d[0].shape == (5, 10, 10, 10, 1)
        assert d[1].shape == (5, 10, 10, 10, 1)
        if i == 9:
            break


def test_DataGenerator2_shuffle():
    datagen = DataGenerator_2in2out(data_3d, data_3d, batch_size=5, shuffle=True)
    for i, d in enumerate(datagen):
        assert d[0].shape == (5, 10, 10, 10, 1)
        assert d[1].shape == (5, 10, 10, 10, 1)
        if i == 9:
            break
