import napari

from pitl.io import io
from pitl.io.datasets import examples_single


def demo_examples():
    """
        ....
    """

    for example in examples_single:
        example_file_path = example.get_path()

        # print(f"Trying to open and make sense of file {example_file_path}")

        array, metadata = io.imread(example_file_path)

        with napari.gui_qt():
            viewer = napari.Viewer()
            viewer.add_image(array, name='image')


demo_examples()
