from pitl.analysis.correlation import correlation_distance
from pitl.io import io
from pitl.io.datasets import examples_single


def demo_analysis():

    for example in examples_single:
        example_file_path = example.get_path()

        # print(f"Trying to open and make sense of file {example_file_path}")

        array, metadata = io.imread(example_file_path)
        print(f"File        :  {example}")
        print(f"Metadata    :  {metadata}")
        print(f"Array shape :  {array.shape}")
        print(f"Array dtype :  {array.dtype}")

        resolution = resolution(array)
        print(f"Resolution:  {resolution} ")


demo_analysis()
