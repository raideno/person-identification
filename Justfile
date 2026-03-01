[group('python')]
python-setup:
    echo "create python environment"

[group('python')]
python-install:
    echo "install python dependencies"

[group('data')]
data-download:
    bash data/download.sh

[group('data')]
data-preprocess:
    echo "preprocess raw data and generate output folder"
    echo "will run all drivers"
