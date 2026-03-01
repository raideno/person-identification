[group('python')]
python-setup:
    echo "create python environment"

[group('python')]
python-install:
    echo "install python dependencies"

[group('data')]
data-download:
    echo "download raw data"

[group('data')]
data-preprocess:
    echo "preprocess raw data and generate output folder"
    echo "will run all drivers"