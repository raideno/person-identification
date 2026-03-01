We can take a classical person detection / identification dataset and augment it by taking crops of persons, etc and train on it.

We could also scrape random Youtube videos, cctv fotoage, videos of big avenues, etc and detect persons on that by taking small crops and using other models to have a good accuracy, and then train our model on that.

- https://huggingface.co/datasets/UniqueData/people-tracking-dataset/tree/main
- https://www.kaggle.com/datasets/adilshamim8/people-detection/discussion?sort=hotness
- https://universe.roboflow.com/titulacin/person-detection-9a6mk/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true
- https://github.com/ViswanathaReddyGajjala/Datasets

The dataset construction will happen prior to the model training, it'll be an outside thing, we'll be having a data folder, inside we'll be having a datasets folder were we'll be storing the raw format downloaded dataset from the internet in a folder and a drivers folder where we'll have files with the same names as the folders in the datasets, for each dataset we'll have an adapter and its goal is to parse the dataset and output it in a common format and store it inside the output folder in a final format. So we'll merge all the datasets.
Then we'll be having a script to run all the drivers at once and generate the final dataset, the raw datasets downloaded from the internet will be stored in some drive and will be able to be downloaded by running some script as well.

The output folder data will have a clear format for both images and annotations and can be easily used with the project. The data in the output folder will be all annotated and a bit augmented since some augmentation require quite a bit of computation it won't be done during training. Some data drivers will include loading models and processing images to do detection etc as the original datasets might not be annotated at all.
