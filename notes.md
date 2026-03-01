We can take a classical person detection / identification dataset and augment it by taking crops of persons, etc and train on it.

We could also scrape random Youtube videos, cctv fotoage, videos of big avenues, etc and detect persons on that by taking small crops and using other models to have a good accuracy, and then train our model on that.

- https://huggingface.co/datasets/UniqueData/people-tracking-dataset/tree/main
- https://www.kaggle.com/datasets/adilshamim8/people-detection/discussion?sort=hotness
- https://universe.roboflow.com/titulacin/person-detection-9a6mk/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true
- https://github.com/ViswanathaReddyGajjala/Datasets
- https://www.kaggle.com/datasets/adilshamim8/people-detection/data

The dataset construction will happen prior to the model training, it'll be an outside thing, we'll be having a data folder, inside we'll be having a datasets folder were we'll be storing the raw format downloaded dataset from the internet in a folder and a drivers folder where we'll have files with the same names as the folders in the datasets, for each dataset we'll have an adapter and its goal is to parse the dataset and output it in a common format and store it inside the output folder in a final format. So we'll merge all the datasets.
Then we'll be having a script to run all the drivers at once and generate the final dataset, the raw datasets downloaded from the internet will be stored in some drive and will be able to be downloaded by running some script as well.

The output folder data will have a clear format for both images and annotations and can be easily used with the project. The data in the output folder will be all annotated and a bit augmented since some augmentation require quite a bit of computation it won't be done during training. Some data drivers will include loading models and processing images to do detection etc as the original datasets might not be annotated at all.

Will make it so the outputed dataset will also be compatible for person identification and re-identification by providing the same id to a person if it comes from a video and this kind of things.

## Output Format

File naming convention: `<dataset>-<sequence-id>-<frame-id>.<ext>`

- `<dataset>`: Dataset name (e.g., "people-detection")
- `<sequence-id>`: UUID identifying a sequence (for videos) or standalone image
- `<frame-id>`: UUID uniquely identifying a frame within the sequence

Example: `people-detection-550e8400-e29b-41d4-a716-446655440000-6ba7b810-9dad-11d1-80b4-00c04fd430c8.jpg`

### Annotation Format

Each image has a corresponding `.json` file with the same base name containing:

```json
{
  "frame_id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
  "sequence_id": "550e8400-e29b-41d4-a716-446655440000",
  "dataset": "people-detection",
  "image_width": 500,
  "image_height": 375,
  "detections": [
    {
      "class_name": "person",
      "bbox": {
        "x_min": 219.0,
        "y_min": 98.0,
        "x_max": 269.0,
        "y_max": 283.0
      }
    }
  ]
}
```

### Image Files

- Format: JPEG
- Dimensions: Preserved from original dataset
- Location: `data/output/`

### Annotation Fields

- `frame_id`: Unique identifier for this frame
- `sequence_id`: Identifier for the sequence (same for all frames in a video, unique for standalone images)
- `dataset`: Source dataset name
- `image_width/height`: Original image dimensions in pixels
- `detections`: Array of detected objects
  - `class_name`: Object class (e.g., "person")
  - `bbox`: Bounding box with absolute pixel coordinates
    - `x_min, y_min`: Top-left corner
    - `x_max, y_max`: Bottom-right corner