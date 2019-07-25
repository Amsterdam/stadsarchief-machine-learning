
Classification of bouwdossiers

# Development

`./start.sh`

Open jupyter as on the web site listed in the terminal and open `TrainClassifier.ipynb` in a browser.



# Model persistence

The full model is stored in multiple files:
* model/model.json, Keras model architecture
* model/weights.json, Keras model weights
* model/transform/image_encoder.npy, image preprocessing encoder configuration
* model/transform/target_encoder.npy, label encoder configuration 


# Quickstart
- get access to the object store `9d078258c1a547c09e0b5f88834554f1:bouwdossiers` (credentials are under the 
name `cloudVPS Bouwdossiers` in rattic. Login the )
- download training set and put it into "examples/aanvraag_besluit"
- add `small_set.csv` in `input/` folder
- Set environment variables (manually set `BOUWDOSSIERS_OBJECTSTORE_PASSWORD`):

    export BOUWDOSSIERS_OBJECTSTORE_PASSWORD=xxx
    export PYTHONUNBUFFERED=1
    export IIIF_API_ROOT=https://acc.images.data.amsterdam.nl/iiif/2/edepot:
    export IIIF_CACHE_DIR=./cache
    export MODEL_DIR=../output/model/
    export OUTPUT_DIR=../output/prediction/

- try to run prediction on `small_set.csv` by running from the src folder: `./run_prediction.py ../input/small_set.csv`


# TODO:
- automate storing prediction-models in data store
- automate training 
- automate testing
- create Jenkins job to run specific model on specific data set 
