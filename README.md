# Machine Learning Stadsarchief
This is a start to use machine learning for the interpretation of a variety of data sources within the gemeente. The 
current state is work in progress for the specific aim to classify bouwdossiers from the Stadsarchief.


## Quickstart
It is possible to both train a new model, and use the resulting model to get predictions for new input.

### 0. Install
Install `requirements/requirements.txt`, possibly inside a virtualenv.

For **GPU support** install `tensorflow-gpu` instead of `tensorflow`.
GPU support is practically required for training.
Otherwise it will take ages.

### 1. Train a new model
Training a model is still done using:

    python run_training.py

After running a model is written to `output/model/`. The added training set is very small 
and doesn't give you a very good model. To do serious training you can use more data from the object store (credentials are under the name `cloudVPS Bouwdossiers` in Rattic). 
Download a training set there and put it into `examples/aanvraag_besluit/`.

### 2. Model evaluation
A model can be evaluated on a dataset using the `EvaluateModel` Jupyter notebook.
First start the notebook server:

    jupyter notebook
    
Then navigate to `src/notebook/EvaluateModel.ipynb`, change parameters for your use case and run everything.


### 3. Make new predictions
The model we trained in the previous step can be used to make predictions for new records/images. Below is an example 
of how to use a small set taken from the object store which we've got defined in `input/small_set.csv`. To do this you 
need to set a couple environment variables (manually set `BOUWDOSSIERS_OBJECTSTORE_PASSWORD` taken from Rattic above):

    export BOUWDOSSIERS_OBJECTSTORE_PASSWORD=xxx
    export PYTHONUNBUFFERED=1
    export IIIF_API_ROOT=https://acc.images.data.amsterdam.nl/iiif/2/edepot:
    export IIIF_CACHE_DIR=./cache
    export MODEL_DIR=../output/model/
    export OUTPUT_DIR=../output/prediction/

After this you can run the following from the `src/` folder:

    ./run_prediction.py ../input/small_set.csv

After this has finished, the result can be found in `output/prediction/`

## Domain model
The stadsarchief datasets consistist of scans along with metadata.
The folder structure of the examples is such that it is easy to work with.
This is different from the stadsarchief document hierarchy.

Stadsarchief scans are grouped into "documents", "subdossiers" and "dossiers".
We are mostly concerned with with single scans or "bestanden".
But of course, for the purpose of building a dataset and in discussing requirements the stadsarchief domain model is of importance.   

The domain model is as follows:

![Stadsarchief dataset domain model UML diagram](doc/domain_model/stadsarchief-dataset-domain-model.png?raw=true "Stadsarchief domain model")

An example of a single dossier is:

![Stadsarchief dataset example diagram](doc/domain_model/stadsarchief-dataset-domain-model-example.png?raw=true "Stadsarchief example")

Note that:
 * both a "dossier" and a "document" can be private/public.
 * "Subdossier" is often also referred to as "map"/"folder", as in, this "document" belongs to this "folder".
 * Different "dossiers" contain the same "Subdossier" titels. In essence a "subdossier" acts as a tag.
 The grouping of "documents inside "subdossiers" inside "dossier" reflects the real world paper inside of folders inside of boxes.
 But, conceptually, this following would be a better model (NOT THE ACTUAL MODEL!):
![Stadsarchief dataset domain model UML diagram based on tagged documents](doc/domain_model/stadsarchief-dataset-domain-model-with-tag.png?raw=true "Stadsarchief domain model based on tagged documents")


## Model persistence
The full model is stored in multiple files within the `output/model/` folder:
* model/model.json, Keras model architecture
* model/weights.h5, Keras model weights
* model/transform/image_encoder.npy, image pre processing encoder configuration
* model/transform/target_encoder.npy, label encoder configuration 


## TODO:
- automate storing prediction-models in data store
- automate training 
- automate testing
- Set up an endpoint to be able to automate using the various trained models
