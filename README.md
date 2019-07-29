
Classification of bouwdossiers

# Development

`./start.sh`

Open jupyter as on the web site listed in the terminal and open `TrainClassifier.ipynb` in a browser.

# Domain model

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
