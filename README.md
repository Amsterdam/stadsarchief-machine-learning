
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
