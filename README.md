
Automatic recognition, classification and OCR of stadsarchief archive images.

# Development

`./start.sh`

Open jupyter as on the web site listed in the terminal and open `TrainClassifier.ipynb` in a browser.



# Labeling

```
./Yolo_mark/yolo_mark \
	examples/0-src/architect_set_object_recognition/full/ \
	examples/0-src/architect_set_object_recognition/train.txt \
	examples/0-src/architect_set_object_recognition/obj.names
```
