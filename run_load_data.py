from examples.aanvraag_besluit.load_data import load_data_aanvraag

img_dim = (200, 200, 3);
img_dim = (250, 250, 3);
# img_dim = (300, 300, 3);
# img_dim = (400, 400, 3);

[Xtrain_raw, Ytrain_raw, Xvalid_raw, Yvalid_raw] = load_data_aanvraag(
    {
        'images': f'examples/aanvraag_besluit/eerste_dataset/resized/{img_dim[0]}x{img_dim[1]}/',
        'labels': 'examples/aanvraag_besluit/eerste_dataset/labels/'
    },
    {
        'images': f'examples/aanvraag_besluit/tweede_dataset/images/{img_dim[0]}x{img_dim[1]}/',
        'labels': 'examples/aanvraag_besluit/tweede_dataset/labels/'
    },
)

print(f"shape Xtrain[0]: {Xtrain_raw[0].shape}")
print(f"shape Xtrain[1]: {Xtrain_raw[1].shape}")
print(f"shape Ytrain: {Ytrain_raw.shape}")

print(f"shape Xvalid[0]: {Xvalid_raw[0].shape}")
print(f"shape Xvalid[1]: {Xvalid_raw[1].shape}")
print(f"shape Yvalid: {Yvalid_raw.shape}")
