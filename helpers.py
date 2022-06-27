import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

CLASSES = ["Inanimate", "Animate"]
THRESHOLD = 0.45
IMAGE_INPUT_SIZE = 224


def get_output_model(image, ort_session):
    input_onnx = ort_session.get_inputs()[0].name
    outputs = ort_session.run(
        None,
        {input_onnx: image},
    )
    return outputs


def preprocessing_image(image):
    image_augmentation = A.Compose([A.Normalize(), ToTensorV2()])
    image = image_augmentation(image=np.array(image))["image"]
    image = np.expand_dims(image, axis=0)
    return image


def postprocessing_image(output):
    return output[0][0][0]


def predict_image(image, ort_session):
    image = np.array(image.resize((IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE)).convert("RGB"))
    input_image = preprocessing_image(image)
    predict = get_output_model(input_image, ort_session)
    predicted_score = postprocessing_image(predict)
    predicted_classes = CLASSES[int(predicted_score > THRESHOLD)]
    return predicted_classes, predicted_score
