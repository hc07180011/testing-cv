import os
import cv2
import glob
import argparse
import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries
from utils.utils import calc_mean_score, save_json, load_image
from utils.losses import earth_movers_distance
from handlers.model_builder import Nima
from handlers.data_generator import TestDataGenerator


def image_file_to_json(img_path):
    img_dir = os.path.dirname(img_path)
    img_id = os.path.basename(img_path).split('.')[0]

    return img_dir, [{'image_id': img_id}]


def image_dir_to_json(img_dir, img_type='jpg'):
    img_paths = glob.glob(os.path.join(img_dir, '*.'+img_type))

    samples = []
    for img_path in img_paths:
        img_id = os.path.basename(img_path).split('.')[0]
        samples.append({'image_id': img_id})

    return samples


def predict(model, data_generator):
    return model.predict_generator(data_generator, workers=8, use_multiprocessing=True, verbose=1)


def gaussian_noise(img, variance):
    mean = 0
    sigma = variance ** 0.5
    row, col, ch = img.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)

    image_source = os.path.join("..", "tests", "test_images",
                                "gaussian", "{}.jpg".format(variance))
    cv2.imwrite(image_source, (img + gauss))

    return image_source


def contrast_brightness(img, alpha, beta):

    img = np.clip(alpha * img + beta, 0, 255)

    image_source = os.path.join("..", "tests", "test_images",
                                "brightness", "{}.jpg".format(int(alpha * 10)))
    cv2.imwrite(image_source, img)

    return image_source


def main(base_model_name, weights_file, image_source, predictions_file, img_format='jpg'):
    nima = Nima(base_model_name, weights=None)
    nima.build()
    nima.nima_model.load_weights(weights_file)

    # for i in range(30):

    #     img = load_image(image_source, target_size=(224, 224))

    #     if i:
    #         img = np.vstack((np.ones((i, 224, 3)).astype(int) * 255, img[:-i]))

    #     explainer = lime_image.LimeImageExplainer()
    #     explaination = explainer.explain_instance(
    #         image=img, hide_color=None, top_labels=10, classifier_fn=nima.nima_model.predict_on_batch)
    #     for l in explaination.top_labels:
    #         if l == 0:
    #             lime_img, mask = explaination.get_image_and_mask(
    #                 label=explaination.top_labels[0], num_features=10, hide_rest=True)
    #             lime_img = mark_boundaries(lime_img, mask)
    #             break

    #     cv2.imwrite(os.path.join("test", "{}.png".format(i)), cv2.cvtColor(
    #         np.float32(lime_img), cv2.COLOR_BGR2RGB))
    # exit()

    # load samples
    if os.path.isfile(image_source):
        image_dir, samples = image_file_to_json(image_source)
    else:
        image_dir = image_source
        samples = image_dir_to_json(image_dir, img_type='jpg')

    # build model and load weights
    nima = Nima(base_model_name, weights=None)
    nima.build()
    nima.nima_model.load_weights(weights_file)

    # initialize data generator
    data_generator = TestDataGenerator(samples, image_dir, 64, 10, nima.preprocessing_function(),
                                       img_format=img_format)

    # get predictions
    predictions = predict(nima.nima_model, data_generator)

    # calc mean scores and add to samples
    for i, sample in enumerate(samples):
        sample['mean_score_prediction'] = calc_mean_score(predictions[i])

    # import matplotlib.pyplot as plt

    # distibution = np.array(predictions[0])

    # plt.bar(np.arange(1, 11, 1), distibution)
    # plt.savefig("tmp.png")

    # print(json.dumps(samples, indent=2))

    if predictions_file is not None:
        save_json(samples, predictions_file)

    print(np.sum(np.arange(1, 11, 1) * np.array(predictions[0])))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base-model-name',
                        help='CNN base model name', default="MobileNet")
    parser.add_argument('-w', '--weights-file',
                        help='path of weights file', default="models/weights_mobilenet_technical_0.11.hdf5")
    parser.add_argument('-is', '--image-source',
                        help='image directory or file', required=True)
    parser.add_argument('-pf', '--predictions-file',
                        help='file with predictions', required=False, default=None)

    args = parser.parse_args()

    main(**args.__dict__)
