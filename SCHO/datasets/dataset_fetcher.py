from sklearn import datasets as sklearn_datasets
from tensorflow.keras import datasets as keras_datasets
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
from DataHandling import Filing
from nlp_helper import NLPEncoder


# TODO: add normalizer

class ToyDatasets:
    @staticmethod
    def get_toy_dataset(dataset_name, return_pre_split_tuple=False):
        if dataset_name == "iris":
            toy_dataset = sklearn_datasets.load_iris()
            X = toy_dataset.data
            Y = toy_dataset.target

        elif dataset_name == "cup":
            toy_dataset = sklearn_datasets.fetch_kddcup99(percent10=False, subset="SA", random_state=10)
            X = toy_dataset.data
            Y = toy_dataset.target
            Y = np.where(pd.Series(Y).astype(str).str.contains("normal"), "normal", "attack")
            Y = pd.factorize(Y)[0]
            XY = pd.DataFrame(np.hstack([X, Y.reshape(len(Y), 1)])).drop_duplicates()
            XY = ToyDatasets._binary_class_rebalancing(data=XY, y_name=(XY.shape[1] - 1))
            X = XY.drop([(XY.shape[1] - 1)], axis=1).to_numpy()
            X = PreProcessingHelper.one_hot_encode_variables(X)
            Y = XY[XY.shape[1] - 1].to_numpy().astype(int)

        elif dataset_name == "diabetes":
            toy_dataset = sklearn_datasets.load_diabetes()
            X = toy_dataset.data
            Y = toy_dataset.target

        elif dataset_name == "cali":
            toy_dataset = sklearn_datasets.fetch_california_housing()
            X = toy_dataset.data
            Y = toy_dataset.target

        elif dataset_name == "friedman1":
            X, Y = sklearn_datasets.make_friedman1(n_samples=20000, n_features=15, random_state=10)

        elif dataset_name == "census":
            raw_data = pd.read_csv("datasets_offline/census/census.data", sep=',')
            raw_data_formatted = pd.DataFrame()
            for l in range(0, raw_data.shape[1]):
                try:
                    raw_data_formatted = pd.concat([raw_data_formatted, raw_data.iloc[:, l].astype(float)], axis=1)
                except:
                    raw_data_formatted = pd.concat([raw_data_formatted, pd.get_dummies(raw_data.iloc[:, l])], axis=1)
            raw_data_formatted = raw_data_formatted.drop([" <=50K"], axis=1).reset_index(drop=True)

            raw_data_formatted = ToyDatasets._binary_class_rebalancing(data=raw_data_formatted, y_name=" >50K")
            X = (raw_data_formatted.drop([" >50K"], axis=1)).to_numpy()
            Y = raw_data_formatted[" >50K"].to_numpy()
            Y = Y.astype('int')

        elif dataset_name == "cancer" or dataset_name == "tabular_test_data":
            toy_dataset = sklearn_datasets.load_breast_cancer()
            X = toy_dataset.data
            Y = toy_dataset.target

        elif dataset_name == "digits":
            toy_dataset = sklearn_datasets.load_digits()
            X = toy_dataset.data / 16
            Y = toy_dataset.target

        elif dataset_name == "20news":
            toy_dataset = sklearn_datasets.fetch_20newsgroups()
            X = np.array(toy_dataset.data)
            Y = toy_dataset.target

            undersampling_index = list(np.random.choice(len(X), 2000, replace=False))
            X = X[undersampling_index]
            Y = Y[undersampling_index]

        elif dataset_name == "covertype":
            raw_data = pd.read_csv(Filing.parent_folder_path + "/datasets/covertype/covtype.data", sep=',')

            X = raw_data.iloc[:, :-1].to_numpy()
            Y = raw_data.iloc[:, -1].to_numpy()

            undersampling_index = list(np.random.choice(len(X), 20, replace=False))
            X = X[undersampling_index, :]
            Y = Y[undersampling_index]
            Y = Y.astype('int')

        elif dataset_name == "olivetti":
            toy_dataset = sklearn_datasets.fetch_olivetti_faces()
            X = toy_dataset.data
            Y = toy_dataset.target

        elif dataset_name == "mnist" or dataset_name == "convolutional_test_data":
            (x_train, y_train), (x_test, y_test) = keras_datasets.mnist.load_data()
            x_train = x_train / 255
            x_test = x_test / 255

            if dataset_name == "convolutional_test_data":
                resample_n = 1000
                undersampling_index = list(np.random.choice(len(x_train), resample_n, replace=False))
                x_train = x_train[undersampling_index, :]
                y_train = y_train[undersampling_index]
                y_train = y_train.astype('int')

                undersampling_index = list(np.random.choice(len(x_train), resample_n, replace=False))
                x_test = x_test[undersampling_index, :]
                y_test = y_test[undersampling_index]
                y_test = y_test.astype('int')

            if return_pre_split_tuple:
                return x_train, y_train, x_test, y_test
            else:
                return x_train, y_train

        elif dataset_name == "cifar10":
            (x_train, y_train), (x_test, y_test) = keras_datasets.cifar10.load_data()
            x_train = x_train / 255
            x_test = x_test / 255
            if return_pre_split_tuple:
                return x_train, y_train, x_test, y_test
            else:
                return x_train, y_train

        elif dataset_name == "colorectal":
            colorectal_raw_data = tfds.load("colorectal_histology", split='train')
            images = []
            labels = []
            # Iterate over a dataset
            for i, image_dict in enumerate(tfds.as_numpy(colorectal_raw_data)):
                images.append(image_dict["image"])
                labels.append(image_dict["label"])
            # images = tf.image.resize(images, [32,32]).numpy()

            X = np.array(images)
            Y = np.array(labels)
            # undersampling_index = list(np.random.choice(len(X), 10000, replace=False))
            # X = X[undersampling_index, :]
            X = X / 255
            # Y = Y[undersampling_index]

            return X, Y

        elif dataset_name == "svhn":
            if return_pre_split_tuple:
                train_svhn_raw_data = tfds.load("svhn_cropped", split='train')
                train_images = []
                train_labels = []
                for i, image_dict in enumerate(tfds.as_numpy(train_svhn_raw_data)):
                    train_images.append(image_dict["image"])
                    train_labels.append(image_dict["label"])
                test_svhn_raw_data = tfds.load("svhn_cropped", split='test')
                test_images = []
                test_labels = []
                for i, image_dict in enumerate(tfds.as_numpy(test_svhn_raw_data)):
                    test_images.append(image_dict["image"])
                    test_labels.append(image_dict["label"])
                return np.array(train_images), np.array(train_labels), np.array(test_images), np.array(test_labels)

            else:
                svhn_raw_data = tfds.load("svhn_cropped", split='train')
                images = []
                labels = []
                for i, image_dict in enumerate(tfds.as_numpy(svhn_raw_data)):
                    images.append(image_dict["image"])
                    labels.append(image_dict["label"])
                # images = tf.image.resize(images, [32,32]).numpy()

                X = np.array(images)
                Y = np.array(labels)
                undersampling_index = list(np.random.choice(len(X), 25000, replace=False))
                X = X[undersampling_index, :]
                X = X / 255
                Y = Y[undersampling_index]

            return X, Y

        elif dataset_name == "stl10":
            stl10_raw_data_train = tfds.load("stl10", split='train')
            stl10_raw_data_test = tfds.load("stl10", split='test')
            train_images = []
            train_labels = []
            for i, image_dict in enumerate(tfds.as_numpy(stl10_raw_data_train)):
                train_images.append(image_dict["image"])
                train_labels.append(image_dict["label"])
            test_images = []
            test_labels = []
            for i, image_dict in enumerate(tfds.as_numpy(stl10_raw_data_test)):
                test_images.append(image_dict["image"])
                test_labels.append(image_dict["label"])
            X = np.array(train_images + test_images)
            Y = np.array(train_labels + test_labels)
            # undersampling_index = list(np.random.choice(len(X), 20000, replace=False))
            # X = X[undersampling_index, :]
            X = X / 255
            # Y = Y[undersampling_index]

            return X, Y

        elif dataset_name == "cifar100":
            (x_train, y_train), (x_test, y_test) = keras_datasets.cifar100.load_data()
            undersampling_index = list(np.random.choice(len(x_train), 20000, replace=False))
            x_train = x_train[undersampling_index, :]
            x_train = x_train / 255
            y_train = y_train[undersampling_index]

            return x_train, y_train

        elif dataset_name == "fashion_mnist":
            (x_train, y_train), (x_test, y_test) = keras_datasets.fashion_mnist.load_data()
            x_train = x_train / 255
            x_test = x_test / 255
            # y_train = y_train[undersampling_index]
            if return_pre_split_tuple:
                return x_train, y_train, x_test, y_test
            else:
                return x_train, y_train

        elif dataset_name == "imdb":
            raw_data = pd.read_csv(Filing.input_parent_folder_path + "datasets_offline/imdb/imdb_dataset.csv")
            X = raw_data["review"].to_numpy()
            Y = raw_data["sentiment"].to_numpy()
            Y[Y == "negative"] = int(0)
            Y[Y == "positive"] = int(1)
            Y = Y.astype('int')

            # # TODO: TOGGLE:
            # undersampling_index = list(np.random.choice(len(X), 500, replace=False))
            # X = X[undersampling_index]
            # Y = Y[undersampling_index]

            if return_pre_split_tuple:

                np.random.seed(1234)
                OOS_index = list(
                    np.random.choice(len(X), round(len(X) * 0.2), replace=False))  # TODO: hard coded 20% OOS proportion
                IS_index = list(np.setdiff1d(np.arange(len(X)), OOS_index))
                try:
                    x_test = X[OOS_index]
                    y_test = Y[OOS_index]
                    x_train = X[IS_index]
                    y_train = Y[IS_index]
                except:
                    print("Value Error: X data must be a single column containing full string sentences per row.")

                tfidf_vectorizer, tfidf_object = NLPEncoder.fit_BOW_vectorizer(X=x_train)
                x_train = NLPEncoder.apply_BOW_vectorizer(vectorizer=tfidf_vectorizer, tfidf_object=tfidf_object,
                                                          X=x_train)
                x_test = NLPEncoder.apply_BOW_vectorizer(vectorizer=tfidf_vectorizer, tfidf_object=tfidf_object,
                                                         X=x_test)

                return x_train, y_train, x_test, y_test

            else:
                tfidf_vectorizer, tfidf_object = NLPEncoder.fit_BOW_vectorizer(X=X)
                X = NLPEncoder.apply_BOW_vectorizer(vectorizer=tfidf_vectorizer, tfidf_object=tfidf_object, X=X)

        return X, Y

    @staticmethod
    def _binary_class_rebalancing(data, y_name, type="majority_undersampling"):
        if type == "majority_undersampling":
            majority_class = data[y_name].mode()[0]
            minority_class_filtered_dataset = data[data[y_name] != majority_class]
            majority_class_filtered_dataset = data[data[y_name] == majority_class]
            majority_class_filtered_dataset = majority_class_filtered_dataset.sample(
                n=len(minority_class_filtered_dataset),
                replace=True, random_state=1234)
            rebalanced_data = pd.concat([minority_class_filtered_dataset, majority_class_filtered_dataset], axis=0)
        elif type == "minority_oversampling":
            majority_class = data[y_name].mode()[0]
            minority_class_filtered_dataset = data[data[y_name] != majority_class]
            majority_class_filtered_dataset = data[data[y_name] == majority_class]
            minority_class_filtered_dataset = minority_class_filtered_dataset.sample(
                n=len(majority_class_filtered_dataset),
                replace=True, random_state=1234)
            rebalanced_data = pd.concat([minority_class_filtered_dataset, majority_class_filtered_dataset], axis=0)

        return rebalanced_data

    @staticmethod
    def _multi_label_class_rebalancing(data, y_name, type="majority_undersampling"):
        if type == "majority_undersampling":
            lowest_represented_class = data[y_name].value_counts().index[-1]
            lowest_represented_class_filtered_dataset = data[data[y_name] == lowest_represented_class]
            rebalanced_data = lowest_represented_class_filtered_dataset.copy()
            for label in data[y_name].unique():
                if label != lowest_represented_class:
                    filtered_dataset = data[data[y_name] == label]
                    resampled_filtered_dataset = filtered_dataset.sample(
                        n=len(lowest_represented_class_filtered_dataset),
                        replace=True, random_state=1234)
                    rebalanced_data = pd.concat([rebalanced_data, resampled_filtered_dataset], axis=0)

        return rebalanced_data


class PreProcessingHelper:
    @staticmethod
    def one_hot_encode_variables(X):
        X_encoded = X.copy()
        X_final = np.zeros(len(X_encoded)).reshape(len(X_encoded), 1)
        for i in range(0, X_encoded.shape[1]):
            try:
                X_encoded[:, i].astype(float)
                X_final = np.hstack([X_final, X_encoded[:, i].reshape(len(X_encoded[:, i]), 1)])
            except:
                X_append = pd.get_dummies(pd.Series(X_encoded[:, i])).to_numpy()
                X_final = np.hstack([X_final, X_append])
        X_final = X_final[:, 1:]
        return X_final
