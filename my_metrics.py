import numpy as np
from scipy import stats
import soundfile as sf
import scipy as sp
from sklearn import cluster
import librosa
import os
import tensorflow as tf
from frechet_audio_distance import FrechetAudioDistance
from tqdm import tqdm


def load_audio_data(file_path):
    all_files = []
    for file in os.listdir(file_path):
        y, sr = librosa.load(os.path.join(file_path, file), sr=16000)
        desired_length = 1 * sr
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)

        current_length = y.shape[0]
        if current_length > desired_length:
            y = y[:desired_length]
        elif current_length < desired_length:
            y = np.pad(y, pad_width=(0, (desired_length - current_length)))

        all_files.append(y)

    return np.array(all_files)


def write_audio_files(array, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    print(f"Writing audiofiles to: {output_directory}")
    with tqdm(total=len(array)) as pbar:
        for i, audio in enumerate(array):
            filename = f"{output_directory}/audio_{i}.wav"
            sf.write(filename, audio, 16000)
            pbar.update(1)
            #print(f"Audio {i} geschrieben: {filename}")


class FAD:
    def __init__(self):
        self.frechet = FrechetAudioDistance(model_name="vggish",
                                            use_pca=False,
                                            use_activation=False,
                                            verbose=False
                                            )

    def calculate_from_path(self, original_data_path: str, generated_data_path: str):
        return self.frechet.score(original_data_path, generated_data_path)

    def calculate(self, original_data_path: str, generated_data, outdir):
        write_audio_files(generated_data, outdir)
        return self.frechet.score(original_data_path, outdir)


class NDB:
    def __init__(self):
        self.name ="Number of Statistically Different Bins"

    def calculate_from_path(self, original_data_path: str, generated_data_path: str, n_bins=50):
        original_data = load_audio_data(original_data_path)
        generated_data = load_audio_data(generated_data_path)
        return self._num_different_bins(original_data, generated_data, n_bins)

    def calculate(self, original_data, generated_data, n_bins=50):
        return self._num_different_bins(original_data, generated_data, n_bins)

    def _num_different_bins(self, real_features, fake_features, num_bins=50, significance_level=0.05):
        clusters = cluster.KMeans(n_clusters=num_bins).fit(real_features)
        real_labels, real_counts = np.unique(clusters.labels_, return_counts=True)
        real_proportions = real_counts / np.sum(real_counts)

        labels = np.array([
            np.argmin(np.sum((fake_feature - clusters.cluster_centers_) ** 2, axis=1))
            for fake_feature in fake_features
        ])
        fake_labels, fake_counts = np.unique(labels, return_counts=True)
        fake_proportions = np.zeros_like(real_proportions)
        fake_proportions[fake_labels] = fake_counts / np.sum(fake_counts)

        different_bins = self._binomial_proportion_test(
            p=real_proportions,
            m=len(real_features),
            q=fake_proportions,
            n=len(fake_features),
            significance_level=significance_level
        )
        return np.count_nonzero(different_bins)

    def _binomial_proportion_test(self, p, m, q, n, significance_level):
        p = (p * m + q * n) / (m + n)
        se = np.sqrt(p * (1 - p) * (1 / m + 1 / n))
        z = (p - q) / se
        p_values = sp.stats.norm.cdf(-np.abs(z)) * 2
        return p_values < significance_level


class IS:
    def __init__(self):
        self.classifier = tf.keras.models.load_model("MFCC_Classifier.h5")

    def calculate_from_path(self, original_data_path: str, generated_data_path: str):
        generated_data = load_audio_data(generated_data_path)
        generated_data = self._transform_sounds_to_mfcc(generated_data)
        predictions = self.classifier.predict(generated_data)
        return self._inception_score(predictions)

    def calculate(self, original_data, generated_data):
        generated_data = self._transform_sounds_to_mfcc(generated_data)
        predictions = self.classifier.predict(generated_data)
        return self._inception_score(predictions)

    def _transform_sounds_to_mfcc(self, generated_sounds):
        mfcc_features = 40
        mfccs = []
        desired_l = 16000
        for sound in generated_sounds:
            # cutting / padding sound to desired length
            current_l = len(sound)
            if current_l > desired_l:
                sound = sound[:desired_l]
            if current_l < desired_l:
                sound = np.pad(sound, pad_width=(0, (desired_l - current_l)))

            mfcc = librosa.feature.mfcc(y=sound, n_mfcc=mfcc_features)
            mfccs.append(mfcc.T)
        return np.array(mfccs)

    def _inception_score(self, preds, eps=1E-16):
        preds = preds / np.sum(preds, axis=-1, keepdims=True)

        marginal = np.mean(preds, axis=0)

        kl = preds * (np.log(preds + eps) - np.log(marginal + eps))
        kl = np.mean(np.sum(kl, axis=1))

        is_score = np.exp(kl)

        return is_score
