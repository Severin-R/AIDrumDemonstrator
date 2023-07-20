import os
import torch
import numpy as np
import random
import glob
import librosa
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

root_dir = os.getcwd()

os.chdir(os.path.join(root_dir, "neural_granular_synthesis-master", "codes"))
from models import hierarchical_model

original_dist = {'Clap': 423 / 10274, 'Combo': 83 / 10274, 'Cymbal': 767 / 10274, 'HandDrum': 142 / 10274,
                 'HiHat': 1933 / 10274, 'Kick': 1594 / 10274, 'MalletDrum': 644 / 10274, 'Metallic': 643 / 10274,
                 'Shaker': 377 / 10274, 'Snare': 2596 / 10274, 'Tom': 918 / 10274, 'Wooden': 154 / 10274}


class Granular_Synthesis_Model:
    def __init__(self):
        self.model = self._load_grain_model()
        self.name = "Neural_Granular_Sound_Synthesis"

    def _load_grain_model(self):
        waveform_mname = "fulldrums_16k_v1"
        latent_mname = waveform_mname + "__" + "embedding_cond_l_E256_1LSTM"
        mname = latent_mname + "__finetuned"
        model_dir = os.path.join(root_dir, "neural_granular_synthesis-master", "codes", "outputs")

        w_ckpt_file = sorted(glob.glob(os.path.join(model_dir, waveform_mname, "checkpoints", "*.ckpt")))[-1]
        w_yaml_file = os.path.join(model_dir, waveform_mname, "hparams.yaml")
        l_ckpt_file = sorted(glob.glob(os.path.join(model_dir, latent_mname, "checkpoints", "*.ckpt")))[-1]
        l_yaml_file = os.path.join(model_dir, latent_mname, "hparams.yaml")

        ckpt_file = sorted(glob.glob(os.path.join(model_dir, mname, "checkpoints", "*.ckpt")))[-1]
        yaml_file = os.path.join(model_dir, mname, "hparams.yaml")
        model = hierarchical_model.load_from_checkpoint(checkpoint_path=ckpt_file, hparams_file=yaml_file,
                                                        map_location='cpu',
                                                        w_ckpt_file=w_ckpt_file, w_yaml_file=w_yaml_file,
                                                        l_ckpt_file=l_ckpt_file, l_yaml_file=l_yaml_file)
        model.eval()
        return model

    def sample(self, cl_idx, temperature=1.):
        with torch.no_grad():
            rand_e = torch.randn((1, self.model.l_model.hparams.e_dim))
            rand_e = rand_e * temperature
            conds = torch.zeros(1).long() + cl_idx
            audio = self.model.decode(rand_e, conds)[0].view(-1).numpy()
        return audio[:16000]

    def generate_n_samples(self, n_samples=120):
        samples = []
        classes = list(original_dist.keys())
        probabilities = list(original_dist.values())
        class_dist = np.random.choice(classes, size=n_samples, p=probabilities)
        for sound_class in class_dist:
            samples.append(self.sample(classes.index(sound_class)))
        return np.array(samples)

    def generate_n_samples_per_class(self, n_samples=1):
        samples = []
        n_classes = len(self.model.l_model.hparams.classes)
        for sound_class in range(n_classes):
            for i in range(n_samples):
                samples.append(self.sample(sound_class))
        return np.array(samples)


os.chdir(os.path.join(root_dir, "CRASH-master"))
from model import UNet
from inference import SDESampling3
from sde import VpSdeCos


class Crash_Model:
    def __init__(self):
        self.model = self._load_model()
        self.sde = VpSdeCos()
        self.sampler = SDESampling3(self.model, self.sde)
        self.classifier = tf.keras.models.load_model(os.path.join(root_dir, "MFCC_Classifier.h5"))
        self.name = "CRASH"

    def _load_ema_weights(self, model, model_dir):
        checkpoint = torch.load(model_dir)
        dic_ema = {}
        for (key, tensor) in zip(checkpoint['model'].keys(), checkpoint['ema_weights']):
            dic_ema[key] = tensor
        model.load_state_dict(dic_ema)
        return model

    def _load_model(self):
        model_dir = os.path.join(root_dir, 'CRASH-master', 'weights', 'weights-143370.pt')
        self.device = torch.device('cuda:0')
        model = UNet().to(self.device)
        model = self._load_ema_weights(model, model_dir)
        return model

    def generate_n_samples(self, n_samples=120):
        noise = torch.randn(n_samples, 15000, device=self.device)
        drums = self.sampler.predict(noise, 100)
        drums = drums.cpu()
        return np.pad(drums, ((0, 0), (0, 1000)))

    def sample(self, cl_idx=0):
        if cl_idx < 0:
            return self.generate_n_samples(1)
        pred_class = -1
        max_count = 0
        # drum_sounds = self.generate_n_samples(10)
        n_generations = 1

        while max_count * n_generations < 250:
            if max_count % 10 == 0:
                drum_sounds = self.generate_n_samples(10)
                mfcc_drum = np.transpose(librosa.feature.mfcc(y=drum_sounds, sr=16000, n_mfcc=40), (0, 2, 1))
                pred_dist = self.classifier.predict(mfcc_drum)
                pred_classes = np.argmax(pred_dist, axis=1)
                max_count = 0
                n_generations += 1
            pred_class = pred_classes[max_count]
            if cl_idx == pred_class:
                return drum_sounds[max_count]
            max_count += 1

        return drum_sounds[max_count-1]

    def generate_n_samples_per_class(self, n_samples=1):
        num_classes = 12
        samples = np.zeros((num_classes, n_samples, 16000))
        random_samples = self.generate_n_samples(n_samples * num_classes * 10)
        mfcc_random = np.transpose(librosa.feature.mfcc(y=random_samples, sr=16000, n_mfcc=40), (0, 2, 1))
        pred_dist = self.classifier.predict(mfcc_random)
        pred_classes = np.argmax(pred_dist, axis=1)
        max_count = 0

        # somehow not too unefficiently fill up array
        while True:
            pred_class = pred_classes[max_count]
            if 0 in samples[pred_class]:
                idx = np.where(samples[pred_class] == 0)[0]
                samples[pred_class][idx] = random_samples[max_count]
            max_count += 1
            if max_count >= random_samples.shape[0] or not np.any(samples == 0):
                break

        # check for leftover classes that are not filled yet
        if np.any(samples == 0):
            for missing_class in range(num_classes):
                if 0 in samples[missing_class]:
                    idx = np.where(samples[missing_class] == 0)[0]
                    samples[missing_class][idx] = self.sample(missing_class)

        return samples


os.chdir(os.path.join(root_dir, "wavegan-master", "train"))


class WaveGAN_Model:
    def __init__(self):
        self._load_model()
        self.classifier = tf.keras.models.load_model(os.path.join(root_dir, "MFCC_Classifier.h5"))
        self.name = "WaveGAN"

    def _load_model(self):
        tf.reset_default_graph()
        saver = tf.train.import_meta_graph(os.path.join(root_dir, "wavegan-master", "train", 'infer', 'infer.meta'))
        self.graph = tf.get_default_graph()
        self.sess = tf.InteractiveSession()
        saver.restore(self.sess, os.path.join(root_dir, "wavegan-master", "train", 'model.ckpt-7373'))

    def sample(self, cl_idx=0, conditioned=False):
        if cl_idx < 0 or conditioned:
            return self.generate_n_samples(1)
        pred_class = -1
        max_count = 0

        while cl_idx != pred_class and max_count < 250:
            drum_sound = self.generate_n_samples(1)
            mfcc_drum = np.transpose(librosa.feature.mfcc(y=drum_sound, sr=16000, n_mfcc=40), (0, 2, 1))
            pred_dist = self.classifier.predict(mfcc_drum)
            pred_class = np.argmax(pred_dist, axis=1)
            max_count += 1

        return drum_sound[0, :]

    def generate_n_samples(self, n_samples=120):
        _z = (np.random.rand(n_samples, 100) * 2.) - 1

        # Synthesize G(z)
        z = self.graph.get_tensor_by_name('z:0')
        G_z = self.graph.get_tensor_by_name('G_z:0')
        _G_z = self.sess.run(G_z, {z: _z})

        return np.array(_G_z[:, :16000, 0])

    def generate_n_samples_per_class(self, n_samples=1):
        num_classes = 12
        samples = np.zeros((num_classes, n_samples, 16000))
        random_samples = self.generate_n_samples(n_samples * num_classes * 10)
        mfcc_random = np.transpose(librosa.feature.mfcc(y=random_samples, sr=16000, n_mfcc=40), (0, 2, 1))
        pred_dist = self.classifier.predict(mfcc_random)
        pred_classes = np.argmax(pred_dist, axis=1)
        max_count = 0

        # somehow not too unefficiently fill up array
        while True:
            pred_class = pred_classes[max_count]
            if 0 in samples[pred_class]:
                idx = np.where(samples[pred_class] == 0)[0]
                samples[pred_class][idx] = random_samples[max_count]
            max_count += 1
            if max_count >= random_samples.shape[0] or not np.any(samples == 0):
                break

        # check for leftover classes that are not filled yet
        if np.any(samples == 0):
            for missing_class in range(num_classes):
                if 0 in samples[missing_class]:
                    idx = np.where(samples[missing_class] == 0)[0]
                    samples[missing_class][idx] = self.sample(missing_class)

        return samples
