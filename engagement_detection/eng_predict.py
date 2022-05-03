import threading
import os, time
import shutil
import numpy as np
import random
import tensorflow as tf
import copy
import time

# from engagement_detection import md_config as cfg
import md_config as cfg
# from engagement_detection.feature_collection import FeatureCollection
from feature_collection import FeatureCollection


from tensorflow.keras.models import Sequential
import keras
from keras.layers import CuDNNLSTM, Dense, TimeDistributed, GlobalAveragePooling1D

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
keras.backend.set_session(session)


def define_model(hparams, model_name):
	current_n_lstms = hparams['NUM_LSTM_LAYERS']
	current_lstm_units = hparams['LSTM_UNITS']
	current_n_denses = hparams['NUM_DENSE_LAYERS']
	current_dense_units = hparams['DENSE_UNITS']
	current_dropout_rates = hparams['DROPOUT_RATES']
	current_time_step = hparams['TIME_STEP']
	current_input_units = hparams['INPUT_UNITS']
	current_densen_act = hparams['ACTIVATION_F']

	model = Sequential()
	if hparams['FC1'][1] > 0:
		model.add(TimeDistributed(Dense(hparams['FC1'][1], activation='relu'),
								  input_shape=(current_time_step, hparams['FC1'][0])))

	model.add(
		CuDNNLSTM(current_lstm_units[0], return_sequences=True, input_shape=(current_time_step, current_input_units),
				  stateful=False))

	if current_n_lstms > 1:
		for idx in range(1, current_n_lstms):
			model.add(CuDNNLSTM(current_lstm_units[idx], return_sequences=True))

	for idx in range(current_n_denses):
		model.add(TimeDistributed(Dense(current_dense_units[idx], activation='relu')))

	model.add(TimeDistributed(Dense(1, activation=current_densen_act)))
	model.add(GlobalAveragePooling1D())

	return model

def get_model(model_index, n_segments=15, input_units=60):
    """
    Make prediction for data_npy
    :param data_npy:
    :return:
    """
    ld_cfg = cfg.md_cfg
    hparams = copy.deepcopy(ld_cfg[model_index])
    ft_type = 'of'


    hparams['TIME_STEP'] = n_segments
    hparams['INPUT_UNITS'] = hparams['FC1'][1] if hparams['FC1'][1] > 0 else input_units
    hparams['optimizer'] = 'adam'
    hparams['ACTIVATION_F'] = 'tanh'
    hparams['CLSW'] = 1

    cur_model = define_model(hparams,hparams['NAME'])
    cur_model.build()
    cur_model.load_weights(
            './engagement_detection/models/{}_{}_models_{}_{}_0_epochs{}_best_weight.h5'.format(hparams['model_path'], ft_type,
                                                                           hparams['n_segments'], hparams['alpha'],
                                                                           hparams['EPOCHS']))

    return cur_model

PROCESSED_DIR = "D:/Downloads/OpenFace-OpenFace_2.2.0/OpenFace-OpenFace_2.2.0/processed"
def calc_scores():
	if os.path.isdir(PROCESSED_DIR):
		v1 = []
		v2 = []
		feature_extraction = FeatureCollection(PROCESSED_DIR)
		ft = np.array(feature_extraction.get_all_data())
		# print(ft.shape)
		with session1.as_default():
			with graph1.as_default():
				for feature in ft:
					v1.append(eye_gaze_v1.predict(feature.reshape(1,15,60)))
		with session2.as_default():
			with graph2.as_default():
				for feature in ft:
					v2.append(eye_gaze_v2.predict(feature.reshape(1,15,60)))

		# print('{} {}'.format(v1,v2))
		v1 = np.array(v1)
		v2 = np.array(v2)
		scores = np.average([v1, v2], axis=0)
		# enga_score = 0.5 * (v1 + v2)
		# print('engagement_score = {}'.format(enga_score))
		# shutil.rmtree(PROCESSED_DIR, ignore_errors=True)
		# print(v2)
		# print(v1)
		print(scores)
		return scores
	else:
		print("Input not found!!")

def predict():
	global graph1, graph2, eye_gaze_v1, eye_gaze_v2
	graph1 = tf.Graph()
	with graph1.as_default():
		global session1
		session1 = tf.compat.v1.Session()
		with session1.as_default():
			eye_gaze_v1 = get_model(model_index=0)
	graph2 = tf.Graph()
	with graph2.as_default():
		global session2
		session2 = tf.compat.v1.Session()
		with session2.as_default():
			eye_gaze_v2 = get_model(model_index=1)
	
	return calc_scores()

predict()