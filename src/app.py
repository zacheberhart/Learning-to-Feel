# -*- coding: utf-8 -*-
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import urllib, requests, spotipy, os, math, shutil, gc
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from spotipy.oauth2 import SpotifyClientCredentials
from copy import deepcopy
from io import BytesIO
from PIL import Image, ImageFile
from fastai.vision import *
from fastai.callbacks import *
from utils import *
from spectrograms import *
from loss import *
from mxresnet import *




##################################################################################
#                                                                                #
# Main                                                                           #
# ::: Handles the navigation / routing and data loading / caching.               #
#                                                                                #
##################################################################################


def main():
	'''Set main() function. Includes sidebar navigation and respective routing.'''

	st.sidebar.title("Explore")
	app_mode = st.sidebar.selectbox( "Choose an Action", [
		"About",
		"Choose an Emotion",
		"Choose an Artist",
		"Classify a Song",
		"Emotional Spectrum",
		"Show Source Code"
	])

	# clear tmp
	clear_tmp()

	# nav
	if   app_mode == "About":            show_about()
	elif app_mode == "Choose an Emotion": explore_classified()
	elif app_mode == 'Choose an Artist':  explore_artists()
	elif app_mode == "Classify a Song":  classify_song()
	elif app_mode == "Emotional Spectrum":  display_emotional_spectrum()
	elif app_mode == "Show Source Code": st.code(get_file_content_as_string("app.py"))


@st.cache
def load_data():
	''' Load main data source with all labels and values. '''
	return read_pkl(path('data/final_scores_meta.pkl'))


@st.cache
def load_tsne():
	''' Load TSNE data for plotting / viz '''
	return read_pkl(path('data/tsne.pkl'))


def clear_tmp():
	''' Clear /tmp on load. Used for new song classification. '''
	shutil.rmtree(path('tmp'))
	for d in [path('tmp'), path('tmp/png'), path('tmp/wav')]: os.mkdir(d)


def path(orig_path):
	''' Path handler for local or production '''
	if len(ROOT_DIR) == 0: return orig_path
	return f'{ROOT_DIR}/{orig_path}'


@st.cache(show_spinner = False)
def get_file_content_as_string(path):
	''' Download a single file and make its content available as a string. '''
	url = 'https://raw.githubusercontent.com/streamlit/demo-self-driving/master/' + path
	response = urllib.request.urlopen(url)
	return response.read().decode("utf-8")


@st.cache(show_spinner = False)
def read_text(fname):
	''' Display copy from a .txt file. '''
	with open(fname, 'r') as f:
		text = f.readlines()
	return text


def show_about():
	''' Home / About page '''
	st.title('Learning to Listen, to Feel')
	for line in read_text(path('about.txt')):
		st.write(line)




##################################################################################
#                                                                                #
# Choose an Emotion                                                              #
# ::: Allow the user to pick one or more labels to get a list of the top songs   #
# ::: classified with the respective label(s). Limit the list of songs returned  #
# ::: to 100, but allow the user to choose the quantity and the "Popularity", a  #
# ::: a metric provided by Spotify's API. Also allow the user to leave the app   #
# ::: and listen to the song on Spotify's Web App using the provided link.       #
#                                                                                #
##################################################################################


def explore_classified():

	# load all data
	df = load_data()
	non_label_cols = ['track_id', 'track_title', 'artist_name', 'track_popularity', 'artist_popularity']
	dims = [c for c in df.columns.tolist() if c not in non_label_cols]

	# Mood or Emotion Selection
	st.title('Explore All Moods & Emotions')
	st.write('''
		Select a mood, an emotion, or a few of each! However, keep in mind that results are best when
		you choose as few as possible -- though you will definitely get some pretty funky results the more you add.
	''')

	# filters
	labels = st.multiselect("Choose:", dims)
	n_songs = st.slider('How many songs?', 1, 100, 20)
	popularity = st.slider('How popular?', 0, 100, (0, 100))

	try:

		# filter data to the labels the user specified
		cols = (non_label_cols, labels)
		df = filter_data(df, cols, n_songs, popularity)

		# show data
		if st.checkbox('Include Preview URLs', value = True):
			df['preview'] = add_stream_url(df.track_id)
			df['preview'] = df['preview'].apply(make_clickable, args = ('Listen',))
			data = df.drop('track_id', 1)
			data = data.to_html(escape = False)
			st.write(data, unsafe_allow_html = True)
		else:
			data = df.drop('track_id', 1)
			st.write(data)

	except: pass

def norm_and_combine(df, labels):
	''' Normalize and log transform scores for better query combination results. '''
	tdf = pd.DataFrame()
	for label in labels:
		tdf[label] = np.log1p(df[label])
		tdf[label] = tdf[label] / tdf[label].max()
	return tdf[labels].sum(1)

def filter_data(df, cols, n_songs, popularity):
	''' Filter the df based on user-selected label, quantity, and popularity selections. '''
	non_label_cols, label_cols = cols
	tdf = deepcopy(df[non_label_cols + label_cols])
	tdf = deepcopy(tdf[(tdf.track_popularity >= popularity[0]) & (tdf.track_popularity <= popularity[1])
					   ].drop(['track_popularity', 'artist_popularity'], 1))
	tdf = deepcopy(tdf.drop_duplicates(['track_title', 'artist_name']))
	if len(label_cols) > 1:
		tdf['combo'] = norm_and_combine(tdf, label_cols)
		for label in label_cols:
			tdf = deepcopy(tdf[tdf[label] >= 0.04])
		return tdf.sort_values('combo', ascending = False)[:n_songs].reset_index(drop = True)
	else:
		return tdf.sort_values(label_cols[0], ascending = False)[:n_songs].reset_index(drop = True)

def add_stream_url(track_ids):
	''' Build Spotify Track URL given its Track ID. '''
	return [f'https://open.spotify.com/track/{t}' for t in track_ids]

def make_clickable(url, hyperlink_text):
	''' Convert URL to clickable HTML link. '''
	return f'<a target="_blank" href="{url}">{hyperlink_text}</a>'




##################################################################################
#                                                                                #
# Choose an Artist                                                               #
# ::: Display the top 10 labels for an artist specified by the user and a        #
# ::: list of tracks that we have for that artist in our db.                     #
#                                                                                #
##################################################################################


def explore_artists():

	# load all data
	df = load_data()
	non_label_cols = ['track_id', 'track_title', 'artist_name', 'track_popularity', 'artist_popularity']
	dims = [c for c in df.columns.tolist() if c not in non_label_cols]

	# user input
	st.title('Explore Artists')
	selected_artist = st.text_input('Search for an Artist:', 'Bon Iver')
	search_results = df[df.artist_name.str.lower() == selected_artist.lower()]

	# display results
	if len(search_results) > 0:
		label_weights = get_top_labels(search_results, dims, 10)
		st.plotly_chart(artist_3d_scatter(label_weights), width = 0)
		search_results['top_label'] = search_results.iloc[:,5:].astype(float).idxmax(1)
		st.write(search_results[['track_title', 'artist_name', 'track_popularity', 'top_label']
			].sort_values(['track_title', 'track_popularity']).drop_duplicates('track_title').reset_index(drop = True))
	else:
		st.write('Sorry, there are no results for that artist in our database :(')


def artist_3d_scatter(label_weights):
	'''Display an artists position in the emotional spectrum.'''
	
	# get data
	label_weights = label_weights.merge(TSNE, on  = 'label')
	tdf = label_weights.rename(columns = {'1d0': 'color', '3d0': 'energy', '3d1': 'style', '3d2': 'acousticness'})

	# build fig
	fig = go.Figure(data = [go.Scatter3d(
		x = tdf['energy'], y = tdf['style'], z = tdf['acousticness'],
		mode = 'markers+text',
		text = tdf['label'], textfont = dict(size = 16),
		marker = dict(
			size = tdf['weight'] * 50,
			color = tdf['color'],
			opacity = 0.6,
			colorscale = 'RdBu',
		)
	)])

	# layout modifications
	fig.update_layout(margin = dict(l = 0, r = 0, b = 0, t = 0), scene = dict(
		xaxis_title = 'Energy',
		yaxis_title = 'Style',
		zaxis_title = 'Acousticness',
		xaxis = dict(showticklabels = False, nticks = 5, range = [-200, 200]),
		yaxis = dict(showticklabels = False, nticks = 5, range = [-200, 200]),
		zaxis = dict(showticklabels = False, nticks = 5, range = [-200, 200]),
	))

	return fig


def get_top_labels(df, dims, n):
	''' Get the top n labels for a given artist. '''
	label_weights = pd.DataFrame(df[dims].sum().sort_values(ascending = False)).reset_index()
	label_weights.columns = ['label', 'weight']
	label_weights.weight = label_weights.weight / label_weights.weight.max()
	return label_weights[:n]




##################################################################################
#                                                                                #
# Classify a Song                                                                #
# ::: Get Top/Bottom 5 Labels for a track specified by the user. If the track    #
# ::: has already been classified, pull from db. Otherwise, pull audio from      #
# ::: Spotify and classify using a distilled version of the model.               #
#                                                                                #
##################################################################################


def classify_song():
	'''
	Potential additional features:
		- Similar tracks (based on emotional signature)
	'''

	# load all data
	df = load_data()
	
	# copy
	st.title('Classify a Song')
	st.markdown('Want to analyze a specific track? Enter the Spotify URL (or Track ID if you know it):')
	st.markdown('<strong>To get the track\'s URL from the Spotify app</strong>: \n - Drag the track into the search box below, or \n - Click on <em>Share >> Copy Song Link</em> and paste below.', unsafe_allow_html = True)

	# user input
	track_id = st.text_input('Enter Track URL:')
	st.markdown('<span style="font-size:0.8em">*Note: Unfortunately, due to licensing restrictions, many songs from some of the more popular artists are unavailable.*</span>', unsafe_allow_html = True)
	if len(track_id) > 22:
		track_id = track_id.split('?')[0].split('/track/')[1]
	show_spectros = st.checkbox('Show Spectrograms', value = False)

	# check if a track_id has been entered
	if len(track_id) > 0:
	
		# get track from Spotify API
		track = get_spotify_track(track_id)
		st.subheader('Track Summary')
		st.table(get_track_summmary_df(track))

		# check if there is track preview available from Spotify
		if track['preview_url']:

			# display 30 second track preview
			st.subheader('Track Preview (What the Algorithm "Hears")')
			st.write('')
			preview = get_track_preview(track_id)
			st.audio(preview)

			# get top and bottom labels for the track
			st.subheader('Track Analysis')
			track_df = deepcopy(DF[DF.track_id == track_id].reset_index(drop = True))

			# return values from db if already classified, otherwise classify
			if len(track_df) > 0:
				track_df = deepcopy(track_df.iloc[:,5:].T.rename(columns = {0: 'score'}).sort_values('score', ascending = False))
				st.table(pd.DataFrame({'Top 5': track_df[:5].index.tolist(), 'Bottom 5': track_df[-5:].index.tolist()}))
				if show_spectros: generate_spectros(preview)
			else:
				generate_spectros(preview)
				track_df = get_predictions()
				st.table(pd.DataFrame({'Top 5': track_df[:5].index.tolist(), 'Bottom 5': track_df[-5:].index.tolist()}))

			if show_spectros:
				st.subheader('Spectrograms (What the Algorithm "Sees")')
				generate_grid()
				st.image(image = path('tmp/png/grid.png'), use_column_width = True)

		# Spotify doesn't have preview for track
		else:
			st.write('Preview unavailable for this track :(')


def get_spotify_track(track_id):
	''' Get track from Spotify, given its Track ID. '''
	return SP.track(track_id)


def get_track_preview(track_id):
	''' Get a 30 Second Preview, if available. '''
	return requests.get(get_spotify_track(track_id)['preview_url']).content


def get_track_summmary_df(track):
	''' Build a summary for a given track for display '''
	return pd.DataFrame([{
				'Track': track['name'],
				'Artist': track['artists'][0]['name'],
				'Album': track['album']['name'],
				'Popularity': track['popularity'],
			}], index = [' '])[['Track', 'Artist', 'Album', 'Popularity']]


class SpecMixUpINCR(SpecMixUp):
	'''Spectral MixUp for Conv Net'''
	def __init__(self, learn:Learner):
		super().__init__(learn)
		self.masking_max_percentage = 0.5
		self.alpha = 0.8


def get_predictions():
	'''Get predictions for a given song. Note that this is just a distilled version
	of the model. Currently, it averages predictions on all spectros found in /tmp'''

	model_weights = {
		'mxrn18_partition4_multi-mixup-4_bce_448-sz_32-bs_6-ep_0.0001-lr_2': 0.526,
		'mxrn18_partition5_multi-mixup-4_bce_448-sz_32-bs_5-ep_0.0001-lr_2': 0.474,
	}

	# build DataBunch for FastAI
	data = (ImageList.from_folder(path('tmp/png'))
			.split_none()
			.label_from_folder()
			.transform(size = IMG_SZ)
			.databunch(bs = 1)
			.normalize(imagenet_stats)
	)

	# get predictions for track with models (total of 8 predictions: [4 spectros x 2 models])
	all_preds = pd.DataFrame()
	for model, weight in model_weights.items():
		print('Predicting with:', model)
		learn = load_learner(MODEL_PATH, f'{model}_export.pkl')
		model_preds = [[item.item() for item in torch.sigmoid(learn.predict(data.train_ds[i][0])[2])] for i in range(4)]
		all_preds = all_preds.append(deepcopy(pd.DataFrame(pd.DataFrame(model_preds, columns = LABELS).mean() * weight)))
		del learn, model_preds; gc.collect()
	all_preds = deepcopy(all_preds.reset_index().groupby('index').mean().sort_values(0).rename(columns = {0: 'score'}))
	
	return all_preds.sort_values('score', ascending = False)


def generate_spectros(audio):
	'''Generate spectrograms of a given audio for input into conv net.'''
	
	# convert mp3 to wav
	spec = Spectrogram()
	sound = AudioSegment.from_file(BytesIO(audio)).set_channels(1)
	sound.export(path('tmp/wav/user_classify.wav'), format = 'wav')

	# generate spectrograms
	spec.load_and_partition(
		wav_fname = path('tmp/wav/user_classify.wav'),
		png_fname = path('tmp/png/user_classify.png'),
		img_sz = 448,
		window_sz = 8192,
		n_mels = 512,
		hop_length = 128,
		top_db = 90,
		cmap = 'magma',
		how = 'slide',
		n = 4,
		to_disk = False,
	)


def generate_grid():
	'''Generate a grid of images from a set of images in a directory (used to display
	spectrograms in lieu of CSS styling.'''

	# Config:
	images_dir = path('tmp/png')
	result_grid_filename = f'{images_dir}/grid.png'
	result_figsize_resolution = 30 # default: 40
	images_list = os.listdir(images_dir)
	images_count = len(images_list)

	# Calculate the grid size:
	grid_size = math.ceil(math.sqrt(images_count))

	# Create plt plot:
	fig, axes = plt.subplots(grid_size, grid_size, figsize=(result_figsize_resolution, result_figsize_resolution))

	current_file_number = 0
	for image_filename in images_list:
		x_position = current_file_number % grid_size
		y_position = current_file_number // grid_size
		plt_image = plt.imread(images_dir + '/' + images_list[current_file_number])
		axes[x_position, y_position].imshow(plt_image)
		print((current_file_number + 1), '/', images_count, ': ', image_filename)
		current_file_number += 1

	plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
	plt.savefig(result_grid_filename)




##################################################################################
#                                                                                #
# Display Emotional Spectrum                                                     #
# ::: Display all labels as a 3D Scatter chart for the user to explore.          #
#                                                                                #
##################################################################################


def display_emotional_spectrum():
	st.title('Emotional Spectrum')
	st.write(
		"""Visually explore the algorithm's mapping of the emotional spectrum of the musical experience below.
		Use the \"Expand\" button in the top right corner of the chart for the best view!"""
	)
	st.plotly_chart(all_labels_scatter(), width = 0)


def all_labels_scatter():
	''' Display a 3D Scatter Plot using all of the labels in the algorithm. '''

	tdf = TSNE.rename(columns = {'1d0': 'color', '3d0': 'energy', '3d1': 'style', '3d2': 'acousticness'})

	fig = go.Figure(data = [go.Scatter3d(
		x = tdf['energy'], y = tdf['style'], z = tdf['acousticness'],
		mode = 'markers+text',
		text = tdf['label'],
		marker = dict(
			color = tdf['color'],
			opacity = 0.6,
			colorscale = 'RdBu',
		)
	)])

	fig.update_layout(margin = dict(l = 0, r = 0, b = 0, t = 0), scene = dict(
		xaxis_title = 'Energy',
		yaxis_title = 'Style',
		zaxis_title = 'Acousticness',
		xaxis = dict(showticklabels = False, nticks = 5, range = [-200, 200]),
		yaxis = dict(showticklabels = False, nticks = 5, range = [-200, 200]),
		zaxis = dict(showticklabels = False, nticks = 5, range = [-200, 200]),
	))

	return fig




##################################################################################
#                                                                                #
# Execute                                                                        #
#                                                                                #
##################################################################################

if __name__ == "__main__":

	# display and machine options
	pd.set_option('display.max_colwidth', -1)
	defaults.device = torch.device('cpu')
	ImageFile.LOAD_TRUNCATED_IMAGES = True

	# Spotify API
	SP = spotipy.Spotify(client_credentials_manager = SpotifyClientCredentials(
		client_id     = '2da510ef59984487a529a8cc09125ae8',
		client_secret = '31fbd44398d647e2864e86f42670ad38',
	))

	# data & constants
	ROOT_DIR   = ''
	MODEL_PATH = path('models/exported')
	IMG_SZ     = 448
	DF         = load_data()
	TSNE       = load_tsne()

	# execute
	main()



