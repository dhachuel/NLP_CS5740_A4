import json
import os
import numpy as np
from collections import Counter
from argparse import ArgumentParser
from model import Model
from alchemy_fsa import AlchemyFSA
from alchemy_world_state import AlchemyWorldState
from fsa import NO_ARG
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
from gensim.models import Word2Vec, KeyedVectors
import multiprocessing

# GLOBALS
global ACTION_TYPES, BEAKER_NUMBERS, CHEMICAL_COLOR_MAP, CHEMICAL_COLORS, WORD_EMB_SIZE
ACTION_TYPES = {
	'pop': 0,
	'push': 1
}
BEAKER_NUMBERS = tuple(list(range(1, 8)))
CHEMICAL_COLOR_MAP = {
	'b':"brown",'g':"green",
	'o':"orange", 'p':"purple",
	'r':"red", 'y':"yellow"
}
CHEMICAL_COLORS = ('b', 'g', 'o', 'p', 'r', 'y')
WORD_EMB_SIZE = 50


def tokenize_word(text, stem=False):
	# Extract words from text
	words = word_tokenize(text=text)

	# Stem
	stemmer = PorterStemmer()

	# Clean characters and tokenize text
	if stem:
		tokens = [stemmer.stem(w.lower()) for w in words]
	else:
		tokens = [w.lower() for w in words]

	return tokens

def action2vec(action):
	action_type_encoding = ACTION_TYPES[action.split(" ")[0]]
	beaker_number_encoding = int(action.split(" ")[1])
	if len(action.split(" ")) != 3:
		chemical_color_encoding = 0
	else:
		chemical_color_encoding = CHEMICAL_COLORS.index(action.split(" ")[2]) + 1

	return [action_type_encoding, beaker_number_encoding, chemical_color_encoding]

def make_dictionary(corpus):
	"""

	:return:
	:param corpus:
	:return:
	"""
	words = tokenize_word(text=corpus)  # Tokenize raw text
	word_list = sorted(list(words))  # Sort alphabetically

	# Sort by frequency and return
	word_frequencies = Counter(word_list)
	words_sorted_by_frequency = sorted(word_frequencies.items(), key=lambda kv: kv[1], reverse=True)
	dictionary = [item[0] for item in words_sorted_by_frequency]
	return {k: v for v, k in enumerate(dictionary)}

def load_data(filename, train_filename):
	"""Loads the data from the JSON files.

	You are welcome to create your own class storing the data in it; e.g., you
	could create AlchemyWorldStates for each example in your data and store it.

	Inputs:
		filename (str): Filename of a JSON encoded file containing the data.

	Returns:
		examples
	"""
	with open(filename, 'r') as json_file:
		data = json.load(json_file)
	with open(train_filename, 'r') as json_file:
		train_data = json.load(json_file)
	corpus_instruction_sentences, action_embeddings, data_instructions = [], [], []
	for example in data:
		for item in example["utterances"]:
			action_embeddings.append([action2vec(action) for action in item["actions"]])
			data_instructions.append(item["instruction"].strip().split(" "))
	for example in train_data:
		for item in example["utterances"]:
			corpus_instruction_sentences.append(item["instruction"].strip().split(" "))

	# Train word embeddings
	if os.path.isdir("cache"):
		os.makedirs("cache")
	word2vec_cache_filename = "cache/word2vec.model"
	cores = multiprocessing.cpu_count()  # Count the number of cores in a computer
	w2v_model = Word2Vec(
		window=4,
		size=WORD_EMB_SIZE,
		sample=6e-5,
		alpha=0.03,
		min_alpha=0.0007,
		negative=5,
		workers=cores - 1
	)
	if not os.path.exists(word2vec_cache_filename):
		w2v_model.build_vocab(corpus_instruction_sentences, progress_per=1000)
		w2v_model.train(corpus_instruction_sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
		w2v_model.init_sims(replace=True)
		w2v_model.save(word2vec_cache_filename)
	else:
		w2v_model = Word2Vec.load(word2vec_cache_filename)
	instruction_embeddings = []
	for instruction in data_instructions:
		instruction_embedding = []
		for word in instruction:
			if word in w2v_model.wv.vocab:
				instruction_embedding.append(w2v_model.wv.get_vector(word=word))
			else:
				instruction_embedding.append(np.random.normal(0, 1, WORD_EMB_SIZE))
		instruction_embeddings.append(instruction_embedding)


	return data, instruction_embeddings, action_embeddings

def train(model, train_data):
	"""Finds parameters in the model given the training data.

	TODO: implement this function -- suggested implementation iterates over epochs,
		computing loss over training set (in batches, maybe), evaluates on a held-out set
		at each round (you are welcome to split train_data here, or elsewhere), and
		saves the final model parameters.

	Inputs:
		model (Model): The model to train.
		train_data (list of examples): The training examples given.
	"""
	pass

def execute(world_state, action_sequence):
	"""Executes an action sequence on a world state.

	TODO: This code assumes the world state is a string. However, you may sometimes
	start with an AlchemyWorldState object. I suggest loading the AlchemyWorldState objects
	into memory in load_data, and moving that part of the code to load_data. The following
	code just serves as an example of how to 1) make an AlchemyWorldState and 2) execute
	a sequence of actions on it.

	Inputs:
		world_state (str): String representing an AlchemyWorldState.
		action_sequence (list of str): Sequence of actions in the format ["action arg1 arg2",...]
			(like in the JSON file).
	"""
	alchemy_world_state = AlchemyWorldState(world_state)
	fsa = AlchemyFSA(alchemy_world_state)

	for action in action_sequence:
		split = action.split(" ")
		act = split[0]
		arg1 = split[1]

		# JSON file doesn't contain  NO_ARG.
		if len(split) < 3:
			arg2 = NO_ARG
		else:
			arg2 = split[2]

		fsa.feed_complete_action(act, arg1, arg2)

	return fsa.world_state()

def predict(model, data, outname):
	"""Makes predictions for data given a saved model.

	This function should predict actions (and call the AlchemyFSA to execute them),
	and save the resulting world states in the CSV files (same format as *.csv).

	TODO: you should implement both "gold-previous" and "entire-interaction"
		prediction.

	In the first case ("gold-previous"), for each utterance, you start in the previous gold world state,
	rather than the on you would have predicted (if the utterance is not the first one).
	This is useful for analyzing errors without the problem of error propagation,
	and you should do this first during development on the dev data.

	In the second case ("entire-interaction"), you are continually updating
	a single world state throughout the interaction; i.e. for each instruction, you start
	in whatever previous world state you ended up in with your prediction. This method can be
	impacted by cascading errors -- if you made a previous incorrect prediction, it's hard
	to recover (or your instruction might not make sense).

	For test labels, you are expected to predict /final/ world states at the end of each
	interaction using "entire-interaction" prediction (you aren't provided the gold
	intermediate world states for the test data).

	Inputs:
		model (Model): A seq2seq model for this task.
		data (list of examples): The data you wish to predict for.
		outname (str): The filename to save the predictions to.
	"""
	pass

def main():
	# A few command line arguments
	parser = ArgumentParser()
	parser.add_argument("--train", type=bool, default=False)
	parser.add_argument("--predict", type=bool, default=False)
	parser.add_argument("--saved_model", type=str, default="")
	args = parser.parse_args()

	assert args.train or args.predict

	# Load the data; you can also use this to construct vocabularies, etc.
	train_data, train_instruction_embeddings, train_action_embeddings = load_data(filename="data/train.json", train_filename="data/train.json")
	dev_data, dev_instruction_embeddings, dev_action_embeddings = load_data(filename="data/dev.json", train_filename="data/train.json")
	test_data, test_instruction_embeddings, _ = load_data(filename="data/test.json", train_filename="data/train.json")

	# Construct a model object.
	model = Model()

	if args.train:
		# Trains the model
		train(model, train_data)
	if args.predict:
		# Makes predictions for the data, saving it in the CSV format
		assert args.saved_model

		# TODO: you can modify this to take in a specified split of the data,
		# rather than just the dev data.
		predict(model, dev_data)

		# Once you predict, you can run evaluate.py to get the instruction-level
		# or interaction-level accuracies.


if __name__ == "__main__":
	main()
