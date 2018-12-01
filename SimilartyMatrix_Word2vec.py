Class SimilarityMatrix:
	def __init__(self,n_words = 20000):
		# self.bow2vec = pickle_loading('../data/Pickles/bow2_vec.pckl')
		
		# need a dictionary of word-embedding
		self.n_gram_vector = pickle_loading('/Users/Shepardlee/PycharmProjects/UTC_Resume_Matching/data/Pickles/bow2_vec_join_n_gram.pkl')
		self.words = list(self.n_gram_vector.keys())[0:n_words]
		self.vectors = list(self.n_gram_vector.values())[0:n_words]
		self.word_index = dict(zip(self.words, range(n_words)))
		self.index_word = dict(zip(range(n_words),self.words))
		self.distance_matrix = cosine_similarity(self.vectors,dense_output=True)
