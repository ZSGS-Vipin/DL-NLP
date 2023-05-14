import re
import time
from sklearn.metrics.pairwise import cosine_similarity
import contractions
import inflect
import numpy as np
import tensorflow_hub as hub
from nltk import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from appos import appos_dict
from slangs import slangs_dict

p = inflect.engine()


def expand_contractions(text):
    return contractions.fix(text)


def preprocess_text(text):
    text = re.sub(r'#\w+', '', text)
    text = text.lower()

    words = text.split()

    new_text = []
    for word in words:
        word_s = word.lower()
        if word_s in appos_dict:
            new_text.append(appos_dict[word_s])
        else:
            new_text.append(word)
    text = " ".join(new_text)

    words = text.split()
    new_text = []
    for word in words:
        word_s = word.lower()
        if word_s in slangs_dict:
            new_text.append(slangs_dict[word_s])
        else:
            new_text.append(word)
    text = " ".join(new_text)

    words = text.split()
    filter_words = [word for word in words if len(word) > 1]
    text = " ".join(filter_words)

    regex_pattern = re.compile(r'[\,+\:\?\!\"\(\)!\'\.\%\[\]]+')
    clean_text = regex_pattern.sub(r' ', text)

    clean_text = ' '.join(clean_text.strip().split())

    urls = re.finditer(r'http[\w]*:\/\/[\w]*\.?[\w-]+\.+[\w]+[\/\w]+', clean_text)
    for i in urls:
        clean_text = re.sub(i.group().strip(), '', clean_text)

    tokens = word_tokenize(clean_text)
    clean_text = ' '.join(tokens)

    words = clean_text.split()
    clean_words = []
    for word in words:
        if not word or word.isspace():
            continue
        if word.isdigit():
            clean_words.append(p.number_to_words(word))
        else:
            clean_words.append(word)
    return clean_words


class Keyword_Cluster:
    vader = None
    sentence_encoder = None

    @classmethod
    def load_USE_encoder(cls):
        if cls.sentence_encoder is None:
            extracted_folder_path = "/workspace/models/universal-sentence-encoder_4"
            cls.sentence_encoder = hub.load(extracted_folder_path)
        return cls.sentence_encoder

    @classmethod
    def load_vader_analyzer(cls):
        if cls.vader is None:
            cls.vader = SentimentIntensityAnalyzer()
        return cls.vader

    def __init__(self, input_data):
        embed_start = time.time()
        self.embed = self.load_USE_encoder()
        embed_end = time.time()
        print("Total Time Taken for loading the USE :: ")
        print(embed_end - embed_start)

        vader_start = time.time()
        self.vader_analyzer = self.load_vader_analyzer()
        vader_end = time.time()
        print("Total Time Taken for loading the vader :: ")
        print(vader_end - vader_start)

        self.input_data = input_data

        preprocess_start = time.time()
        self.preprocessed_sentence = [' '.join(preprocess_text(s)) for s in self.input_data]
        preprocess_end = time.time()
        print("Total length of  preprocess :: ")
        print(len(self.preprocessed_sentence))

        embeddings_start = time.time()
        self.sentence_embeddings = self.embed(self.preprocessed_sentence).numpy()
        embeddings_end = time.time()
        print("Total length of embeddings")
        print(len(self.sentence_embeddings))

        cosine_start = time.time()
        self.similarity_matrix = cosine_similarity(self.sentence_embeddings)
        cosine_end = time.time()
        print("Total Length of cosine similarity is :: ")
        print(len(self.similarity_matrix))

        clusters_start = time.time()
        self.clusters = self.cluster_sentence()
        clusters_end = time.time()
        print("Total time taken for the clusters is :: ")
        print(clusters_end - clusters_start)

        cluster_map_start = time.time()
        self.cluster_map = self.get_cluster_map()
        cluster_map_end = time.time()
        print("Total time taken for the cluster map is :: ")
        print(cluster_map_end - cluster_map_start)

        self.positive_keywords = {}
        self.negative_keywords = {}
        self.common_map = {}

    def get_similarity_matrix(self, embeddings):
        return np.dot(embeddings, embeddings.T)

    def cluster_sentence(self):
        start = time.time()
        threshold = 0.6
        clusters = []
        visited_indices = set()

        for i in range(len(self.preprocessed_sentence)):
            if i in visited_indices:
                continue
            new_cluster = {i}
            potential_cluster_indices = np.where(self.similarity_matrix[i] >= threshold)[0]
            potential_cluster_indices = set(potential_cluster_indices) - visited_indices - {i}

            new_cluster.update(potential_cluster_indices)
            visited_indices.update(new_cluster)
            clusters.append(new_cluster)

        end = time.time()
        total = end - start
        print("total length of clusters is :: ")
        print(len(clusters))
        return clusters

    def get_cluster_map(self):
        start = time.time()
        cluster_map = {}
        for idx, cluster in enumerate(self.clusters, start=1):
            cluster_sentences = [self.preprocessed_sentence[i] for i in cluster]
            cluster_map[idx] = cluster_sentences
        end = time.time()
        # print("Total time taken for making cluster map is :: ")
        # print(end - start)
        print("TOTAL LENGTH OF CLUSTERS MAP IS ::: ")
        print(len(cluster_map))
        return cluster_map

    def initialize_sentiment_analysis_pipeline(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()

    def classify_sentiment(self, text):
        sentiment_scores = self.vader_analyzer.polarity_scores(text)
        compound_score = sentiment_scores['compound']
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'

    def separate_positive_negative_keywords(self):
        start = time.time()
        for cluster_id, sentences in self.cluster_map.items():
            self.positive_keywords[cluster_id] = []
            self.negative_keywords[cluster_id] = []
            for sentence in sentences:
                sentiment = self.classify_sentiment(sentence)
                if sentiment == 'positive' or sentiment == 'neutral':
                    self.positive_keywords[cluster_id].append(sentence)
                elif sentiment == 'negative':
                    self.negative_keywords[cluster_id].append(sentence)
        end = time.time()
        total = end - start
        print("total length of positive and negative map is :: ")
        print(len(self.positive_keywords))
        print(len(self.negative_keywords))
        print(total)

    def find_keywords(self):
        start = time.time()
        for key, values in self.positive_keywords.items():
            max_polarity_score = -2
            max_keyword = ""
            for value in values:
                polarity_score = self.vader_analyzer.polarity_scores(value)["compound"]
                if polarity_score > max_polarity_score:
                    max_polarity_score = polarity_score
                    max_keyword = value
            if max_keyword in self.common_map:
                self.common_map[max_keyword].extend(values)
            else:
                self.common_map[max_keyword] = values

        for key, values in self.negative_keywords.items():
            min_polarity_score = -2
            min_keyword = ""
            for value in values:
                polarity_score = self.vader_analyzer.polarity_scores(value)["compound"]
                if polarity_score > min_polarity_score:
                    min_polarity_score = polarity_score
                    min_keyword = value
            if min_keyword in self.common_map:
                self.common_map[min_keyword].extend(values)
            else:
                self.common_map[min_keyword] = values

        self.common_map = {key: values for key, values in self.common_map.items() if key}
        end = time.time()
        total = end - start
        print("total length of common map is ")
        print(len(self.common_map))
        print(total)
        return self.common_map
