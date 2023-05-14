import re
import time

from sklearn.metrics.pairwise import cosine_similarity
import inflect
import tensorflow_hub as hub
from nltk import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from appos import appos_dict
import numpy as np
from scipy import spatial

p = inflect.engine()


class API_Keyword_Cluster:
    vader_analyzer = None
    use_encoder = None

    def __init__(self, messages):
        self.vader_analyzer = self.load_dependencies()
        self.use_encoder = self.load_use()
        self.messages = messages

    def process_keywords(self, data):
        """
        Store the keywords and their sentiments from the params and
        store the topic as the key in topic_keywords dict
        and
        store the additional_keywords in the additional_keyword
        """
        keywords_sentiment = {}
        topic_keywords = {}
        additional_keyword = []

        if "primary_keyword" in data:
            for keyword_dict in data["primary_keyword"]:
                keyword_dict["keyword"] = self.preprocess_keywords(keyword_dict["keyword"])

                keyword = keyword_dict["keyword"]
                sentiment = keyword_dict["sentiment"]
                keywords_sentiment[keyword] = sentiment

                if keyword in topic_keywords:
                    if keyword not in topic_keywords[keyword]:
                        topic_keywords[keyword].append(keyword)
                else:
                    topic_keywords[keyword] = [keyword]

        for keyword_dict in data["candidate_keyword"]:
            keyword_dict["keyword"] = self.preprocess_keywords(keyword_dict["keyword"])

            keyword = keyword_dict["keyword"]
            sentiment = keyword_dict["sentiment"]
            if keyword not in additional_keyword:
                keywords_sentiment[keyword] = sentiment
                additional_keyword.append(keyword)

        if "primary_keyword" in data:
            print(topic_keywords)
            return keywords_sentiment, topic_keywords, additional_keyword
        else:
            return keywords_sentiment, additional_keyword

    def preprocess_keywords(self, text):
        text = re.sub(r'#\w+', '', text)
        text = text.lower()

        # Replace underscores with spaces
        text = text.replace('_', ' ')

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
        p = inflect.engine()
        clean_words = []
        for word in words:
            if not word or word.isspace():
                continue
            if word.isdigit():
                clean_words.append(p.number_to_words(word))
            else:
                clean_words.append(word)
        cleaned_text = ' '.join(clean_words)
        return cleaned_text

    @classmethod
    def load_dependencies(cls):
        if cls.vader_analyzer is None:
            cls.vader_analyzer = SentimentIntensityAnalyzer()
        return cls.vader_analyzer

    @classmethod
    def load_use(cls):
        if cls.use_encoder is None:
            extracted_folder_path = "/workspace/models/universal-sentence-encoder_4"
            cls.use_encoder = hub.load(extracted_folder_path)
        return cls.use_encoder

    def get_embeddings(self, preprocessed_sentence):
        start = time.time()
        sentence_embeddings = self.use_encoder(preprocessed_sentence).numpy()
        end = time.time()
        total = end - start
        print("Total time taken for embeddings is :: ")
        print(total)
        return sentence_embeddings

    def get_similarity_matrix(self, sentence_embeddings):
        return cosine_similarity(sentence_embeddings)

    def get_topic_keyword_cluster(self, keywords_sentiment, topic_keyword, additional_keyword):
        clusters = topic_keyword.copy()
        threshold = 0.65
        start = time.time()
        topic_keyword_embeddings = {}
        for key in topic_keyword:
            topic_keyword_embeddings[key] = self.get_embeddings([key])

        additional_keyword_embeddings = {}
        for sentence in additional_keyword:
            additional_keyword_embeddings[sentence] = self.get_embeddings([sentence])

        new_topic_keyword_embeddings = {}

        for new_sentence in additional_keyword:
            new_sentiment = keywords_sentiment[new_sentence]
            assigned = False

            for topic_key in clusters:
                topic_sentiment = keywords_sentiment[topic_key]

                if topic_key in new_topic_keyword_embeddings:
                    distance = spatial.distance.cdist(
                        new_topic_keyword_embeddings[topic_key],
                        additional_keyword_embeddings[new_sentence],
                        'cosine'
                    )
                    score_cosine = 1 - distance[0][0]
                    distance = spatial.distance.cdist(
                        new_topic_keyword_embeddings[topic_key],
                        additional_keyword_embeddings[new_sentence],
                        'correlation'
                    )
                    score_correlation = 1 - distance[0][0]

                else:
                    distance = spatial.distance.cdist(
                        topic_keyword_embeddings[topic_key],
                        additional_keyword_embeddings[new_sentence],
                        'cosine'
                    )
                    score_cosine = 1 - distance[0][0]

                    distance = spatial.distance.cdist(
                        new_topic_keyword_embeddings[topic_key],
                        additional_keyword_embeddings[new_sentence],
                        'correlation'
                    )
                    score_correlation = 1 - distance[0][0]

                weight_cosine = 0.9
                weight_correlation = 0.3
                total_cosine = weight_cosine * score_cosine
                total_correlation = weight_correlation * score_correlation
                score = total_cosine + total_correlation

                if score >= threshold and new_sentiment == topic_sentiment:
                    if new_sentence not in clusters[topic_key]:
                        clusters[topic_key].append(new_sentence)
                    assigned = True
                    break

            if not assigned:
                clusters[new_sentence] = [new_sentence]
                new_topic_keyword_embeddings[new_sentence] = additional_keyword_embeddings[
                    new_sentence]

        end = time.time()
        print("Total time taken for clustering :: ")
        print(end-start)
        return clusters

    def get_cluster_keywords(self, additional_keyword, keywords_sentiment):
        threshold = 0.65
        clusters = []

        sentence_embeddings = np.array([self.get_embeddings([sentence]) for sentence in additional_keyword])

        for i in range(len(additional_keyword)):
            if any(i in cluster for cluster in clusters):
                continue
            new_cluster = {i}
            print("i")
            print(additional_keyword)
            for j in range(i + 1, len(additional_keyword)):
                distance = spatial.distance.cdist(
                    sentence_embeddings[i].reshape(1, -1),
                    sentence_embeddings[j].reshape(1, -1),
                    'cosine'
                )
                print("j")
                print(j)
                score = 1 - distance[0][0]

                if (score >= threshold) and (
                        not any(j in cluster for cluster in clusters)) and (
                        keywords_sentiment[additional_keyword[i]] == keywords_sentiment[additional_keyword[j]]):
                    new_cluster.add(j)

            clusters.append(new_cluster)

        cluster_map = {cluster_index: [additional_keyword[i] for i in cluster] for cluster_index, cluster in
                       enumerate(clusters)}

        return self.get_keys(cluster_map)

    def get_keys(self, cluster_map):
        keyword_dict = {}
        for key, values in cluster_map.items():
            max_polarity_score = -2
            max_keyword = ""
            for value in values:
                polarity_score = self.vader_analyzer.polarity_scores(value)["compound"]
                if polarity_score > max_polarity_score:
                    max_polarity_score = polarity_score
                    max_keyword = value
            if max_keyword in keyword_dict:
                keyword_dict[max_keyword].extend(values)
            else:
                keyword_dict[max_keyword] = values

        keyword_dict = {key: values for key, values in keyword_dict.items() if key}
        return keyword_dict

