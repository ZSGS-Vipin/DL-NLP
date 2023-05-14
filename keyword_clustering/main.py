from flask import Flask, request
from collections import OrderedDict
from flask import Response
import json
from without_model import Keyword_Cluster




#
# @app.route('/text-analysis/clusterdraft', methods=['POST'])
# def analyze2():
#     input_json = request.json
#     keywords = input_json['keywords']
#     new_keywords = input_json['new_keywords']
#
#     sentences_list = []
#     for key, values in keywords.items():
#         if key not in sentences_list:
#             sentences_list.append(key)
#         for value in values:
#             if value not in sentences_list:
#                 sentences_list.append(value)
#
#     for new_keyword in new_keywords:
#         if new_keyword not in sentences_list:
#             sentences_list.append(new_keyword)
#
#     print(type(sentences_list))
#     print("Sentence List Length:: ")
#     print(len(sentences_list))
#
#     clustering = Keyword_Cluster(sentences_list)
#     clustering.separate_positive_negative_keywords()
#     common_map = clustering.find_keywords()
#     print("Common Map")
#     print(common_map)
#     response = OrderedDict()
#     if common_map:
#         response["status_code"] = 200
#         response["status"] = "success"
#         response["response_keywords"] = common_map
#     else:
#         response["status_code"] = 500
#         response["status"] = "internal server error"
#         response["response_keywords"] = None
#
#     response_json = json.dumps(response)
#     return Response(response_json, content_type='application/json')


app = Flask(__name__)


@app.route('/text-analysis/cluster', methods=['POST'])
def analyze2():
    input_json = request.json
    email_keywords = input_json['email_keywords']
    new_keywords = input_json['new_keywords']

    sentences_list = []

    for email_keyword in email_keywords:
        if email_keyword not in sentences_list:
            sentences_list.append(email_keyword)

    for new_keyword in new_keywords:
        if new_keyword not in sentences_list:
            sentences_list.append(new_keyword)

    clustering = Keyword_Cluster(sentences_list)
    clustering_map = clustering.separate_positive_negative_keywords()
    common_map = clustering.find_keywords()

    response = OrderedDict()
    if common_map:
        response["status_code"] = 200
        response["status"] = "success"
        response["response_keywords"] = common_map
    else:
        response["status_code"] = 500
        response["status"] = "internal server error"
        response["response_keywords"] = None

    response_json = json.dumps(response)
    print(response_json)
    return Response(response_json, content_type='application/json')


# @app.route('/text-analysis/cluster', methods=['POST'])
# def analyze3():
#     input_json = request.json
#     messages = input_json['messages']
#
#     clustering = API_Keyword_Cluster(messages)
#     if "primary_keyword" in messages:
#         keyword_sentiment, topic_keywords, additional_keyword = clustering.process_keywords(messages)
#         cluster_dict = clustering.get_topic_keyword_cluster(keyword_sentiment, topic_keywords, additional_keyword)
#     else:
#         keyword_sentiment, additional_keyword = clustering.process_keywords(messages)
#         cluster_dict = clustering.get_cluster_keywords(additional_keyword, keyword_sentiment)
#
#     response = OrderedDict()
#     if cluster_dict:
#         response["status_code"] = 200
#         response["status"] = "success"
#         response["response_keywords"] = cluster_dict
#     else:
#         response["status_code"] = 500
#         response["status"] = "internal server error"
#         response["response_keywords"] = None
#
#     response_json = json.dumps(response)
#     print(response_json)
#     return Response(response_json, content_type='application/json')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)
