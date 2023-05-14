from collections import OrderedDict

import streamlit as st

from without_model import Keyword_Cluster


def analyze2(input_json):
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
    response["output_keywords"] = common_map
    return response


def load_example_data(example_type):
    if example_type == "Example 1":
        return {
            "keywords":
                {
                    "amazing product": [
                        "nice product",
                        "excellent product"
                    ],
                    "good feedback": [
                        "constructive feedback",
                        "helpful feedback"
                    ],
                    "awesome experience": [
                        "nice experience",
                        "good experience"
                    ]
                }
        }
    elif example_type == "Example 2":
        return {
            "keywords":
                {
                    "wonderful comments": [
                        "awesome Comments",
                        "worst performance"
                    ],
                    "Acceptable output": [
                        "Decent output",
                        "Acceptable output"
                    ],
                    "allowable output": [
                        "Excellent comments",
                        "worse Performance"
                    ]
                }
        }
    else:
        return {"keywords": {"sample": []}, "new_keywords": []}


st.set_page_config(page_title="Text Clustering", page_icon=":books:", layout="wide")
st.markdown("<h4 class='main-text'>Keyword Reasoning Model</h4>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Settings")
    example_type = st.radio("Choose an example:", options=["Example 1", "Example 2"])

keywords = st.session_state.get("keywords", {})

previous_example_type = st.session_state.get("previous_example_type", "")

if example_type != previous_example_type:
    st.session_state["previous_example_type"] = example_type
    st.session_state["input_boxes"] = list(range(len(keywords)))
    st.session_state["keywords"] = keywords

input_boxes = st.session_state.get("input_boxes", list(range(len(keywords))))
for i in range(len(input_boxes)):
    keyword = st.session_state.get(f"keyword_{i}", "")
    related_sentences = st.session_state.get(f"related_sentences_{i}", "")
    if keyword and related_sentences:
        related_sentences_list = [s.strip() for s in related_sentences.split(",")]
        keywords[keyword] = related_sentences_list


def display_input_boxes(index, keyword, related_sentences):
    col1, col2 = st.columns([0.2, 0.6])

    box_style = """
    background-color: #4B4B4B;
    color: #ffffff;
    padding: 10px;
    margin-bottom: 10px;
    border-radius: 5px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    """

    keyword_style = """
    font-size: 1em;
    """

    related_sentences_style = """
    font-size: 1em;
    """

    with col1:
        st.markdown(f'<div style="{box_style}; {keyword_style}">{keyword}</div>', unsafe_allow_html=True)

    with col2:
        st.markdown(f'<div style="{box_style}; {related_sentences_style}">{related_sentences}</div>',
                    unsafe_allow_html=True)


def add_keyword_form(index):
    if keywords and index < len(keywords):
        keyword = list(keywords.keys())[index]
        related_sentences = ', '.join(keywords[list(keywords.keys())[index]])
        display_input_boxes(index, keyword, related_sentences)


for i in range(len(input_boxes)):
    add_keyword_form(i)

st.markdown("<h6>New Keywords</h6>", unsafe_allow_html=True)
new_keywords_input = st.text_area("Add new list of keywords in the below box",
                                  placeholder="awesome book, awesome response", height=140)

if new_keywords_input:
    new_keywords = [keyword.strip() for keyword in new_keywords_input.split(",")]
    st.session_state["new_keywords"] = new_keywords
else:
    new_keywords = st.session_state.get("new_keywords", [])

output_keywords = st.session_state.get("output_keywords", [])


def display_output_boxes(index):
    col1, col2 = st.columns([0.2, 0.6])

    box_style = """
    background-color: #4B4B4B;
    color: #ffffff;
    padding: 10px;
    margin-bottom: 10px;
    border-radius: 5px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    """

    keyword_style = """
    font-size: 1em;
    """

    related_sentences_style = """
    font-size: 1em;
    """

    with col1:
        if output_keywords and index < len(output_keywords):
            keyword = list(output_keywords.keys())[index]
            st.markdown(f'<div style="{box_style}; {keyword_style}">{keyword}</div>', unsafe_allow_html=True)

    with col2:
        if output_keywords and index < len(output_keywords):
            related_sentences = ', '.join(output_keywords[list(output_keywords.keys())[index]])
            st.markdown(f'<div style="{box_style}; {related_sentences_style}">{related_sentences}</div>',
                        unsafe_allow_html=True)


if st.button("Analyze"):
    with st.spinner("Analyzing... ⏳"):
        input_json = {"keywords": keywords, "new_keywords": new_keywords}
        print("Printing the input json")
        print(input_json)
        result = analyze2(input_json)
        st.session_state["result"] = result

result = st.session_state.get("result", {})

if result:
    st.subheader("Results")
    output_keywords = result.get("output_keywords", [])
    st.session_state["output_keywords"] = output_keywords

    for i in range(len(output_keywords)):
        display_output_boxes(i)

if __name__ == "__main__":
    st.markdown("")
