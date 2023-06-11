import logging

from com.tse.custom_sentence_splitter import CustomSentenceSplitter
from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.cluster import KMeans
import math
import pickle
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import operator
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
from datetime import datetime
import json
import os


# The dict below is a map between number of words (to nearest hundred ) and summary percentage.
# Subtracting 0.1 more because usually the summary length is > desired as we can
# not cut sentence in the middle
sum_percentage_map = {0: (0.8 - 0.1), 100: (0.7 - 0.1), 200: (0.55 - 0.1), 300: (0.45 - 0.1),
                      400: (0.35 - 0.1), 500: (0.30 - 0.1)}


def vectorize_sent(tokens, model):
    return np.mean([model[w] for w in tokens if w in model]
                   or [np.zeros(300)], axis=0)


def get_sen_cluster_map(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    return kmeans.labels_


def get_sentence_vectors(sentences, model, tokenizer):
    return np.array([list(get_sen_vector(str(sen), model, tokenizer)) for sen in sentences])


def get_noun_phrases(sentences):
    return [len(sen.noun_phrases) for sen in sentences]


def get_tf_idf_map(sentence_tokens, df_map, D):
    tf_idf_map = {}
    for i, sen in enumerate(sentence_tokens):
        tf_idf_map[i] = calculate_tf_idf_sum(sen, df_map, D)
    return tf_idf_map


def get_avg_cosine_sim(matrix):
    agg_cosine_sim = {}
    for i, m in enumerate(matrix):
        cs = 0
        for j, n in enumerate(matrix):
            if i != j:
                cs += cosine_similarity(m.reshape(1, -1), n.reshape(1, -1))
        agg_cosine_sim[i] = float(cs)
    return agg_cosine_sim


def calculate_tf_idf_sum(tokens, df_map, D):
    tf_map = dict(Counter(tokens))
    tf_idf_map = {k: (v * math.log(D / df_map.get(k, 1))) for k, v in tf_map.items()}
    return sum(tf_idf_map.values()) / len(tf_idf_map)


def get_sen_length(sentences):
    sen_length_map = {}
    for i, sen in enumerate(sentences):
        sen_length_map[i] = len(word_tokenize(str(sen)))
    return sen_length_map


def get_proper_nouns(sentences):
    nnp_list = []
    for sen in sentences:
        nnp_list.append(len([(word, pos) for (word, pos) in sen.tags if pos == 'NNP']))
    return nnp_list


def get_min_max(list_):
    return max(list_), min(list_)


def get_cue_phrases(sentences):
    cue_phrases = ['because', 'thus', 'conclusion', 'consequences', 'eventually',
                   'hardly', 'therby', 'significant', 'therby', 'reason', 'summarize',
                   'summarise', 'hence', 'plan']
    cue_phrase_return = []
    for sen in sentences:
        cue_phrase_return.append(any([x in str(sen).lower() for x in cue_phrases]))
    return cue_phrase_return


def get_final_ranks(n, sentences, tf_idf_map, sen_length_map,
                    avg_cosine_sim, noun_ph_list, cue_phrase_list, proper_noun_list):
    final_ranks = {}
    tf_idf_max, tf_idf_min = get_min_max(list(tf_idf_map.values()))
    sen_len_max, sen_len_min = get_min_max(list(sen_length_map.values()))
    avg_cosine_max, avg_cosine_min = get_min_max(list(avg_cosine_sim.values()))
    noun_ph_max, noun_ph_min = get_min_max(noun_ph_list)
    proper_noun_max, proper_noun_min = get_min_max(proper_noun_list)

    for i in range(n):
        tf_idf = get_val(tf_idf_map[i], tf_idf_min, tf_idf_max)
        sen_length = get_val(sen_length_map[i], sen_len_min, sen_len_max)
        cos_sim = get_val(avg_cosine_sim[i], avg_cosine_min, avg_cosine_max)
        noun_phrase = get_val(noun_ph_list[i], noun_ph_min, noun_ph_max)
        cue_ph = 0 if cue_phrase_list[i] else 1
        pr_noun = get_val(proper_noun_list[i], proper_noun_min, proper_noun_max)

        sen_pos = 1 - (((i + 1) - 1) / len(sentences))
        val = 0.6 * tf_idf + \
              0.05 * sen_length + \
              0.1 * pr_noun + \
              0.1 * noun_phrase + \
              0.05 * cos_sim + \
              0.05 * sen_pos + \
              0.05 * cue_ph
        final_ranks[i] = val

    return final_ranks


def get_final_cluster_score_map(final_rank_map, cluster_list, total_sentences):
    final_cluster_score_map = {}
    per_cluster_elements = {}
    for index in range(total_sentences):
        total_score = final_rank_map[index]
        cluster = cluster_list[index]
        final_cluster_score_map.setdefault(cluster, {})[index] = total_score
    final_cluster_sorted_rank_map = {}
    # Sort all these maps
    for cluster, rank_map in final_cluster_score_map.items():
        final_cluster_sorted_rank_map[cluster] = sorted(rank_map.items(), key=operator.itemgetter(1), reverse=True)
        per_cluster_elements[cluster] = len(rank_map)

    return final_cluster_sorted_rank_map, per_cluster_elements


def get_summary_sentences(summary_indices, sens):
    return [str(sens[x]) for x in summary_indices]


def get_val(val, min_val, max_val):
    if val == 0:
        return 0
    min_val = 0.001 if min_val <= 0 else min_val
    nor_val = ((val - min_val) + 0.0001) / ((max_val - min_val) + 0.0001)
    return nor_val


def get_sen_length(sentences):
    sen_length_map = {}
    for i, sen in enumerate(sentences):
        sen_length_map[i] = len(word_tokenize(str(sen)))
    return sen_length_map


def get_sorted_tuples(per_cluster_elements, final_cluster_sorted_rank_map, num_clusters):
    max_values = max(per_cluster_elements.values())
    sorted_tuples = []
    k = 0
    for i in range(max_values):
        sub_list = []
        for cluster in range(num_clusters):
            if k < per_cluster_elements[cluster]:
                sub_list.append(final_cluster_sorted_rank_map[cluster][k])
        k += 1
        sorted_tuples.append(sub_list)
    return sorted_tuples


def get_summary_indices(sorted_tuples, total_sum_words, sentences):
    sum_words = 0
    summary_indices = []
    for tpl in sorted_tuples:
        sorted_lst = sorted(dict(tpl).items(), key=operator.itemgetter(1), reverse=True)
        for entry in sorted_lst:
            index = entry[0]
            sum_words += len(sentences[index].words)
            if sum_words <= (total_sum_words + 10):
                # Adding 10  here because, while adding the sentences,
                # we the total words should not be more than summary_words + 10
                summary_indices.append(index)
            else:
                break
    return summary_indices


def get_sen_vector(text, model, tokenizer):
    marked_text = "[CLS] " + text + " [SEP]"
    # Tokenize our sentence with the BERT tokenizer.
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Mark each of the 22 tokens as belonging to sentence "1".
    segments_ids = [1] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)

    # `encoded_layers` has shape [12 x 1 x 22 x 768]

    # `token_vecs` is a tensor with shape [22 x 768]
    token_vecs = encoded_layers[11][0]

    # Calculate the average of all 22 token vectors.
    sentence_embedding = torch.mean(token_vecs, dim=0)
    return sentence_embedding


def get_summary_words(total_words):
    return int(sum_percentage_map.get((total_words // 100 * 100), 0.30) * total_words)


class SummaryGenerator(object):
    def __init__(self, df_file, D, num_clusters):
        logging.info("Here in the start of the class")
        self.splitter = CustomSentenceSplitter()
        logging.info("Going to load the model")

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.eval()

        logging.info("Model loaded successfully")

        self.num_clusters = num_clusters
        self.df_map = pickle.load(open(df_file, 'rb'))
        self.D = D
        logging.info("Done ...")

    def calculate_sentence_ranks(self, original_sentences, sentence_tokens, total_sum_words):
        tf_idf_map = get_tf_idf_map(sentence_tokens, self.df_map, self.D)
        logging.debug("tf-idf calculated")
        matrix = get_sentence_vectors(original_sentences, self.model, self.tokenizer)
        logging.debug("vectors calculated")
        total_sentences = len(original_sentences)
        print(self.num_clusters)
        num_clusters = self.num_clusters if total_sentences >= self.num_clusters else total_sentences
        print(num_clusters)
        print(total_sentences)
        logging.debug("clusters calculated")
        cluster_list = get_sen_cluster_map(matrix, num_clusters)
        sen_length_map = get_sen_length(original_sentences)
        logging.debug("sen_length_map calculated")
        avg_cosine_sim = get_avg_cosine_sim(matrix)
        logging.debug("avg_cosine_sim calculated")
        noun_ph_list = get_noun_phrases(original_sentences)
        logging.debug("noun_ph_list calculated")

        cue_phrase_list = get_cue_phrases(original_sentences)
        logging.debug("cue_phrase_list calculated")

        proper_noun_list = get_proper_nouns(original_sentences)
        logging.debug("proper_noun_list calculated")
        final_rank_map = get_final_ranks(total_sentences, original_sentences, tf_idf_map, sen_length_map,
                                         avg_cosine_sim, noun_ph_list, cue_phrase_list, proper_noun_list)

        logging.debug("final_rank_map calculated")

        final_cluster_sorted_rank_map, per_cluster_elements = get_final_cluster_score_map(
            final_rank_map, cluster_list, total_sentences)
        logging.debug("final_cluster_sorted_rank_map calculated")

        sorted_tuples = get_sorted_tuples(per_cluster_elements,
                                          final_cluster_sorted_rank_map, num_clusters)
        summary_indices = get_summary_indices(sorted_tuples,
                                              total_sum_words=total_sum_words, sentences=original_sentences)

        logging.debug("summary_indices calculated")

        summary_indices.sort()
        return get_summary_sentences(summary_indices, original_sentences)

    def get_summary_from_text(self, text):
        logging.info("Here going to generate the summary")
        original_sentences, sentence_tokens, msg = self.splitter.get_sentences_from_mail(text)
        logging.debug("Here coming back from sen splitter", len(original_sentences))
        total_words = sum([len(x.words) for x in [y for y in original_sentences]])
        if total_words > 1500 and len(original_sentences) > 70:
            logging.info("Text length increases the cutoff , going to decrease the text length")
            original_sentences = original_sentences[:70]
            sentence_tokens = sentence_tokens[:70]
            total_words = sum([len(x.words) for x in [y for y in original_sentences]])
        logging.info("total words {}".format(total_words))
        total_sum_words = get_summary_words(total_words)
        result = '\n'.join(self.calculate_sentence_ranks(original_sentences, sentence_tokens, total_sum_words))
        file_name = str(datetime.now().timestamp())
        log_wr = open("/tmp/" + file_name, 'w')
        log_wr.write(json.dumps({'mail': text, "result": result}) + "\n")
        logging.info("Completed summary generation")
        print(result)
        return result,msg


if __name__ == '__main__':
    D = 510000
    num_clusters = 3

    print("Here  at the top")
    home_dir = os.system("pwd")
    print("`num clusters%", num_clusters)
    sum_obj = SummaryGenerator('/Users/mudasir/ShortMail/df.map', D, num_clusters)
    content = """First, freeze all of your pip packages in the requirements.txt file using the command

pip freeze > requirements.txt
This should create the requirements.txt file in the correct format. Then try installing using the command

pip install -r requirements.txt
Make sure you're in the same folder as the file when running this command.

If you get some path name instead of the version number in the requirements.txt file, use this pip command to work around it.

pip list --format=freeze > requirements.txt"""
    file1 = open("MyFile.txt", "w")
    print("...................................\n")
    a=sum_obj.get_summary_from_text(content)
    file1.write(a)