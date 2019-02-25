import os, sys
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# this function receives one or two dataframes, where each dataframe contains the vectors of all articles of one topic
# if one dataframe is specified, the average vector similarity of each article in the topic dataframe is computed with vector of each article in the same topic
# if two dataframes are specified, the average vector similarity of each article in first topic dataframe is computed with vector of each article in the second topic dataframe
def compute_avg_similarity(topic_1_articles, topic_2_articles = None):
    # this means that we compute average similarity of each article in topic 1 over topic 1, averaged out
    if topic_2_articles is None:
        comparison_topic_articles = topic_1_articles
    # this means that we compute average similarity of each article in topic 1 over topic 2, averaged out
    else:
        comparison_topic_articles = topic_2_articles
        
    avg_cosine_sim_topic = []
    similarities = cosine_similarity(topic_1_articles, comparison_topic_articles)
    avg_cosine_sim_topic = np.mean(similarities)
    # for index, row in topic_1_articles.iterrows():
    #     avg_cosine_sim_doc = []
    #     for sub_index, sub_row in comparison_topic_articles.iterrows():
    #         cosine_sim = cosine_similarity([row], [sub_row])
    #         avg_cosine_sim_doc.append(cosine_sim[0][0])
    #     avg_cosine_sim_doc = sum(avg_cosine_sim_doc)/len(avg_cosine_sim_doc)
    #     avg_cosine_sim_topic.append(avg_cosine_sim_doc)
    
    # avg_cosine_sim_topic = sum(avg_cosine_sim_topic)/len(avg_cosine_sim_topic)
    return avg_cosine_sim_topic



parser = argparse.ArgumentParser(description="Compute some similarity statistics.")
parser.add_argument("vectorfile", type=str,
                    help="The name of the input  file for the matrix data.")

args = parser.parse_args()

if not args.vectorfile.lower().endswith("csv"):
	print("Please specify an input file of .csv format!")
	exit(1)

print("Reading matrix from {}.".format(args.vectorfile))
# read csv file specified in command line arguments
df = pd.read_csv(args.vectorfile, index_col=[0,1], skipinitialspace=True)

df_levels = list(df.groupby(level=0))
if len(df_levels) < 1:
	print("Please specify a file containing multi-indexed data of two different topics")
	exit(1)

print("Computing similarities...")
# get the topic names
topic_1 = df_levels[0][0].split("/")[-1]
topic_2 = df_levels[1][0].split("/")[-1]
# get the topic dataframes
topic_1_articles = df_levels[0][1]
topic_2_articles = df_levels[1][1]

# compute average similarity over all articles in topic 1, averaged out
avg_similarity_crude_crude = compute_avg_similarity(topic_1_articles)
# compute average similarity over all articles in topic 1, averaged out
avg_similarity_grain_grain = compute_avg_similarity(topic_2_articles)
# compute average similarity over all articles in topic 1 with all articles in topic 2, averaged out
avg_similarity_crude_grain = compute_avg_similarity(topic_1_articles, topic_2_articles)
# compute average similarity over all articles in topic 2 with all articles in topic 1, averaged out
avg_similarity_grain_crude = compute_avg_similarity(topic_2_articles, topic_1_articles)

print("Average similarity - %s %s: %f" % (topic_1, topic_1, avg_similarity_crude_crude))
print("Average similarity - %s %s: %f" % (topic_1, topic_2, avg_similarity_crude_grain))
print("Average similarity - %s %s: %f" % (topic_2, topic_1, avg_similarity_grain_crude))
print("Average similarity - %s %s: %f" % (topic_2, topic_2, avg_similarity_grain_grain))