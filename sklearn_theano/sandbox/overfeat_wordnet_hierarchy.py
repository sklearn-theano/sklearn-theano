from __future__ import print_function
import numpy as np
from nltk.corpus import wordnet
from sklearn_theano.feature_extraction.overfeat_class_labels import (
    get_all_overfeat_labels)
import json


labels = get_all_overfeat_labels()

synsets = [wordnet.synset(
        label.split(',')[0].replace(" ", "_") + ".n.1")
           for label in labels]

wordnet_to_labels = dict(zip([synset.name() for synset in synsets], labels))
labels_to_wordnet = dict(zip(labels, [synset.name() for synset in synsets]))

hypernym_paths = [synset.hypernym_paths() for synset in synsets]

hierarchy = dict()
for synset, hpaths in zip(synsets, hypernym_paths):
    print(synset)
    hierarchy[synset.name()] = hierarchy.get(synset.name(),
                                             dict(children=[], parents=[]))
    for hpath in hpaths:
        old_item = synset.name()
        for item in hpath[::-1][1:]:
            new_element = hierarchy[item.name()] = hierarchy.get(item.name(),
                                     dict(children=[], parents=[]))
            hierarchy[old_item]["parents"] = list(np.unique(
                    hierarchy[old_item]["parents"] + [item.name()]))
            new_element["children"] = list(np.unique(
                    new_element["children"] + [old_item]))
            old_item = item.name()


def get_all_leafs(synset_name):
    hitem = hierarchy.get(synset_name, None)
    if hitem is None:
        raise Exception('synset is not in hierarchy')

    if hitem['children']:
        leafs = []
        for csynset in hitem['children']:
            leafs = leafs + get_all_leafs(csynset)
        leafs = list(np.unique(leafs))
        return leafs
    else:
        return [synset_name]


overfeat_leafs_for_wordnet_concept = dict()
for synset_name in hierarchy.keys():
    overfeat_leafs_for_wordnet_concept[synset_name] = [
        wordnet_to_labels[leaf]
        for leaf in get_all_leafs(synset_name)]


wordnet_to_labels_file = "wordnet_to_labels.json"
labels_to_wordnet_file = "labels_to_wordnet.json"
overfeat_leafs_file = "overfeat_leafs.json"
hierarchy_file = "hierarchy.json"

json.dump(wordnet_to_labels, open(wordnet_to_labels_file, "w"))
json.dump(labels_to_wordnet, open(labels_to_wordnet_file, "w"))
json.dump(overfeat_leafs_for_wordnet_concept, open(overfeat_leafs_file, "w"))
json.dump(hierarchy, open(hierarchy_file, "w"))

