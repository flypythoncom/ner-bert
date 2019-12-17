import urllib.request as request
import urllib
import sys
import os

tasks_urls = {
    "conll2003": [
        ["eng.testa", "https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.testa"],
        ["eng.testb", "https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.testb"],
        ["eng.train", "https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.train"]
    ]}


def download_data(task_name, data_dir):
    req = urllib
    if sys.version_info >= (3, 0):
        req = request
    for data_file, url in tasks_urls[task_name]:
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        _ = req.urlretrieve(url, os.path.join(data_dir, data_file))

download_data('conll2003', 'conll2003')

