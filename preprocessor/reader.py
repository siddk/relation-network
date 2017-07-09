"""
reader.py

Reads in and processes the bAbI Task files, serializing them for future reference.
"""
import numpy as np
import os
import pickle
import re

PAD_ID, PAD_TOKEN = 0, "<<pad>>"
SPLIT_RE = re.compile('(\W+)?')
tokenize = lambda x: [token.strip().lower() for token in re.split(SPLIT_RE, x) if token.strip()]


def parse(mode, pik_path, voc_path, task_id=1):
    if mode == 'train':
        if os.path.exists(pik_path):
            with open(pik_path, 'r') as f:
                S, S_len, Q, Q_len, A = pickle.load(f)
            with open(voc_path, 'r') as f:
                word2id, a_word2id, _, _ = pickle.load(f)
            return S, S_len, Q, Q_len, word2id, a_word2id
        else:
            # Parse into Stories, Queries, Answers
            all_stories = []
            for i in range(1, 21):
                with open('data/qa%d_train.txt' % i, 'r') as f:
                    all_stories += parse_stories(f.readlines())

            # Build Vocabulary
            vocab, answer_vocab, max_s, max_q = set(), set(), 0, 0
            for s, _, q, _, a in all_stories:
                for line in s:
                    vocab.update(line)
                    max_s = max(max_s, len(line))
                vocab.update(q)
                max_q = max(max_q, len(q))
                answer_vocab.add(a)

            word2id = {w: i for i, w in enumerate([PAD_TOKEN] + list(vocab))}
            a_word2id = {w: i for i, w in enumerate(list(answer_vocab))}
            S, S_len, Q, Q_len, A = vectorize(all_stories, word2id, a_word2id, max_s, max_q)

            with open(pik_path, 'w') as f:
                pickle.dump((S, S_len, Q, Q_len, A), f)

            with open(voc_path, 'w') as f:
                pickle.dump((word2id, a_word2id, max_s, max_q), f)

            return S, S_len, Q, Q_len, A, word2id, a_word2id

    elif mode == 'valid' or mode == 'test':
        # Load Vocab, Metadata
        with open(voc_path, 'r') as f:
            word2id, a_word2id, max_s, max_q = pickle.load(f)

        # Parse Task
        with open('data/qa%d_%s.txt' % (task_id, mode), 'r') as f:
            all_stories = parse_stories(f.readlines())

        # Vectorize + Serialize
        S, S_len, Q, Q_len, A = vectorize(all_stories, word2id, a_word2id, max_s, max_q)
        with open(pik_path, 'w') as f:
            pickle.dump((S, S_len, Q, Q_len, A), f)

        return S, S_len, Q, Q_len, A, word2id, a_word2id


def parse_stories(lines, truncate=20):
    """
    Boilerplate bAbI Task Format Parser.
    """
    stories, story = [], []
    for line in lines:
        line = line.strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            query, answer, supporting = line.split('\t')
            query = tokenize(query)
            sub_story = [x for x in story if x][-truncate:]
            sub_story_len = [len(x) for x in sub_story]
            stories.append((sub_story, sub_story_len, query, len(query), answer.lower()))
            story.append('')
        else:
            sentence = tokenize(line)
            story.append(sentence)
    return stories


def vectorize(s, word2id, answer_word2id, max_s, max_q, s_len=20):
    """
    Vectorize raw stories, using the provided general and answer-specific vocabularies.
    """
    S, Q, A = np.zeros([len(s), s_len, max_s], int), np.zeros([len(s), max_q], int), np.zeros([len(s)], int)
    S_len, Q_len = np.zeros([len(s), s_len], int), np.zeros([len(s)], int)
    for i in range(len(s)):
        story, story_len, query, query_len, answer = s[i]
        for j in range(len(story)):
            for k in range(len(story[j])):
                S[i][j][k] = word2id[story[j][k]]
        for j in range(len(story_len)):
            S_len[i][j] = story_len[j]
        for j in range(len(query)):
            Q[i][j] = word2id[query[j]]
        Q_len[i] = query_len
        A = answer_word2id[answer]
    return S, S_len, Q, Q_len, A
