# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 16:32:48 2022

@author: zll
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
START_TAG = "<START>"
STOP_TAG = "<STOP>"

# %%


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def log_sum_exp(x):
    """calculate log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    """
    max_score = x.max(-1)[0]
    return max_score + (x - max_score.unsqueeze(-1)).exp().sum(-1).log()


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, batch_size):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.bs = batch_size

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)

        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        self.start_idx = self.tag_to_ix[START_TAG]
        self.stop_idx = self.tag_to_ix[STOP_TAG]
        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[self.start_idx, :] = -10000
        self.transitions.data[:, self.stop_idx] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, self.bs , self.hidden_dim//2 ),
                torch.randn(2, self.bs , self.hidden_dim//2))

    def _get_lstm_features(self, sentence):
        B, length = sentence.shape
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(B, length,  -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(B, length, self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _forward_alg(self, features):
        """calculate the partition function with forward algorithm.
        TRICK: log_sum_exp([x1, x2, x3, x4, ...]) = log_sum_exp([log_sum_exp([x1, x2]),
                                                                 log_sum_exp([x3, x4]), ...])

        :param features: features. [B, L, C]
        :return:    [B], score in the log space
        """
        B, L, C = features.shape
        scores = torch.full((B, C), -1e4,
                            device=features.device)  # [B, C]
        scores[:, self.start_idx] = 0.
        trans = self.transitions.unsqueeze(0)  # [1, C, C]

        # Iterate through the sentence
        for t in range(L):
            emit_score_t = features[:, t].unsqueeze(2)  # [B, C, 1]
            # [B, 1, C] + [1, C, C] + [B, C, 1] => [B, C, C]
            scores = scores.unsqueeze(1) + trans + emit_score_t
            scores = log_sum_exp(scores)  # [B, C]

        scores = log_sum_exp(scores + self.transitions[self.stop_idx])
        return scores

    def _score_sentence(self, features, tags):
        """Gives the score of a provided tag sequence

        :param features: [B, L, C]
        :param tags: [B, L]
        :return: [B] score in the log space
        """
        B, L, C = features.shape

        # emission score
        emit_scores = features.gather(
            dim=2, index=tags.unsqueeze(-1)).squeeze(-1)

        # transition score
        start_tag = torch.full((B, 1), self.start_idx,
                               dtype=torch.long, device=tags.device)
        tags = torch.cat([start_tag, tags], dim=1)  # [B, L+1]
        trans_scores = self.transitions[tags[:, 1:], tags[:, :-1]]

        score = (trans_scores + emit_scores).sum(1) + \
            self.transitions[self.stop_idx, tags[:, -1]]
        return score

    def _viterbi_decode(self, features):
        B, L, C = features.shape
        bps = torch.zeros(B, L, C, dtype=torch.long,
                          device=features.device)  # back pointers

        # Initialize the viterbi variables in log space
        max_score = torch.full(
            (B, C), -1e4, device=features.device)  # [B, C]
        max_score[:, self.start_idx] = 0

        for t in range(L):
            emit_score_t = features[:, t]  # [B, C]
            acc_score_t = max_score.unsqueeze(
                1) + self.transitions  # [B, C, C]= [B, 1, C] + [C, C]
            acc_score_t, bps[:, t, :] = acc_score_t.max(dim=-1)
            acc_score_t += emit_score_t
            max_score = acc_score_t

        # Transition to STOP_TAG
        max_score += self.transitions[self.stop_idx]
        best_score, best_tag = max_score.max(dim=-1)

        # Follow the back pointers to decode the best path.
        best_paths = []
        bps = bps.cpu().numpy()
        for b in range(B):
            best_tag_b = best_tag[b].item()

            best_path = [best_tag_b]
            for bps_t in reversed(bps[b]):
                best_tag_b = bps_t[best_tag_b]
                best_path.append(best_tag_b)
            # drop the last tag and reverse the left
            best_paths.append(best_path[-2::-1])
        return best_score, best_paths

    def forward(self, sentences, tags=None):
        """
        B: batch size, L: sequence length,
        param features: [B, L, C], batch of unary scores
        return:
            training:
                loss
            eval:
                best_score*best_paths
                best_score: [B]
                best_paths: [B, L]
        """
        features = self._get_lstm_features(sentences)

        if model.training:
            forward_score = self._forward_alg(features)
            gold_score = self._score_sentence(
                features, tags.long())
            loss = (forward_score - gold_score).mean()
            return loss
        else:
            return self._viterbi_decode(features)


# %%
if __name__ == "__main__":

    # Make up some training data
    training_data = [(
        "the journal reported  apple corporation made money".split(),
        "B  I O B I O O".split()
    ), (
        "georgia tech is a university in georgia".split(),
        "B I O O O O B".split()
    )]

    word_to_ix = {}
    for sentence, tags in training_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

    data_train = []
    for sentence, tags in training_data:
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
        data_train.append((sentence_in, targets))

    train_data = DataLoader(data_train, batch_size=2,
                            shuffle=True, num_workers=0)

    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix,
                       embedding_dim=5,  hidden_dim=4,batch_size=2)
    optimizer = optim.SGD(model.parameters(), lr=0.05, weight_decay=1e-4)

    # %%
    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    model.train()
    for epoch in range(300):
        for sentence_in, targets in train_data:
            model.zero_grad()
            loss = model(sentence_in, targets)
            loss.backward()
            optimizer.step()
            print(loss)
    # %%
    # # Check predictions after training
    model.eval()
    with torch.no_grad():
        # precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
        print(model(sentence_in))
    # # We got it!
