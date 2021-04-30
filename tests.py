import unittest

from models import TFPanGuAlphaLMHeadModel
from transformers import GPT2Config
import tensorflow as tf


class TFPanGuAlphaLMHeadModelTest(unittest.TestCase):
    n_layer = 2
    n_embd = 512
    n_head = 4
    vocab_size = 100

    def setUp(self):
        config = GPT2Config(
            n_layer=self.n_layer,
            n_embd=self.n_embd,
            n_head=self.n_head,
            vocab_size=self.vocab_size,
        )
        self.model = TFPanGuAlphaLMHeadModel(config)

    def test_output(self):
        ids = [1, 2, 3, 4]
        input = tf.constant([ids])
        res = self.model(input)
        assert res[0].shape == (1, len(ids), self.vocab_size)
