# lstm-crf-with-batch_size
 
This code combines https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html with https://github.com/jidasheng/bi-lstm-crf. 

It's easy to understand.

It could realize the pos-tagging function with batched data.

Note, curently the model only deal with the input data  with the same length.

Bilstm_crf.py uses the bi-directional lstm model and lstm_crf.py uses the single-directional lstm model.
