import tensorflow as tf

from config import DEFAULT

def build_lstm():
    tf.reset_default_graph()
    lstm_graph = tf.Graph()
    config = DEFAULT

    with lstm_graph.as_default():
        learning_rate = tf.placeholder(tf.float32,None, name="learning rate")
        inputs = tf.placeholder(tf.float32, [None, config.num_steps, config.input_size], name="inputs")
        targets = tf.placeholder(tf.float32, [None, config.input_size], name="targets")

        def createOneBlock():
            lstm_block = tf.nn.rnn_cell.LSTMCell(config.lstm_size, state_is_tuple=True)
            if config.keep_prob < 1.0:
                lstm_block = tf.nn.rnn_cell.DropoutWrapper(lstm_block,output_keep_prob=config.keep_prob)
            return lstm_block

        block = tf.nn.rnn_cell.MultiRNNCell(
            [createOneBlock() for _ in range(config.num_layers)],
            state_is_tuple=True
        ) if config.num_layers > 1 else createOneBlock()

        val, _ = tf.nn.dynamic_rnn(block, inputs, dtype=tf.float32, scope="hamza_rnn")
        val = tf.transpose(val, [1, 0, 2])

        with tf.name_scope("output_layer"):
            last = tf.gather(val, int(val.get_shape()[0])-1, name="lstm_output")
            weight = tf.Variable(tf.random_normal([config.lstm_size, config.input_size]), name="hamza_weights")
            bias = tf.Variable(tf.constant(0.1, shape=[config.input_size]), name="hamza_biases")
            prediction = tf.matmul(last, weight) + bias

            tf.summary.histogram("lstm_output", last)
            tf.summary.histogram("weights", weight)
            tf.summary.histogram("biases", bias)

        with tf.name_scope("train"):
            loss = tf.reduce_mean(tf.square(prediction - targets), name="lose_mse")
            # Adam optimizer paper by Kingma and Ba, Link: https://arxiv.org/pdf/1412.6980v8.pdf
            optimizer = tf.train.AdamOptimizer(learning_rate)
            minimize = optimizer.minimize(loss, name="adam_optimized_loss")
            tf.summary.scalar("loss_mse", loss)

        for op in [prediction, loss]:
            tf.add_to_collection('ops_to_restore', op)

    return lstm_graph