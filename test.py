import argparse
import os
import time
import numpy as np

import tensorflow as tf

from worker import Worker
import sys

from env_doom import Doom
from net import Net
from utils import print_net_params_number, preprocess

def choose_action(sess, env, network, p):
    return np.random.choice(np.arange(env.action_dim), p=p[0])


def play_episodes(sess, env, network):
    ep_reward = 0
    ep_step_count = 0

    s = env.reset()
    ep_frames = []
    ep_frames.append(s)
    s = preprocess(s)
    rnn_state = network.state_init

    while True:
        p, v, rnn_state = sess.run([
            network.policy,
            network.value,
            network.state_out
        ], {
            network.inputs: [s],
            network.state_in[0]: rnn_state[0],
            network.state_in[1]: rnn_state[1]
        })
        a = choose_action(sess, env, network, p)

        s1, r, d = env.step(a)
        if d:
            break
        ep_frames.append(s1)
        r /= 100.0  # scale rewards
        s1 = preprocess(s1)

        ep_reward += r
        s = s1
        ep_step_count += 1
        time.sleep(1)


def main(args):
    if args.save_path is not None and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    tf.reset_default_graph()
    
    # global_ep = tf.Variable(
    #     0, dtype=tf.int32, name='global_ep', trainable=False)

    env = Doom(visiable=True)

    global_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, 'global')

    network =  Net(env.state_dim, env.action_dim, 'global', None)



    saver = tf.train.Saver()

    # sys.exit('..................')

    with tf.Session() as sess:
        if args.model_path is not None:
            print('Loading model...')
            ckpt = tf.train.get_checkpoint_state(args.model_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sys.exit('No model path found...')
        # print_net_params_number()
        while True:
            play_episodes(sess, env, network)





if __name__ == '__main__':
    # ignore warnings by tensorflow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path', default=None,
        help='Whether to use a saved model. (*None|model path)')
    parser.add_argument(
        '--save_path', default='./a3c_doom/model/',
        help='Path to save a model during training.')
    parser.add_argument(
        '--save_every', default=50, help='Interval of saving model')
    parser.add_argument(
        '--max_ep_len', default=300, help='Max episode steps')
    parser.add_argument(
        '--max_ep', default=3000, help='Max training episode')
    # parser.add_argument(
    #     '--parallel', default=multiprocessing.cpu_count(),
    #     help='Number of parallel threads')
    main(parser.parse_args())