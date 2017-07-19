"""
run.py

Core script for building, training, and evaluating a Relation Network on the bAbI Tasks Dataset (10k Joint-Training).
"""
from model.relation_network import RelationNetwork
from preprocessor.reader import parse
import os
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# Run Parameters
tf.app.flags.DEFINE_string("mode", "valid", "Mode to run - choose from [train, valid, test].")
tf.app.flags.DEFINE_string("ckpt_dir", "ckpt/", "Directory to store checkpoints, log files.")

# Eval Mode Parameters
tf.app.flags.DEFINE_integer("task", 1, "Task to evaluate trained model on. [0] - Means evaluate on all!")


def main(_):
    if FLAGS.mode == "train":
        # Parse Data
        print '[*] Parsing Data!'
        S, S_len, Q, Q_len, A, word2id, a_word2id = parse("train",
                                                          pik_path=os.path.join(FLAGS.ckpt_dir, 'train', 'train.pik'),
                                                          voc_path=os.path.join(FLAGS.ckpt_dir, 'voc.pik'))

        # Initialize Model
        print '[*] Creating Model!'
        rn = RelationNetwork(S, S_len, Q, Q_len, A, word2id, a_word2id, restore=False)

        # Train for 50 Epochs
        print '[*] Training Model!'
        rn.fit(epochs=50)

    elif FLAGS.mode == "valid":
        # Restore Model
        print '[*] Restoring Model!'
        S, S_len, Q, Q_len, A, word2id, a_word2id = parse("train",
                                                          pik_path=os.path.join(FLAGS.ckpt_dir, 'train', 'train.pik'),
                                                          voc_path=os.path.join(FLAGS.ckpt_dir, 'voc.pik'))
        rn = RelationNetwork(S, S_len, Q, Q_len, A, word2id, a_word2id,
                             restore=tf.train.latest_checkpoint(os.path.join(FLAGS.ckpt_dir, 'ckpts')))

        if FLAGS.task == 0:
            print '[*] Validating on all Tasks!'
            for task in range(1, 21):
                print '[*] Loading Task %d!' % task
                S, S_len, Q, Q_len, A, _, _ = parse("valid",
                                                    pik_path=os.path.join(FLAGS.ckpt_dir, 'valid',
                                                                          'valid_%d.pik' % task),
                                                    voc_path=os.path.join(FLAGS.ckpt_dir, 'voc.pik'), task_id=task)
                accuracy = rn.eval(S, S_len, Q, Q_len, A)
                print 'Task %d\tAccuracy: %.3f' % (task, accuracy)

        else:
            task = FLAGS.task
            print '[*] Validating on Task %d' % task
            S, S_len, Q, Q_len, A, _, _ = parse("valid",
                                                pik_path=os.path.join(FLAGS.ckpt_dir, 'valid', 'valid_%d.pik' % task),
                                                voc_path=os.path.join(FLAGS.ckpt_dir, 'voc.pik'), task_id=task)
            accuracy = rn.eval(S, S_len, Q, Q_len, A)
            print 'Task %d\tAccuracy: %.3f' % (task, accuracy)

    elif FLAGS.mode == "test":
        # Restore Model
        print '[*] Restoring Model!'
        S, S_len, Q, Q_len, A, word2id, a_word2id = parse("train",
                                                          pik_path=os.path.join(FLAGS.ckpt_dir, 'train', 'train.pik'),
                                                          voc_path=os.path.join(FLAGS.ckpt_dir, 'voc.pik'))
        rn = RelationNetwork(S, S_len, Q, Q_len, A, word2id, a_word2id,
                             restore=tf.train.latest_checkpoint(os.path.join(FLAGS.ckpt_dir, 'ckpts')))

        if FLAGS.task == 0:
            print '[*] Testing on all Tasks!'
            for task in range(1, 21):
                print '[*] Loading Task %d!' % task
                S, S_len, Q, Q_len, A, _, _ = parse("test",
                                                    pik_path=os.path.join(FLAGS.ckpt_dir, 'test',
                                                                          'test_%d.pik' % task),
                                                    voc_path=os.path.join(FLAGS.ckpt_dir, 'voc.pik'), task_id=task)
                accuracy = rn.eval(S, S_len, Q, Q_len, A)
                print 'Task %d\tAccuracy: %.3f' % (task, accuracy)

        else:
            task = FLAGS.task
            print '[*] Testing on Task %d' % task
            S, S_len, Q, Q_len, A, _, _ = parse("test",
                                                pik_path=os.path.join(FLAGS.ckpt_dir, 'test', 'test_%d.pik' % task),
                                                voc_path=os.path.join(FLAGS.ckpt_dir, 'voc.pik'), task_id=task)
            accuracy = rn.eval(S, S_len, Q, Q_len, A)
            print 'Task %d\tAccuracy: %.3f' % (task, accuracy)
    else:
        print "Unsupported Mode, use one of [train, valid, test]"
        raise UserWarning


if __name__ == "__main__":
    tf.app.run()
