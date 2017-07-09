"""
run.py

Core script for building, training, and evaluating a Relation Network on the bAbI Tasks Dataset (10k Joint-Training).
"""
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# Run Parameters
tf.app.flags.DEFINE_string("mode", "train", "Mode to run - choose from [train, eval].")
tf.app.flags.DEFINE_string("ckpt_dir", "ckpt/", "Directory to store checkpoints, log files.")

# Train Mode Parameters



def main(_):
    if FLAGS.mode == "train":
        pass
    elif FLAGS.mode == "eval":
        pass
    else:
        print "Unsupported Mode, use one of [train, eval]"
        raise UserWarning



if __name__ == "__main__":
    tf.app.run()