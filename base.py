import tensorflow as tf
import numpy as np
import os 
import time
tf.compat.v1.enable_eager_execution() 

# file_path = 'hp2.txt'
file_path = tf.keras.utils.get_file('harry.txt', 'http://www.glozman.com/TextPages/Harry%20Potter%202%20-%20Chamber%20of%20Secrets.txt')

# tf.enable_eager_execution()

# reading data
text = open(file_path, 'r', encoding='utf-8').read()
print('Length of text: %d characters' %(len(text)))

# count the unique characers
uniqlo = sorted(set(text)) # creating as set, then sorting
print('%d unique characters' %(len(uniqlo)))

# creating mapping direction --> unique characters to indexes
uni2idx = {u: i for i, 
           u in enumerate(uniqlo)} # enumerate gives the set values a counter, turns uniqlo into enumerate class object
idx2uni = np.array(uniqlo) # create array
text_as_int = np.array([uni2idx[c] for c in text])

print('{')
for char,_ in zip(uni2idx, range(10)):
    print(' {:s}-->{:d},'.format(repr(char), uni2idx[char]))
print('...\n}')

print('{} --- characters mapped to int ---> {} \n'.format(repr(text[:13]), text_as_int[:13]))

seq_len = 100
examples_per_epoch = len(text)

chars = tf.data.Dataset.from_tensor_slices(text_as_int)

sequences = chars.batch(seq_len+1, drop_remainder=True)
#for item in sequences.take(5):
#    print(repr(''.join(idx2uni[item.numpy()])))
    
def split_that_shit_up(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_that_shit_up)
for input_example, target_example in dataset.take(1):
    print('Input data: ', repr(''.join(idx2uni[input_example.numpy()])))
    print('Target data:', repr(''.join(idx2uni[target_example.numpy()])))
    
BATCH_SIZE = 64
BUFFER_SIZE = 1000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# using only three layers (input, RNN, output) 
vocab_size = len(uniqlo)
embedding_dim = 256
rnn_units = 1024

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

model = build_model(
  vocab_size = len(uniqlo),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)

##################################################################3

model.summary()

for input_example_batch, target_example_batch in dataset.take(1):
  example_batch_predictions = model(input_example_batch)
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()

print("Input: \n", repr("".join(idx2uni[input_example_batch[0]])))
print()
print("Next Char Predictions: \n", repr("".join(idx2uni[sampled_indices ])))


def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

example_batch_loss  = loss(target_example_batch, example_batch_predictions)
# print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
# print("scalar_loss:      ", example_batch_loss.numpy().mean())

model.compile(optimizer='adam', loss=loss)

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

EPOCHS=30
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

tf.train.latest_checkpoint(checkpoint_dir)
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

def generate_text(model, start_string):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = 1000

  # Converting our start string to numbers (vectorizing)
  input_eval = [uni2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # using a categorical distribution to predict the character returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted character as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx2uni[predicted_id])

  return (start_string + ''.join(text_generated))

print(generate_text(model, start_string=u"Harry"))




