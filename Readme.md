# End-to-end Speech Recognition using the Deepspeech2 Architecture implemented with Tensorflow
[![StreamlitCloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/to-schi/asr-deepspeech2-webdemo/main)


#
The engine's architecture is similar to [Deepspeech2](https://arxiv.org/abs/1512.02595) and includes a conversion of audio data to mel spectrograms, char-tokenization of the transcription, a tensorflow input pipeline, a recurrent neural network (RNN) and CTC-loss/decoder-functions.
A [demo-app](https://github.com/to-schi/asr-deepspeech2-webdemo) of the speech recognition has been deployed on [StreamlitCloud](https://share.streamlit.io/to-schi/asr-deepspeech2-webdemo/main).

![DeepSpeech2](./img/DeepSpeech2.drawio.svg)
![Decoder](./img/RNN%2BCTC.drawio.svg)

The model was trained on the full [LibriSpeech](https://www.openslr.org/12/) dataset with 960 hours of reading from audiobooks. The preparation of this dataset was surprisingly complicated, as each audiobook has it's own folder and transcription file. On the other hand this is of advantage, because Google Drive produces "Input/output" errors when handling large amounts of files in one folder. A list of paths had to be compiled for the audio files and the transciptions had to be merged to one dataframe. Data preparation and a quick exploratory data analysis (EDA) can be seen in the [asr_rnn_data_preparation](https://github.com/to-schi/speech-recognition-from-scratch/blob/main/asr_rnn_data_preparation.ipynb) notebook [(open in colab)](https://colab.research.google.com/github/to-schi/speech-recognition-from-scratch/blob/main/asr_rnn_data_preparation.ipynb).

Before training, outliers of duration, speed and transcription length were removed from the training data and the remaining samples were sorted by length to produce a higher batch-consistency. Sorting by the length of transcription performed better than by duration.

![distribution of duration](img/dist_duration.png)
![distributionn of transcription length](./img/dist_char-length.png)
![distributionn of speed](./img/dist_speed.png)

The audio data was converted to mel-scaled spectrograms. The [mel scale](https://en.wikipedia.org/wiki/Mel_scale) (melody scale) takes into account, that human hearing ability varies over the frequency range. The switch from standard spectrograms to mel spectrograms improved the word error rate (WER) on the validation data by about 10 percent.

Short-time Fourier transform spectrogram  
![stft-spectrogram](img/stft-spectrogram.jpg)

Mel spectrogram  
![mel-spectrogram](img/mel-spectrogram.jpg)

Because spectrograms are images, convolutional layers can be used to extract features from them. Additionally speech is time-series data, so it is beneficial to use bidirectional RNN layers such as gated recurrent units (GRU) to capture time-frequency patterns from the features detected by the convolutional layers. The final RNN has 28,100,862 trainable parameters.


## Training
For the [training](https://colab.research.google.com/github/to-schi/speech-recognition-from-scratch/blob/main/asr_rnn_training.ipynb) I used Google Colab Pro+ and luckily sometimes a GPU with 40GB RAM got connected (A100-SXM4-40GB). The connection will break at least untill 24 hours and has to be restarted. Overall the model trained more than 100 epochs.

![training](./img/history_plot.svg)

[spectrogram augmentation](https://arxiv.org/abs/1904.08779) reduced overfitting significantly. Also parts of the ctc-loss function were changed, which noticeably accelerated the learning of the model especially during the first epochs. Decreasing the learning rate over time is of equal importance. This could have been done with a learning rate schedule, but unfortunately the google colab runtime disconnected usually after 4 epochs (24 hours). The manual schedule was like this:

| epoch  | learning rate   |
| ------ | --------------- |
| 1-3    | 0.0002 (2e-4)   |
| 4-6    | 0.00015 (15e-4) |
| 7-10   | 0.0001 (1e-4)   |
| 11-35  | 0.00008 (8e-5)  |
| 36-49  | 0.00005 (5e-5)  |
| 50-55  | 0.00003 (3e-5)  |
| 56-71  | 0.00001 (1e-5)  |
| 72-79  | 0.000008 (8e-6) |
| 80-85  | 0.000005 (5e-6) |
| 86-96  | 0.000003 (3e-6) |
| 97-105 | 0.000001 (1e-6) |

The word-error-rate improved to **10,8 %** on the Librispeech test-clean-dataset.

### Language Model
To be able to use pyctcdecode as the decoder with a kenlm-model, the loss function (tf.keras.backend.ctc_decode) has to be adapted and the blank index set to "0" (oov_token="" is also index "0" of the vocbulary). This way the output classes are equal to the vocabulary list.

With the use of the language model the performance improved significantly. It works as a scorer for letter-sequence probabilities in the decoder ([pyctcdecode](https://github.com/kensho-technologies/pyctcdecode)). Different language models with 3 to 6 grams were created with KenLM from 400k, 500k and 600k of most common words in the Libri Speech vocbulary. A bigger model raises the performance a little bit while taking slightly more compute time and significantly more disk space. 
The best language model improves the word-error-rate to **6.3 %** on the Librispeech test-clean-dataset.

### Evaluation
As the model is trained on read speech, it does not perform as good on spontaneous speech or especially on singing. To improve the robustness of the speech recognition, data with spontaneous speech could be added or the current dataset could be further augmented by the addition of noise and changes of speed and pitch.
