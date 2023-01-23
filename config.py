pickle_path = "features/modaudiotext_red5_mod.pkl"
length_train = "features/length_train.npy"
length_test = "features/length_test.npy"
train_npy_image_path = "features/train_image.npy"
train_npy_text_path = "features/train_text.npy"
train_npy_audio_path = "features/train_audio.npy"
train_npy_label_path = "features/train_labels.npy"
train_text_csv_path = "features/train_text.csv"
test_npy_image_path = "features/test_image.npy"
test_npy_audio_path = "features/test_audio.npy"
test_npy_text_path = "features/test_text.npy"
test_npy_label_path = "features/test_labels.npy"
test_text_csv_path = "features/test_text.csv"
unimodal_folder = "unimodal/"
NUM_HIDDEN_LAYERS = 2
NUM_ATTENTION_HEADS = 10
HIDDEN_SIZE = 140
HIDDEN_SIZE_TRANS = 120
AUDIO_DIM = 125
TEXT_DIM = 300
IMAGE_DIM = 256
GRU_DIM = 140
USE_TEXT = True
USE_AUDIO = True
USE_IMAGE = False
USE_GRU = True
USE_EE = False
USE_TRANS = False
USE_ONLY_LSTM = False
USE_HYBRID = False
LR = 0.0003
EPOCHS = 25