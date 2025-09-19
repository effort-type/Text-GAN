import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gensim
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers import LeakyReLU
from keras.layers import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import backend as K
from keras.activations import softmax
import tensorflow as tf
from gensim.models import FastText

with open("test.txt", "r", encoding="UTF-8") as f:
    lines = f.readlines()
raw_data = pd.DataFrame({"text": [line.strip() for line in lines if line.strip()]})

global_dim = 64
glover_len = 7
data = []
x_point = []
y_point = []
y_point2 = []

sentence_1_point = []
sentence_3_point = []
sentence_5_point = []
sentence_7_point = []

for i in raw_data["text"]:
    split_text = i.split(" ")

    if len(split_text) < glover_len:
        pass
    else:
        cnt = 0
        data_temp = []
        for j in split_text:
            data_temp.append(j)

            cnt += 1
            if cnt == 7:
                cnt = 0
                data.append(data_temp)
                data_temp = []

        if cnt > 0 and cnt < 7:
            for k in range(7 - cnt):
                data_temp.append("<pad>")

            data.append(data_temp)

# model = FastText.load_fasttext_format('raw/kor_model_100d_4min_5w.bin')
model = FastText(data, vector_size=global_dim, window=5, min_count=1)

sentence_to_word_vec = np.zeros(shape=(len(data), glover_len, global_dim))
for sentence_index, i in enumerate(data):
    temp_list = np.zeros(shape=(glover_len, global_dim))
    for idx, j in enumerate(i):
        try:
            temp_list[idx] = np.array([model.wv.get_vector(j)])
        except:
            temp_list[idx] = np.array([model.wv.get_vector(".")])
    temp_list = np.reshape(temp_list, (1, glover_len, global_dim))
    sentence_to_word_vec[sentence_index] = temp_list

sentence_to_word_vec.shape

sentence_to_word_vec = np.expand_dims(sentence_to_word_vec, axis=-1)


class GAN2vec:
    def __init__(self):
        # Input shape
        self.sentence_length = glover_len
        self.word_dimension = global_dim
        self.channels = 1
        self.sentence_shape = (self.sentence_length, self.word_dimension, self.channels)
        self.latent_dim = 128  # 증가된 latent dimension

        # 서로 다른 학습률 설정
        d_optimizer = Adam(
            learning_rate=0.0001, beta_1=0.5, beta_2=0.999
        )  # ver2에는 0.0002 (판별자가 2배 빠르게 학습)
        g_optimizer = Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.999)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss="binary_crossentropy", optimizer=d_optimizer)

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        sentence = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(sentence)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss="binary_crossentropy", optimizer=g_optimizer)

    def build_generator(self):
        model = Sequential()

        # 7 * 64 = 448 차원으로 시작
        model.add(Dense(7 * 64 * 4))  # 1792
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Dropout(0.3))
        model.add(Reshape((7, 64, 4)))

        # Conv2DTranspose로 채널 수 줄이기
        model.add(Conv2DTranspose(2, kernel_size=(3, 3), strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(Conv2DTranspose(1, kernel_size=(3, 3), strides=1, padding="same"))
        model.add(Activation("tanh"))  # tanh 활성화 함수 사용

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        sentence = model(noise)

        return Model(noise, sentence)

    def build_discriminator(self):
        model = Sequential()

        model.add(
            Conv2D(
                32,
                kernel_size=(3, 3),
                strides=1,
                input_shape=self.sentence_shape,
                padding="same",
            )
        )
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, kernel_size=(3, 3), strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=(3, 3), strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation="sigmoid"))
        model.summary()

        sentence = Input(shape=self.sentence_shape)
        validity = model(sentence)

        return Model(sentence, validity)

    def pretrain_D(self, epochs, batch_size=128):
        X_train = sentence_to_word_vec

        print("pretraining D")
        for epoch in range(epochs):
            print("{}epochs".format(epoch))

            # 더 부드러운 라벨 스무딩
            valid = np.ones((batch_size, 1)) * 0.95
            fake = np.ones((batch_size, 1)) * 0.05

            idx = np.random.randint(0, X_train.shape[0], batch_size)
            sentences = X_train[idx]

            # 더 다양한 노이즈 생성
            noise = np.random.uniform(-1, 1, (batch_size, self.latent_dim))
            gen_sentences = self.generator.predict(noise, verbose=0)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(sentences, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_sentences, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    def train(self, epochs, batch_size=128, save_interval=50):
        # Load the dataset
        X_train = sentence_to_word_vec

        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator (2회)
            # ---------------------
            for _ in range(2):  # Discriminator를 더 자주 훈련
                # 더 부드러운 라벨 스무딩
                valid = np.ones((batch_size, 1)) * 0.95
                fake = np.ones((batch_size, 1)) * 0.05

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]

                # 더 다양한 노이즈 생성 (uniform distribution)
                noise = np.random.uniform(-1, 1, (batch_size, self.latent_dim))
                gen_imgs = self.generator.predict(noise, verbose=0)

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator (1회)
            # ---------------------
            # 새로운 노이즈로 Generator 훈련
            noise = np.random.uniform(-1, 1, (batch_size, self.latent_dim))
            valid_for_gen = np.ones((batch_size, 1))  # Generator는 진짜라고 속이려고 함

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_for_gen)

            # Plot the progress
            print(
                "%d [D loss: %f, acc.: %.2f%%] [G loss: %f]"
                % (epoch, d_loss[0], 100 * d_loss[1], g_loss)
            )

            x_point.append(epoch)
            y_point2.append(d_loss[0])
            y_point.append(g_loss)

            # 매 epoch마다 문장 생성 확인
            if epoch % 10 == 0:  # 10 epoch마다만 출력
                self.show_sentence(epoch)

            if epoch % 100 == 0:
                self.generator.save("gan2vec_model_" + str(epoch) + ".keras")

    def show_sentence(self, epoch):
        r, c = 5, 1  # 5개 문장만 생성
        noise = np.random.uniform(-1, 1, (r * c, self.latent_dim))
        gen_sentence = self.generator.predict(noise, verbose=0)
        test = np.squeeze(gen_sentence)

        sentences = []
        sum_1 = 0.0
        sum_3 = 0.0
        sum_5 = 0.0
        sum_7 = 0.0

        for i, sentence_vec in enumerate(test):
            sentence = ""
            cnt = 0
            for j in sentence_vec:
                try:
                    temp = model.wv.similar_by_vector(j)
                    word = temp[0][0]
                    similarity = temp[0][1]
                    sentence = sentence + word + " "

                    if cnt < 1:
                        sum_1 += similarity
                    if cnt < 3:
                        sum_3 += similarity
                    if cnt < 5:
                        sum_5 += similarity

                    sum_7 += similarity
                    cnt += 1
                except:
                    sentence = sentence + "<unk> "
                    cnt += 1

            sentences.append(sentence.strip())

            # 첫 번째 문장만 출력
            if i == 0:
                print(f"Epoch {epoch}: {sentence.strip()}")
                break

        # 평균 유사도 계산
        if cnt > 0:
            sum_7 = sum_7 / 7
            sum_5 = sum_5 / 5
            sum_3 = sum_3 / 3
            sum_1 = sum_1 / 1

            sentence_7_point.append(sum_7)
            sentence_5_point.append(sum_5)
            sentence_3_point.append(sum_3)
            sentence_1_point.append(sum_1)

    def predict(self, num_sentences=5):
        noise = np.random.uniform(-1, 1, (num_sentences, self.latent_dim))
        gen_sentence = self.generator.predict(noise, verbose=0)
        test = np.squeeze(gen_sentence)

        sentence_list = []
        for i in test:
            sentence = ""
            for j in i:
                try:
                    temp = model.wv.similar_by_vector(j)
                    sentence = sentence + temp[0][0] + " "
                except:
                    sentence = sentence + "<unk> "
            sentence_list.append(sentence.strip())
        return sentence_list


# 모델 생성 및 훈련
gan2vec = GAN2vec()
gan2vec.pretrain_D(epochs=2)  # 사전 훈련 조금 더
gan2vec.train(epochs=1000, batch_size=128, save_interval=50)
gan2vec.generator.save("gan2vec_model_final.keras")

# 테스트 문장 생성
print("\n=== 최종 생성된 문장들 ===")
for i, sentence in enumerate(gan2vec.predict(10)):
    print(f"{i+1}: {sentence}")

# 손실 함수 그래프
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.ylabel("loss")
plt.xlabel("epochs")
plt.plot(x_point, y_point, label="G loss")
plt.plot(x_point, y_point2, label="D loss")
plt.legend()
plt.title("Training Loss")

plt.subplot(1, 2, 2)
plt.plot(x_point[::10], sentence_7_point, label="word 7")
plt.plot(x_point[::10], sentence_5_point, label="word 5")
plt.plot(x_point[::10], sentence_3_point, label="word 3")
plt.plot(x_point[::10], sentence_1_point, label="word 1")
plt.legend()
plt.title("Word Similarity")
plt.xlabel("epochs")
plt.ylabel("similarity")

plt.tight_layout()
plt.show()
