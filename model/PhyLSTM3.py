"""
@author: Ruiyang Zhang
Email: ruiyang.zhang@hotmail.com
"""

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dropout, Dense
from keras.optimizers import RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam, SGD, TFOptimizer
from keras.layers import CuDNNLSTM, Flatten, LSTM, Reshape, BatchNormalization, Activation, UpSampling1D, ZeroPadding1D, PReLU
from random import shuffle
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # CPU:-1; GPU0: 1; GPU1: 0;


class DeepPhyLSTM:
    # Initialize the class
    def __init__(self, eta, eta_t, g, ag, ag_c, lift, Phi_t, save_path):

        # data
        self.eta = eta
        self.eta_t = eta_t
        self.g = g
        self.ag = ag
        self.lift = lift
        self.ag_c = ag_c
        self.Phi_t = Phi_t
        self.save_path = save_path

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        # placeholders for data
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.best_loss = tf.placeholder(tf.float32, shape=[])
        self.eta_tf = tf.placeholder(tf.float32, shape=[None, None, self.eta.shape[2]])
        self.eta_t_tf = tf.placeholder(tf.float32, shape=[None, None, self.eta.shape[2]])
        self.g_tf = tf.placeholder(tf.float32, shape=[None, None, self.eta.shape[2]])
        self.ag_tf = tf.placeholder(tf.float32, shape=[None, None, 1])
        self.lift_tf = tf.placeholder(tf.float32, shape=[None, None, 1])
        self.ag_c_tf = tf.placeholder(tf.float32, shape=[None, None, 1])
        self.Phi_tf = tf.placeholder(tf.float32, shape=[None, self.eta.shape[1], self.eta.shape[1]])

        # physics informed neural networks
        self.eta_pred, self.eta_t_pred, self.eta_tt_pred, self.eta_dot_pred, self.g_pred, self.g_t_pred = self.net_structure(self.ag_tf)
        self.eta_t_pred_c, self.eta_dot_pred_c, self.g_t_pred_c, self.g_dot_pred_c, self.lift_c_pred = self.net_f(self.ag_c_tf)

        # loss
        # for measurements
        self.loss_u = tf.reduce_mean(tf.square(self.eta_tf - self.eta_pred))
        self.loss_udot = tf.reduce_mean(tf.square(self.eta_t_tf - self.eta_dot_pred))
        self.loss_g = tf.reduce_mean(tf.square(self.g_tf - self.g_pred))
        # for collocations
        self.loss_ut_c = tf.reduce_mean(tf.square(self.eta_t_pred_c - self.eta_dot_pred_c))
        self.loss_gt_c = tf.reduce_mean(tf.square(self.g_t_pred_c - self.g_dot_pred_c))
        self.loss_e = tf.reduce_mean(tf.square(tf.matmul(self.lift_tf, tf.ones([self.lift.shape[0], 1, self.eta.shape[2]], dtype=tf.float32)) - self.lift_c_pred))

        self.loss = self.loss_u + self.loss_udot + self.loss_ut_c + self.loss_gt_c + self.loss_e

        # optimizers
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 20000,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.train_op = self.optimizer_Adam.minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.saver = tf.train.Saver()

    def LSTM_model(self, X):
        model = Sequential()
        model.add(CuDNNLSTM(100, return_sequences=True, stateful=False, input_shape=(None, 1)))
        model.add(Activation('relu'))
        model.add(CuDNNLSTM(100, return_sequences=True, stateful=False))
        model.add(Activation('relu'))
        model.add(CuDNNLSTM(100, return_sequences=True, stateful=False))
        model.add(Activation('relu'))
        # model.add(Dense(100))
        model.add(Dense(3*self.eta.shape[2]))
        y = model(X)
        return y

    def LSTM_model_f(self, X):
        model = Sequential()
        model.add(CuDNNLSTM(100, return_sequences=True, stateful=False, input_shape=(None, 3*self.eta.shape[2])))
        model.add(Activation('relu'))
        model.add(CuDNNLSTM(100, return_sequences=True, stateful=False))
        model.add(Activation('relu'))
        model.add(CuDNNLSTM(100, return_sequences=True, stateful=False))
        model.add(Activation('relu'))
        # model.add(Dense(100))
        model.add(Dense(self.eta.shape[2]))
        y = model(X)
        return y

    def LSTM_model_g(self, X):
        model = Sequential()
        model.add(CuDNNLSTM(100, return_sequences=True, stateful=False, input_shape=(None, 2*self.eta.shape[2])))
        model.add(Activation('relu'))
        model.add(CuDNNLSTM(100, return_sequences=True, stateful=False))
        model.add(Activation('relu'))
        model.add(CuDNNLSTM(100, return_sequences=True, stateful=False))
        model.add(Activation('relu'))
        # model.add(Dense(100))
        model.add(Dense(self.eta.shape[2]))
        y = model(X)
        return y

    def net_structure(self, ag):
        output = self.LSTM_model(ag)
        eta = output[:, :, 0:self.eta.shape[2]]
        eta_dot = output[:, :, self.eta.shape[2]:2*self.eta.shape[2]]
        g = output[:, :, 2*self.eta.shape[2]:]

        eta_t = tf.matmul(self.Phi_tf, eta)
        eta_tt = tf.matmul(self.Phi_tf, eta_dot)
        g_t = tf.matmul(self.Phi_tf, g)

        return eta, eta_t, eta_tt, eta_dot, g, g_t

    def net_f(self, ag):
        eta, eta_t, eta_tt, eta_dot, g, g_t = self.net_structure(ag)
        f = self.LSTM_model_f(tf.concat([eta, eta_dot, g], 2))
        lift = eta_tt + f

        eta_dot1 = eta_dot[:, :, 0:1]
        g_dot = self.LSTM_model_g(tf.concat([eta_dot1, g], 2))
        return eta_t, eta_dot, g_t, g_dot, lift
    
    def train(self, num_epochs, learning_rate, bfgs):

        Loss_u = []
        Loss_udot = []
        Loss_ut_c = []
        Loss_gt_c = []
        Loss_g = []
        Loss_e = []
        Loss = []
        Loss_val = []
        best_loss = 100

        for epoch in range(num_epochs):

            Ind = list(range(self.ag.shape[0]))
            shuffle(Ind)
            ratio_split = 0.8
            Ind_tr = Ind[0:round(ratio_split * self.ag.shape[0])]
            Ind_val = Ind[round(ratio_split * self.ag.shape[0]):]

            self.ag_tr = self.ag[Ind_tr]
            self.eta_tr = self.eta[Ind_tr]
            self.eta_t_tr = self.eta_t[Ind_tr]
            self.g_tr = self.g[Ind_tr]
            self.ag_val = self.ag[Ind_val]
            self.eta_val = self.eta[Ind_val]
            self.eta_t_val = self.eta_t[Ind_val]
            self.g_val = self.g[Ind_val]

            start_time = time.time()

            tf_dict = {self.eta_tf: self.eta_tr, self.eta_t_tf: self.eta_t_tr, self.g_tf: self.g_tr,
                       self.ag_tf: self.ag_tr, self.lift_tf: self.lift, self.ag_c_tf: self.ag_c,
                       self.Phi_tf: self.Phi_t, self.learning_rate: learning_rate}

            tf_dict_val = {self.eta_tf: self.eta_val, self.eta_t_tf: self.eta_t_val, self.g_tf: self.g_val,
                           self.ag_tf: self.ag_val, self.lift_tf: self.lift, self.ag_c_tf: self.ag_c,
                           self.Phi_tf: self.Phi_t, self.learning_rate: learning_rate}

            self.sess.run(self.train_op, tf_dict)

            loss_value, learning_rate_value = self.sess.run([self.loss, self.learning_rate], tf_dict)
            loss_val_value = self.sess.run(self.loss, tf_dict_val)

            Loss_u.append(self.sess.run(self.loss_u, tf_dict))
            Loss_udot.append(self.sess.run(self.loss_udot, tf_dict))
            Loss_g.append(self.sess.run(self.loss_g, tf_dict))
            Loss_ut_c.append(self.sess.run(self.loss_ut_c, tf_dict))
            Loss_gt_c.append(self.sess.run(self.loss_gt_c, tf_dict))
            Loss_e.append(self.sess.run(self.loss_e, tf_dict))
            Loss.append(self.sess.run(self.loss, tf_dict))
            Loss_val.append(self.sess.run(self.loss, tf_dict_val))

            # Save the best val model
            if loss_val_value < best_loss and loss_val_value < 1e-2:
                best_loss = loss_val_value

                # self.saver.save(sess=self.sess, save_path=self.save_path)

            elapsed = time.time() - start_time
            print('Epoch: %d, Loss: %.3e, Loss_val: %.3e, Best_loss: %.3e, Time: %.2f, Learning Rate: %.3e'
                  % (epoch, loss_value, loss_val_value, best_loss, elapsed, learning_rate_value))

        if bfgs == 1:

            start_time = time.time()

            tf_dict_all = {self.eta_tf: self.eta_tr, self.eta_t_tf: self.eta_t_tr, self.g_tf: self.g_tr,
                       self.ag_tf: self.ag_tr, self.lift_tf: self.lift, self.ag_c_tf: self.ag_c,
                       self.Phi_tf: self.Phi_t, self.learning_rate: learning_rate, self.best_loss: best_loss}

            self.optimizer.minimize(self.sess,
                                    feed_dict=tf_dict_all,
                                    fetches=[self.loss, self.best_loss],
                                    loss_callback=self.callback)
                                    # step_callback=self.step_callback)

            Loss_u.append(self.sess.run(self.loss_u, tf_dict_all))
            Loss_udot.append(self.sess.run(self.loss_udot, tf_dict_all))
            Loss_g.append(self.sess.run(self.loss_g, tf_dict))
            Loss_ut_c.append(self.sess.run(self.loss_ut_c, tf_dict_all))
            Loss_gt_c.append(self.sess.run(self.loss_gt_c, tf_dict_all))
            Loss_e.append(self.sess.run(self.loss_e, tf_dict_all))
            Loss.append(self.sess.run(self.loss, tf_dict_all))
            Loss_val.append(self.sess.run(self.loss, tf_dict_val))

        return Loss_u, Loss_udot, Loss_g, Loss_ut_c, Loss_gt_c, Loss_e, Loss, Loss_val, best_loss

    def callback(self, loss, best_loss):

        global Loss_BFGS
        global Loss_val_BFGS
        Loss_BFGS = np.append(Loss_BFGS, loss)

        loss_val = self.sess.run(self.loss, feed_dict={self.eta_tf: self.eta_val, self.eta_t_tf: self.eta_t_val, self.g_tf: self.g_val,
                               self.ag_tf: self.ag_val, self.lift_tf: self.lift, self.ag_c_tf: self.ag_c, self.Phi_tf: self.Phi_t})

        Loss_val_BFGS = np.append(Loss_val_BFGS, loss_val)

        print('Loss: %.3e, Loss_val: %.3e, Best_loss: %.3e'
              % (loss, loss_val, best_loss))

    def step_callback(self, loss):

        loss_val = self.sess.run(self.loss, feed_dict={self.eta_tf: self.eta_val, self.eta_t_tf: self.eta_t_val, self.g_tf: self.g_val,
                                 self.ag_tf: self.ag_val, self.lift_tf: self.lift, self.ag_c_tf: self.ag_c, self.Phi_tf: self.Phi_t})

        eta_star, eta_t_star, eta_tt_star, eta_dot_star, g_star = self.predict(self.ag_val, self.Phi_t[0:self.ag_val.shape[0]])
        loss_val = np.mean(np.square(eta_star, self.eta_val))

        print('Loss_val: %.3e', loss_val)

    def callback1(self, loss):
        print('Loss:', loss)

    def predict(self, ag_star, Phi_star):

        tf_dict = {self.ag_tf: ag_star, self.Phi_tf: Phi_star}

        eta_star = self.sess.run(self.eta_pred, tf_dict)
        eta_t_star = self.sess.run(self.eta_t_pred, tf_dict)
        eta_tt_star = self.sess.run(self.eta_tt_pred, tf_dict)
        eta_dot_star = self.sess.run(self.eta_dot_pred, tf_dict)
        g_star = self.sess.run(self.g_pred, tf_dict)

        return eta_star, eta_t_star, eta_tt_star, eta_dot_star, g_star

    def predict_c(self, ag_star, Phi_star):

        tf_dict = {self.ag_c_tf: ag_star, self.Phi_tf: Phi_star}
        lift_star = self.sess.run(self.lift_c_pred, tf_dict)

        return lift_star

    def predict_best_model(self, path, ag_star, Phi_star):
        self.saver.restore(sess=self.sess, save_path=path)

        tf_dict = {self.ag_tf: ag_star, self.Phi_tf: Phi_star}

        eta_star = self.sess.run(self.eta_pred, tf_dict)
        eta_t_star = self.sess.run(self.eta_t_pred, tf_dict)
        eta_tt_star = self.sess.run(self.eta_tt_pred, tf_dict)
        eta_dot_star = self.sess.run(self.eta_dot_pred, tf_dict)
        g_star = self.sess.run(self.g_pred, tf_dict)

        return eta_star, eta_t_star, eta_tt_star, eta_dot_star, g_star

if __name__ == "__main__": 

    dataDir = ".../"
    mat = scipy.io.loadmat(dataDir + 'data_boucwen_GM.mat')

    t = mat['time']
    dt = 0.02
    n1 = int(dt / 0.005)
    t = t[::n1]

    ag_data = mat['input_tf'][:, ::n1]  # ag, ad, av
    u_data = mat['target_X_tf'][:, ::n1]
    ut_data = mat['target_Xd_tf'][:, ::n1]
    utt_data = mat['target_Xdd_tf'][:, ::n1]
    ag_data = ag_data.reshape([ag_data.shape[0], ag_data.shape[1], 1])
    u_data = u_data.reshape([u_data.shape[0], u_data.shape[1], 1])
    ut_data = ut_data.reshape([ut_data.shape[0], ut_data.shape[1], 1])
    utt_data = utt_data.reshape([utt_data.shape[0], utt_data.shape[1], 1])

    ag_pred = mat['input_pred_tf'][:, ::n1]  # ag, ad, av
    u_pred = mat['target_pred_X_tf'][:, ::n1]
    ut_pred = mat['target_pred_Xd_tf'][:, ::n1]
    utt_pred = mat['target_pred_Xdd_tf'][:, ::n1]
    ag_pred = ag_pred.reshape([ag_pred.shape[0], ag_pred.shape[1], 1])
    u_pred = u_pred.reshape([u_pred.shape[0], u_pred.shape[1], 1])
    ut_pred = ut_pred.reshape([ut_pred.shape[0], ut_pred.shape[1], 1])
    utt_pred = utt_pred.reshape([utt_pred.shape[0], utt_pred.shape[1], 1])

    n = u_data.shape[1]
    phi1 = np.concatenate([np.array([-3 / 2, 2, -1 / 2]), np.zeros([n - 3, ])])
    temp1 = np.concatenate([-1 / 2 * np.identity(n - 2), np.zeros([n - 2, 2])], axis=1)
    temp2 = np.concatenate([np.zeros([n - 2, 2]), 1 / 2 * np.identity(n - 2)], axis=1)
    phi2 = temp1 + temp2
    phi3 = np.concatenate([np.zeros([n - 3, ]), np.array([1 / 2, -2, 3 / 2])])
    Phi_t0 = 1 / dt * np.concatenate(
            [np.reshape(phi1, [1, phi1.shape[0]]), phi2, np.reshape(phi3, [1, phi3.shape[0]])], axis=0)
    Phi_t0 = np.reshape(Phi_t0, [1, n, n])

    ag_star = ag_data
    eta_star = u_data
    eta_t_star = ut_data
    eta_tt_star = utt_data
    ag_c_star = np.concatenate([ag_data, ag_pred[0:53]])
    lift_star = -ag_c_star
    eta_c_star = np.concatenate([u_data, u_pred[0:53]])
    eta_t_c_star = np.concatenate([ut_data, ut_pred[0:53]])
    eta_tt_c_star = np.concatenate([utt_data, utt_pred[0:53]])

    eta = eta_star
    ag = ag_star
    lift = lift_star
    eta_t = eta_t_star
    eta_tt = eta_tt_star
    g = -eta_tt - ag
    ag_c = ag_c_star

    # Training Data
    eta_train = eta
    ag_train = ag
    lift_train = lift
    eta_t_train = eta_t
    eta_tt_train = eta_tt
    g_train = g
    ag_c_train = ag_c

    Loss_BFGS = np.empty([0])
    Loss_val_BFGS = np.empty([0])
    Phi_t = np.repeat(Phi_t0, ag_c_star.shape[0], axis=0)

with tf.device('/device:GPU:0'):
    # with tf.device('/cpu:0'):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.4
    session = tf.Session(config=config)
    # tf.Session(config=tf.ConfigProto(log_device_placement=True))

    # Training
    model = DeepPhyLSTM(eta_train, eta_t_train, g_train, ag_train, ag_c, lift, Phi_t, save_path=save_path)

    Loss_u1, Loss_udot1, Loss_g1, Loss_ut_c1, Loss_gt_c1, Loss_e1, Loss1, Loss_val1, best_loss1 = model.train(num_epochs=5000, learning_rate=1e-3, bfgs=0)

    train_loss = Loss1
    test_loss = Loss_val
    best_loss = best_loss

    plt.figure()
    plt.plot(np.log(train_loss), label='loss')
    plt.plot(np.log(test_loss), label='loss_val')
    plt.legend()

    # Results
    # Training performance
    X_train = ag_data
    y_train_ref = u_data
    yt_train_ref = ut_data
    ytt_train_ref = utt_data
    lift_train_ref = -X_train
    g_train_ref = -ytt_train_ref + lift_train_ref

    eta, eta_t, eta_tt, eta_dot, g = model.predict(X_train, np.repeat(Phi_t0, len(X_train), axis=0))
    lift = model.predict_c(X_train, np.repeat(Phi_t0, len(X_train), axis=0))
    y_train_pred = eta
    yt_train_pred = eta_t
    ytt_train_pred = eta_tt
    g_train_pred = -eta_tt + lift

    dof = 0
    for ii in range(len(y_train_ref)):
        plt.figure()
        plt.plot(y_train_ref[ii, :, dof], '-', label='True')
        plt.plot(y_train_pred[ii, :, dof], '--', label='Predict')
        plt.title('Training_u')
        plt.legend()

    for ii in range(len(y_train_ref)):
        plt.figure()
        plt.plot(yt_train_ref[ii, :, dof], label='True')
        plt.plot(yt_train_pred[ii, :, dof], label='Predict')
        plt.title('Training_u_t')
        plt.legend()

    for ii in range(len(y_train_ref)):
        plt.figure()
        plt.plot(ytt_train_ref[ii, :, dof], label='True')
        plt.plot(ytt_train_pred[ii, :, dof], label='Predict')
        plt.title('Training_u_tt')
        plt.legend()

    for ii in range(len(y_train_ref)):
        plt.figure()
        plt.plot(g_train_ref[ii, :, dof], label='True')
        plt.plot(g_train_pred[ii, :, dof], label='Predict')
        plt.title('Training_g')
        plt.legend()

    for ii in range(len(y_train_ref)):
        plt.figure()
        plt.plot(y_train_ref[ii, :, dof], g_train_ref[ii, :, dof], label='True')
        plt.plot(y_train_pred[ii, :, dof], g_train_pred[ii, :, dof], label='Predict')
        plt.title('Training_Hysteresis')
        plt.legend()

    # Prediction performance
    X_pred = ag_pred
    y_pred_ref = u_pred
    yt_pred_ref = ut_pred
    ytt_pred_ref = utt_pred
    lift_pred_ref = -X_pred
    g_pred_ref = -ytt_pred_ref + lift_pred_ref

    eta, eta_t, eta_tt, eta_dot, g = model.predict(X_pred, np.repeat(Phi_t0, len(X_pred), axis=0))
    lift = model.predict_c(X_pred, np.repeat(Phi_t0, len(X_pred), axis=0))
    y_pred = eta
    yt_pred = eta_t
    ytt_pred = eta_tt
    g_pred = -eta_tt + lift

    dof = 0
    for ii in range(len(y_pred_ref)):
        plt.figure()
        plt.plot(y_pred_ref[ii, :, dof], label='True')
        plt.plot(y_pred[ii, :, dof], label='Predict')
        plt.title('Prediction_u')
        plt.legend()

    for ii in range(len(Index_pred)):
        plt.figure()
        plt.plot(yt_pred_ref[ii, :, dof], label='True')
        plt.plot(yt_pred[ii, :, dof], label='Predict')
        plt.title('Prediction_u_t')
        plt.legend()

    for ii in range(len(Index_pred)):
        plt.figure()
        plt.plot(ytt_pred_ref[ii, :, dof], label='True')
        plt.plot(ytt_pred[ii, :, dof], label='Predict')
        plt.title('Prediction_u_tt')
        plt.legend()

    for ii in range(len(Index_pred)):
        plt.figure()
        plt.plot(g_pred_ref[ii, :, dof], label='True')
        plt.plot(g_pred[ii, :, dof], label='Predict')
        plt.title('Prediction_g')
        plt.legend()

    for ii in range(len(Index_pred)):
        plt.figure()
        plt.plot(y_pred_ref[ii, :, dof], g_pred_ref[ii, :, dof], label='True')
        plt.plot(y_pred[ii, :, dof], g_pred[ii, :, dof], label='Predict')
        plt.title('Prediction_Hysteresis')
        plt.legend()

scipy.io.savemat(dataDir + 'results/results_PhyLSTM3.mat',
                 {'y_train_ref': y_train_ref, 'yt_train_ref': yt_train_ref, 'ytt_train_ref': ytt_train_ref, 'g_train_ref': g_train_ref,
                  'y_train_pred': y_train_pred, 'yt_train_pred': yt_train_pred, 'ytt_train_pred': ytt_train_pred, 'g_train_pred': g_train_pred,
                  'y_pred_ref': y_pred_ref, 'yt_pred_ref': yt_pred_ref, 'ytt_pred_ref': ytt_pred_ref, 'g_pred_ref': g_pred_ref,
                  'y_pred': y_pred, 'yt_pred': yt_pred, 'ytt_pred': ytt_pred, 'g_pred': g_pred,
                  'X_train': X_train, 'X_pred': X_pred, 'time': t, 'dt': dt,
                  'train_loss': train_loss, 'test_loss': test_loss, 'best_loss': best_loss})
