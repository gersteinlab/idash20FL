import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    import concurrent.futures
    import contextlib
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import pandas as pd
    import pickle as pkl
    import seaborn as sns
    import shutil
    import spur
    import sys
    import tensorflow as tf
    from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
    from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPGradientDescentGaussianOptimizer
    import time
    sns.set()

    # SERVER_0 is the master server, which drives the program and holds the federated model
    # SERVER_1 and SERVER_2 have access to data and train models on parts of the data
    SERVER_0 = '172.24.145.62'
    SERVER_1 = '172.24.145.64'
    SERVER_2 = '172.24.145.63'

    def tolist(f):
        xn = pd.read_csv(f, sep='\t', index_col=0)
        xn = [x[1:] for x in xn.itertuples()]
        return xn

    def find_noise_multiplier(epsilon, features, batch_size, epochs, delta):
        decay = 0.001
        nm = 0.5

        with contextlib.redirect_stdout(None):
            budget, r = compute_dp_sgd_privacy.compute_dp_sgd_privacy(
                n=features, batch_size=batch_size, epochs=epochs, noise_multiplier=nm, delta=delta)
        while budget < epsilon or budget > (epsilon + 0.04):
            with contextlib.redirect_stdout(None):
                budget, r = compute_dp_sgd_privacy.compute_dp_sgd_privacy(
                    n=features, batch_size=batch_size, epochs=epochs, noise_multiplier=nm, delta=delta)
                nm *= (1 - decay) if epsilon > budget else (1 + decay)

        return nm

    def run_clients(epsilon, client, usr_client, pwd_client, data, usr_server, pwd_server, batch_size, num_microbatches, epochs, learning_rate, l2_norm_clip, noise_multiplier):
        shell = spur.LocalShell()
        command = ['sshpass', '-p', pwd_client, 'scp', '/home/shared/idash20/Test_iDASH_' + str(epsilon) + '/current_model',
                   usr_client + '@' + client + ':/home/' + usr_client + '/Epsilon_' + str(epsilon)]
        shell.run(command)

        shell = spur.SshShell(
            hostname=client, username=usr_client, password=pwd_client)
        command = ['/home/ak2020/anaconda3/envs/TensorFlow/bin/python3', '/home/ak2020/Epsilon_' + str(epsilon) + '/Client.py', str(epsilon), str(data), str(usr_server), str(pwd_server), str(
            SERVER_0), str(batch_size), str(num_microbatches), str(epochs), str(learning_rate), str(l2_norm_clip), str(noise_multiplier)]
        server_response = shell.run(command).output.decode('utf-8').split(' ')

        return server_response

    def clear_clients(epsilon, client, data, usr_client, pwd_client):
        shell = spur.SshShell(
            hostname=client, username=usr_client, password=pwd_client)
        command = ['rm', 'Epsilon_' + str(epsilon) + '/current_model', 'Epsilon_' + str(
            epsilon) + '/' + data, 'Epsilon_' + str(epsilon) + '/validate.pkl']
        shell.run(command)

    def main(iter, data1, data2, epsilon, usr0, pwd0, usr1, pwd1, usr2, pwd2, patience, batch_size, num_microbatches, epochs, learning_rate, learning_rate_decay, features, delta, l2_norm_clip, noise_multiplier):
        # Delete past client data
        with concurrent.futures.ThreadPoolExecutor() as executor:
            command_1 = executor.submit(
                clear_clients, epsilon, SERVER_1, data1, usr1, pwd1)
            command_2 = executor.submit(
                clear_clients, epsilon, SERVER_2, data2, usr2, pwd2)

        # Model architecture
        federated_model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=features),
            tf.keras.layers.Dropout(0.2, input_shape=(features,)),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        federated_model.save('current_model')
        federated_model.save('best_model')

        iterations = 0
        patience_counter = 0
        lowest_validation = float('inf')
        early_stopping = False

        while(early_stopping == False):
            # Reset temporary directory
            dir = os.path.abspath(os.path.dirname(__file__)) + '/temp_server'
            if os.path.exists(dir):
                shutil.rmtree(dir)
            os.mkdir(dir)

            # Set LR
            learning_rate *= learning_rate_decay

            with concurrent.futures.ThreadPoolExecutor() as executor:
                command_1 = executor.submit(run_clients, epsilon, SERVER_1, usr1, pwd1, data1, usr0, pwd0,
                                            batch_size, num_microbatches, epochs, learning_rate, l2_norm_clip, noise_multiplier)
                command_2 = executor.submit(run_clients, epsilon, SERVER_2, usr2, pwd2, data2, usr0, pwd0,
                                            batch_size, num_microbatches, epochs, learning_rate, l2_norm_clip, noise_multiplier)

                server_1_response = [float(j) for j in command_1.result()]
                server_2_response = [float(j) for j in command_2.result()]

            # Wait until the two servers return their weights files
            while (os.path.isfile(dir + '/' + data1) == False or os.path.isfile(dir + '/' + data2 == False)):
                time.sleep(5)

            # Check if the current validation is better than the previous best one
            relative_weights = [server_1_response[0] / (server_1_response[0] + server_2_response[0]),
                                server_2_response[0] / (server_1_response[0] + server_2_response[0])]
            current_validation = server_1_response[3] * \
                relative_weights[0] + \
                server_2_response[3] * relative_weights[1]

            if current_validation < lowest_validation:
                patience_counter = 0
                federated_model.save('best_model')
                lowest_validation = current_validation
                print(f'Validation Loss: {current_validation}')
            else:
                patience_counter += 1

            # Conduct federated averaging to update the federated_model if we have not exceeded patience
            if patience_counter > patience:
                early_stopping = True
            else:
                server_1_model = tf.keras.models.load_model(dir + '/' + data1)
                server_2_model = tf.keras.models.load_model(dir + '/' + data2)

                weights = [server_1_model.get_weights(
                ), server_2_model.get_weights()]

                new_weights = []
                for weights_list_tuple in zip(*weights):
                    new_weights.append(np.array([np.average(np.array(
                        weights_), axis=0, weights=relative_weights) for weights_ in zip(*weights_list_tuple)]))

                federated_model.set_weights(new_weights)
                federated_model.save('current_model')

                iterations += 1
                print(iter, iterations)

        thresholds = list(np.linspace(0, 0.999, 1000))
        model = tf.keras.models.load_model('best_model')
        optimizer = DPGradientDescentGaussianOptimizer(
            l2_norm_clip=l2_norm_clip,
            noise_multiplier=noise_multiplier,
            num_microbatches=num_microbatches,
            learning_rate=learning_rate
        )
        x_normal = tolist('test-Normal.txt')
        x_tumor = tolist('test-Tumor.txt')
        y = np.append(np.zeros(len(x_normal)), np.ones(len(x_tumor)))
        x = np.float32(x_normal + x_tumor)

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[
                      tf.keras.metrics.Precision(thresholds=thresholds), tf.keras.metrics.Recall(thresholds=thresholds)])
        loss, precision, recall = model.evaluate(x, y, verbose=0)

        precision = np.append(precision, 1)
        recall = np.append(recall, 0)

        with open('precision_' + str(iter) + '.pkl', 'wb') as f:
            pkl.dump(precision, f)
        with open('recall_' + str(iter) + '.pkl', 'wb') as f:
            pkl.dump(recall, f)

    if __name__ == '__main__':
        # Inputs
        data1, data2, epsilon, usr0, pwd0, usr1, pwd1, usr2, pwd2 = [
            sys.argv[i] for i in range(1, 10)]

        # Hyperparameters
        patience = 10

        batch_size = 32
        num_microbatches = 1
        epochs = 20
        learning_rate = 5.26E-4
        learning_rate_decay = 0.95

        features = 17814
        delta = 1/features

        l2_norm_clip = 10
        epsilon = int(epsilon)
        noise_multiplier = find_noise_multiplier(
            epsilon, features, batch_size, epochs, delta)

        for w in range(0, 10):
            main(w, data1, data2, epsilon, usr0, pwd0, usr1, pwd1, usr2, pwd2, patience, batch_size,
                 num_microbatches, epochs, learning_rate, learning_rate_decay, features, delta, l2_norm_clip, noise_multiplier)
