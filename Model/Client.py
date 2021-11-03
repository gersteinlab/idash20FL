import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    import contextlib
    import math
    import numpy as np
    import os
    import pandas as pd
    import pickle as pkl
    import random
    import spur
    import sys
    from tensorflow import keras as keras
    from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
    from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPGradientDescentGaussianOptimizer


    def tolist(f):
        xn = pd.read_csv(f, sep='\t', index_col=0)
        xn = [x[1:] for x in xn.itertuples()]
        return xn


    def main():
        # Read in the arguments provided by the master server
        epsilon, data, usr0, pwd0, SERVER_0, batch_size, num_microbatches, epochs, learning_rate, l2_norm_clip, noise_multiplier = [sys.argv[i] for i in range(1, 12)]
        dir = os.path.abspath(os.path.dirname(__file__)) 
        try:
            epsilon = int(epsilon)
        except:
            epsilon = float(epsilon)
        batch_size = int(batch_size)
        num_microbatches = int(num_microbatches)
        epochs = int(epochs)
        learning_rate = float(learning_rate)
        l2_norm_clip = float(l2_norm_clip)
        noise_multiplier = float(noise_multiplier)

        # Import the data
        x_normal = tolist(dir + '/' + data + '-Normal.txt')
        x_tumor = tolist(dir + '/' + data + '-Tumor.txt')

        # Check if the test and validate indicies are already created
        if os.path.isfile(dir + '/' + 'validate.pkl') == False:
            total_len = len(x_normal) + len(x_tumor)
            validate_index = random.sample(range(0, total_len), (int)(0.11 * (total_len)))
            
            with open(dir + '/' + 'validate.pkl', 'wb') as validate_file:
                pkl.dump(validate_index, validate_file)

        # Separate out the test and validate data
        x_validate = []
        y_validate = []

        with open(dir + '/' + 'validate.pkl', 'rb') as validate_file:
            validate_index = pkl.load(validate_file)
        delete_index = validate_index

        for i in validate_index:
            if i < len(x_normal):
                x_validate.append(x_normal[i])
                y_validate.append(0)
            else:
                x_validate.append(x_tumor[i-len(x_normal)])
                y_validate.append(1)
        # Delete the test and validate data from training data
        for i in sorted(delete_index, reverse=True):
            if i < len(x_normal):
                del x_normal[i]
            else:
                del x_tumor[i-len(x_normal)]

        # Oversample training data
        if len(x_normal) > len(x_tumor):
            x_tumor = x_tumor * math.ceil(len(x_normal) / len(x_tumor))
            x_tumor = x_tumor[:len(x_normal)]

        if len(x_normal) < len(x_tumor):
            x_normal = x_normal * math.ceil(len(x_tumor) / len(x_normal))
            x_normal = x_normal[:len(x_tumor)]

        y_validate = np.float32(y_validate)
        x_validate = np.float32(x_validate)
        y_train = np.append(np.zeros(len(x_normal)), np.ones(len(x_tumor)))
        x_train = np.float32(x_normal + x_tumor)
        
        # Set up and run the neural network
        client_model = keras.models.load_model(dir + '/' + 'current_model')
        optimizer = DPGradientDescentGaussianOptimizer(
            l2_norm_clip=l2_norm_clip,
            noise_multiplier = noise_multiplier,
            num_microbatches = num_microbatches,
            learning_rate = learning_rate
        )
        client_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        validate_loss, validate_acc = client_model.evaluate(x_validate, y_validate, verbose=0)

        history = keras.callbacks.History()
        with contextlib.redirect_stdout(None):
            client_model.fit(x_train, y_train, verbose=0, epochs=epochs, batch_size=batch_size, callbacks=[history])

        client_model.save(dir + '/' + data)

        train_loss = history.history['loss'][-1]
        train_acc = history.history['acc'][-1]

        # Send the weights back to master server
        shell = spur.LocalShell()
        command_1 = ['sshpass', '-p', pwd0, 'scp', dir + '/' + data, usr0 + '@' + SERVER_0 + ':/home/shared/idash20/Test_iDASH_' + str(epsilon) + '/temp_server']
        shell.run(command_1)

        # Reset stdout and print
        print(len(x_train), train_loss, train_acc, validate_loss, validate_acc)




    if __name__ == '__main__':
        main()