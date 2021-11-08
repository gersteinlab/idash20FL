# idash20FL

## Benchmarking:
This directory consists of all the code used to analyze data produced by the model. 
iterations.py creates a graph of the number of iterations used by epsilon value, 
loss_accuracy.py creates graphs of loss and accuracy by epsilon value, 
precision_recall.py creates tables of precision-recall pairs by epsilon value, 
and vanilla_nn.ipynb creates these images for a non-federated neural network used for benchmarking.

## Model:
data_generator.ipynb consists of a framework of code which we used to generate training/validation/testing datasets. 
By simply modifying the numbers in this code we can generate IID & Equal, Non-IID & Equal, IID & Unequal, and Non-IID & Unequal data splits 
as described in our paper. Client.py consists of all the code required to train and collect metrics from clients within the federated network. 
Server.py consists of all the code required to coordinate and average metrics produced by clients in the federated network.
The code in this directory should be used as a framework for implementation. 
When implementing this code, ensure to use relevant parts. Put the server.py and client.py in directories on two different servers and 
run server.py with the appropriate configuration to communicate with client.py.

### Steps Using server.py and client.py:
In order to use server.py and client.py, you should put them each in their own directory on different machines. 
Specifically, you should put server.py on one server in its own directory, and client.py on two or more other servers w
ith data files included in their directories. Then, within the server.py file, change the server and client IP addresses, 
which are currently specific to our testing configuration. Additionally, change the filepaths within server.py and client.py 
to be specific to your directory architecture. Finally, you can change the hyperparameters we use in the server.py file. 
To run server.py, you must run it via the shell and additionally insert system arguments of the data files' name in clients 1 and 2, 
the target epsilon value, and the username-password combinations for each of the server and client side.

After setting up the paths in the server.py and Client.py scripts, use

```python3 server.py <data1> <data2> <epsilon> <userNameServer> <passWordServer> <userNameClient1> <passWordClient1> <userNameClient2> <passWordClient2>```

data1: name of the training data file on Client 1
data2: name of the training data file on Client 2
epsilon: desired privacy parameter in differential privacy (smaller values mean more privacy enforced)
userNameServer: user name on the server for ssh access
passWordServer: password on the server for ssh access
userNameClient1/2: user name on the client for ssh access
passWordClient1/2: password on the client for ssh access


