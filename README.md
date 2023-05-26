# ODM
#### Tensoarboard 
- to launch tensorboard run the following command in the root of the project 3: `python -m tensorboard.main --logdir=[PATH_TO_LOGDIR]`

## Project 3
### Implementation
#### FQI
The adaptation of the FQI method using xrt is implemented in the file `FQI.py` and use the `replaybuffer` implement in `utils.py`
#### DDPG
The DDPG method is implemented in the file `DDPG.py` and depends of `Network.py` and `utils.py`
#### DQN
The DQN method is implemented in the file `DQN.py` and depends of `utils.py`
#### Gaussian
The Softmax method is implemented in the file `REINFORCE_normal.py`
#### Softmax 
The Softmax method is implemented in the file `REINFORCE_softmax.py`
#### interface 
The interface is implemented in the file `interface.py`and depends of `DDPG.py`
### Interface
#### FQI
To launch the FQI algorithm with `interface.py`, just type `interface.py <env_name> FQI <Nbr_action> <model_path> <execpted_reward>` for example `interface.py InvertedPendulum fqi 3 ./fqi/model/InvertedPendulum-v4_9_model_tuned.pkl False` for respectively the simple pendulum, 3 discrete action, and without computing the excepeted reward
#### Softmax 
To launch the softmax algorithm with `interface.py`, just type `interface.py 0 softmax_simple 0 0 False` or `interface.py 0 softmax_double 0 0 False` for respectively the simple and double pendulum
#### DDPG
To launch the FQI algorithm with `interface.py`, just type `interface.py <env_name> DDPG <Actor_path> <Critic_path> <execpted_reward>`.  for example, `InvertedDoublePendulum DDPG ./DDPG/InvertedDoublePendulum-v4modelActor_net_ddpg ./DDPG/InvertedDoublePendulum-v4modelCritic_net_ddpg False`

