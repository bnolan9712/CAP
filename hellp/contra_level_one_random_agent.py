import retro
import gym
import numpy as np
from collections import deque

from skimage import transform # Help us to preprocess the frames
from skimage.color import rgb2gray # Help us to gray our frames

import tensorflow as tf

import random
import warnings
import math

# if __name__ == "__main__":
env = retro.make("Contra-Nes", state="Level1", scenario="scenario")
env.reset()
print("The size of our frame is: ", env.observation_space)
print("The actoin size is : ", env.action_space.n)
possible_actions = np.identity(9,dtype=int).tolist()
#frame size: Box(224, 240, 3)
#action size: 9

stack_size = 4
stacked_frames  =  deque([np.zeros((100,120), dtype=np.int) for i in range(stack_size)], maxlen=4)

def stack_frames(stacked_frames, state, is_new_episode):
    frame = state


    gray = rgb2gray(frame)

    frame = transform.resize(gray, [100, 120])

    
    if is_new_episode:
        stacked_frames = deque([np.zeros((100,120), dtype=np.int) for i in range(stack_size)], maxlen=4)

        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)

        stacked_state = np.stack(stacked_frames, axis=2)
    else:
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis=2)
    return stacked_state, stacked_frames

state_size = [100, 120, 4]
action_size = env.action_space.n
print(action_size)
learning_rate =  0.00025

total_episodes = 5000
max_steps = 5000
batch_size = 64

max_tau = 10000

explore_start = 1.0
explore_stop = 0.01
decay_rate = 0.00005

gamma = 0.95
pretrain_length = 10000
memory_size = 100000

training = True
pretrain = False
episode_render= False

class DDDQNNet:
    def __init__(self, state_size, action_size, learning_rate, name):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.name = name

        with tf.variable_scope(self.name):
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name = "inputs_")
            self.ISWeights_ = tf.placeholder(tf.float32, [None,1], name='IS_weights')
            self.actions_ = tf.placeholder(tf.float32, [None, action_size], name="actions_")
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            self.conv1 = tf.layers.conv2d(inputs = self.inputs_,
                                         filters = 32,
                                         kernel_size = [8,8],
                                         strides = [4,4],
                                         padding = "VALID",
                                         kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv1")
            self.conv1_out = tf.nn.elu(self.conv1, name="conv1_out")

            self.conv2 = tf.layers.conv2d(inputs = self.conv1_out,
                                 filters = 64,
                                 kernel_size = [4,4],
                                 strides = [2,2],
                                 padding = "VALID",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv2")
            self.conv2_out = tf.nn.elu(self.conv2, name="conv2_out")

            self.conv3 = tf.layers.conv2d(inputs = self.conv2_out,
                                 filters = 128,
                                 kernel_size = [4,4],
                                 strides = [2,2],
                                 padding = "VALID",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv3")

            self.conv3_out = tf.nn.elu(self.conv3, name="conv3_out")

            self.flatten = tf.layers.flatten(self.conv3_out)

            self.value_fc = tf.layers.dense(inputs = self.flatten,
                                  units = 512,
                                  activation = tf.nn.elu,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="value_fc")
            
            self.value = tf.layers.dense(inputs = self.value_fc,
                                        units = 1,
                                        activation = None,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="value")

            self.advantage_fc = tf.layers.dense(inputs = self.flatten,
                                  units = 512,
                                  activation = tf.nn.elu,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="advantage_fc")
            
            self.advantage = tf.layers.dense(inputs = self.advantage_fc,
                                        units = self.action_size,
                                        activation = None,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="advantages")
            self.output = self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage, axis=1,keepdims=True))

            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)

            self.absolute_errors = tf.abs(self.target_Q - self.Q)
            
            self.loss = tf.reduce_mean(self.ISWeights_ * tf.squared_difference(self.target_Q, self.Q))
            
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

tf.reset_default_graph()

DQNetwork = DDDQNNet(state_size, action_size, learning_rate, name="DQNetwork")

TargetNetwork = DDDQNNet(state_size, action_size, learning_rate, name="TargetNetwork")

class SumTree(object):
    
    data_pointer = 0
    
    def __init__(self, capacity):
        self.capacity = capacity 
        dtype = [('states', np.dtype((np.float64, (100, 120, 4)))), ('actions', np.dtype((np.int, (9)))),
                 ('rewards', 'i4'), ('next_states', np.dtype((np.float64, (100, 120, 4)))), ('done', 'i4')]

        self.tree = np.memmap("F:\data_files\\tree.npy", mode='r+', shape=(2*capacity - 1,))

        """ tree:
            0
           / \
          0   0
         / \ / \
        0  0 0  0  [Size: capacity] it's at this line that there is the priorities score (aka pi)
        """

        # Contains the experiences (so the size of data is capacity)
        # self.data = np.zeros(capacity, dtype=object)
        self.data = np.memmap("F:\data_files\data.npy", dtype=dtype, mode='r+', shape=(capacity))
    
    
    def add(self, priority, data):

        tree_index = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
    
        self.update (tree_index, priority)
        
        self.data_pointer += 1
        
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
            
    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        
        while tree_index != 0:
            
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change
    def get_leaf(self, v):
        parent_index = 0
        
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            
            else:
                
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                    
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index
            
        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]
    
    @property
    def total_priority(self):
        return self.tree[0]

class Memory(object):
    PER_e = 0.01
    PER_a = 0.6
    PER_b = 0.4
    
    PER_b_increment_per_sampling = 0.001
    
    absolute_error_upper = 1.

    def __init__(self, capacity):        
        self.tree = SumTree(capacity)
        

    def store(self, experience):
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        
        if max_priority == 0 or math.isnan(max_priority):
            max_priority = self.absolute_error_upper
        
        self.tree.add(max_priority, experience)

    def sample(self, n):
        memory_b = []
        
        b_idx, b_ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1), dtype=np.float32)
        
        priority_segment = self.tree.total_priority / n
    
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])
        
        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        max_weight = (p_min * n) ** (-self.PER_b)
        
        for i in range(n):
        
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)
            
            index, priority, data = self.tree.get_leaf(value)
            
            
            sampling_probabilities = priority / self.tree.total_priority
            
            
            b_ISWeights[i, 0] = np.power(n * sampling_probabilities, -self.PER_b)/ max_weight
                                   
            b_idx[i]= index
            
            experience = [data]
            
            memory_b.append(experience)
        
        return b_idx, memory_b, b_ISWeights
    
    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

memory = Memory(memory_size)

state = env.reset()

if pretrain:
    for i in range(pretrain_length):
        if i == 0:
            # state = env.get_screen()
            state, stacked_frames = stack_frames(stacked_frames, state, True)
        
        action = random.choice(possible_actions)
        
        next_state, reward, done, info = env.step(action)
        

        if done:
            # We finished the episode
            next_state = np.zeros(state.shape)
            
            # Add experience to memory
            #experience = np.hstack((state, [action, reward], next_state, done))
            
            experience = state, action, reward, next_state, done
            memory.store(experience)
            
            # Start a new episode
            state= env.reset()
            
            # First we need a state
            
            # Stack the frames
            state, stacked_frames = stack_frames(stacked_frames, state, True)
            
        else:
            # Get the next state
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            
            # Add experience to memory
            experience = state, action, reward, next_state, done
            memory.store(experience)
            
            # Our state is now the next_state
            state = next_state
# Setup TensorBoard Writer
writer = tf.summary.FileWriter("/tensorboard/dddqn/1")

## Losses
tf.summary.scalar("Loss", DQNetwork.loss)

write_op = tf.summary.merge_all()

def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions):
    ## EPSILON GREEDY STRATEGY
    # Choose action a from state s using epsilon greedy.
    ## First we randomize a number
    exp_exp_tradeoff = np.random.rand()

    # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
    
    if (explore_probability > exp_exp_tradeoff):
        # Make a random action (exploration)
        action = random.choice(possible_actions)
        
    else:
        # Get action from Q-network (exploitation)
        # Estimate the Qs values state
        Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state.reshape((1, *state.shape))})
        
        # Take the biggest Q value (= the best action)
        choice = np.argmax(Qs)
        action = possible_actions[int(choice)]
                
    return action, explore_probability

def update_target_graph():
    
    # Get the parameters of our DQNNetwork
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "DQNetwork")
    
    # Get the parameters of our Target_network
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "TargetNetwork")

    op_holder = []
    
    # Update our target_network parameters with DQNNetwork parameters
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

# while True:
#     action = env.action_space.sample()
#     ob, rew, done, info = env.step(action)
#     env.render()
#     print("Action", action, "Reward", rew)

saver = tf.train.Saver()

if training == True:
    with tf.Session() as sess:
        # Initialize the variables
        sess.run(tf.global_variables_initializer())
        
        # Initialize the decay rate (that will use to reduce epsilon) 
        decay_step = 0
        
        # Set tau = 0
        tau = 0

        # Init the env
        # env.init()
        # print("got here")        
        # Update the parameters of our TargetNetwork with DQN_weights
        update_target = update_target_graph()
        sess.run(update_target)
        
        for episode in range(total_episodes):
            # Set step to 0
            step = 0
            
            # Initialize the rewards of the episode
            episode_rewards = []
            
            # Make a new episode and observe the first state
            state = env.reset()
        
            
            # Remember that stack frame function also call our preprocess function.
            state, stacked_frames = stack_frames(stacked_frames, state, True)
        
            while step < max_steps:
                step += 1
                
                # Increase the C step
                tau += 1
                
                # Increase decay_step
                decay_step +=1
                
                # With ϵ select a random action atat, otherwise select a = argmaxQ(st,a)
                action, explore_probability = predict_action(explore_start, explore_stop, decay_rate, decay_step, state, possible_actions)
                obs, reward, done, info = env.step(action)
                #env.render()

                # Look if the episode is finished
                
                # Add the reward to total reward
                episode_rewards.append(reward)

                # If the env is finished
                if done or step > 10000:
                    # the episode ends so no next state
                    next_state = np.zeros((120,140), dtype=np.int)
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                    # Set step = max_steps to end the episode
                    step = max_steps

                    # Get the total reward of the episode
                    total_reward = np.sum(episode_rewards)

                    print('Episode: {}'.format(episode),
                              'Total reward: {}'.format(total_reward),
                              'Training loss: {:.4f}'.format(loss),
                              'Explore P: {:.4f}'.format(explore_probability))

                    # Add experience to memory
                    experience = state, action, reward, next_state, done
                    memory.store(experience)

                else:
                    # Get the next state
                    next_state = env.get_screen()
                    
                    # Stack the frame of the next_state
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                    

                    # Add experience to memory
                    experience = state, action, reward, next_state, done
                    memory.store(experience)
                    
                    # st+1 is now our current state
                    state = next_state


                ### LEARNING PART            
                # Obtain random mini-batch from memory
                tree_idx, batch, ISWeights_mb = memory.sample(batch_size)
                
                states_mb = np.array([each[0][0] for each in batch], ndmin=3)
                actions_mb = np.array([each[0][1] for each in batch])
                rewards_mb = np.array([each[0][2] for each in batch]) 
                next_states_mb = np.array([each[0][3] for each in batch], ndmin=3)
                dones_mb = np.array([each[0][4] for each in batch])

                target_Qs_batch = []

                q_next_state = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: next_states_mb})
                
                # Calculate Qtarget for all actions that state
                q_target_next_state = sess.run(TargetNetwork.output, feed_dict = {TargetNetwork.inputs_: next_states_mb})
                for i in range(0, len(batch)):
                    terminal = dones_mb[i]
                    
                    # We got a'
                    action = np.argmax(q_next_state[i])

                    # If we are in a terminal state, only equals reward
                    if terminal:
                        target_Qs_batch.append(rewards_mb[i])
                        
                    else:
                        # Take the Qtarget for action a'
                        target = rewards_mb[i] + gamma * q_target_next_state[i][action]
                        target_Qs_batch.append(target)
                        

                targets_mb = np.array([each for each in target_Qs_batch])

                
                _, loss, absolute_errors = sess.run([DQNetwork.optimizer, DQNetwork.loss, DQNetwork.absolute_errors],
                                    feed_dict={DQNetwork.inputs_: states_mb,
                                               DQNetwork.target_Q: targets_mb,
                                               DQNetwork.actions_: actions_mb,
                                              DQNetwork.ISWeights_: ISWeights_mb})
                memory.batch_update(tree_idx, absolute_errors)
                
                
                # Write TF Summaries
                summary = sess.run(write_op, feed_dict={DQNetwork.inputs_: states_mb,
                                                   DQNetwork.target_Q: targets_mb,
                                                   DQNetwork.actions_: actions_mb,
                                              DQNetwork.ISWeights_: ISWeights_mb})
                writer.add_summary(summary, episode)
                writer.flush()
                
                if tau > max_tau:
                    # Update the parameters of our TargetNetwork with DQN_weights
                    update_target = update_target_graph()
                    sess.run(update_target)
                    tau = 0
                    print("Model updated")

            # Save model every 5 episodes
            if episode % 5 == 0:
                save_path = saver.save(sess, "./models/model.ckpt")
                print("Model Saved")

with tf.Session() as sess:
    
    env = retro.make("Contra-Nes", state="Level1", scenario="scenario")
    
    env.init()    
    
    # Load the model
    saver.restore(sess, "./models/model.ckpt")
    env.init()
    
    for i in range(10):
        
        state, reward, done, info = env.reset()
        # state = env.get_screen()
        state, stacked_frames = stack_frames(stacked_frames, state, True)
    
        while not done:
            ## EPSILON GREEDY STRATEGY
            # Choose action a from state s using epsilon greedy.
            ## First we randomize a number
            exp_exp_tradeoff = np.random.rand()
            

            explore_probability = 0.01
    
            if (explore_probability > exp_exp_tradeoff):
                # Make a random action (exploration)
                action = random.choice(possible_actions)
        
            else:
                # Get action from Q-network (exploitation)
                # Estimate the Qs values state
                Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state.reshape((1, *state.shape))})
        
                # Take the biggest Q value (= the best action)
                choice = np.argmax(Qs)
                action = possible_actions[int(choice)]
            
            obs, reward, done, info = env.step(action)
            
        
            if done:
                break  
                
            else:
                next_state = env.get_screen()
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                state = next_state
        
        score = env.get_total_reward()
        print("Score: ", score)
    
    env.close()

""" class Discretizer(gym.ActionWrapper):
    def __init__(self, env):
        super(Discretizer, self).__init__(env)
        buttons = ["B", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A"]
        actions = [['B'], ['B', 'RIGHT'], ['B', 'LEFT'], ['RIGHT'], ['LEFT'], ['LEFT','L'], ['RIGHT','R'], ['B', 'L'], ['B', 'R'], ['B', 'A']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):
        return self._actions[a].copy() """