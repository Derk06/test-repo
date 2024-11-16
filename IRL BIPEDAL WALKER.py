import serial
import time
import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
import os
from time import sleep
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
from mediapipe.python.solutions.face_mesh import FaceMesh
import gc

cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)
class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        next_state = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, next_state, dones

class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims, fc2_dims, name, chkpt_dir='tmp/td3'):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3.weights.h5')

        self.fc1 = Dense(self.fc1_dims, activation="relu")
        self.fc2 = Dense(self.fc2_dims, activation="relu")
        self.q = Dense(1, activation=None)

    def call(self, state, action):
        q1_action_value = self.fc1(tf.concat([state, action], axis=1))
        q1_action_value = self.fc2(q1_action_value)

        q = self.q(q1_action_value)

        return q

class ActorNetwork(keras.Model):
    def __init__(self, fc1_dims, fc2_dims, n_actions, name, chkpt_dir="tmp/td3"):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3.weights.h5')

        self.fc1 = Dense(self.fc1_dims, activation="relu")
        self.fc2 = Dense(self.fc2_dims, activation="relu")
        self.mu = Dense(self.n_actions, activation="tanh")

    def call(self, state):
        prob = self.fc1(state)
        prob = self.fc2(prob)

        mu = self.mu(prob)

        return mu

class Agent:
    def __init__(self, alpha, beta, input_dims, tau, gamma=0.99, update_actor_interval=2, warmup=1000, n_actions=2,
                 max_size=1000000, layer1_size=400, layer2_size=300, batch_size=300, noise=0.1):
        self.gamma = gamma
        self.tau = tau
        self.max_action = 180
        self.min_action = 0
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval

        self.actor = ActorNetwork(layer1_size, layer2_size, n_actions=n_actions, name='actor')

        self.critic_1 = CriticNetwork(layer1_size, layer2_size, name='critic_1')
        self.critic_2 = CriticNetwork(layer1_size, layer2_size, name='critic_2')

        self.target_actor = ActorNetwork(layer1_size, layer2_size, n_actions=n_actions, name='target_actor')

        self.target_critic_1 = CriticNetwork(layer1_size, layer2_size, name='target_critic_1')
        self.target_critic_2 = CriticNetwork(layer1_size, layer2_size, name='target_critic_2')

        self.actor.compile(optimizer=Adam(learning_rate=alpha), loss='mean')
        self.critic_1.compile(optimizer=Adam(learning_rate=beta), loss='mean_squared_error')
        self.critic_2.compile(optimizer=Adam(learning_rate=beta), loss='mean_squared_error')

        self.target_actor.compile(optimizer=Adam(learning_rate=alpha), loss='mean')
        self.target_critic_1.compile(optimizer=Adam(learning_rate=beta), loss='mean_squared_error')
        self.target_critic_2.compile(optimizer=Adam(learning_rate=beta), loss='mean_squared_error')

        self.noise = noise
        self.update_network_parameters(tau=1)


    def choose_action(self, observation):
        if self.time_step < self.warmup:
            mu = np.random.normal(scale=self.noise, size=(self.n_actions,))
        else:
            state = tf.convert_to_tensor([observation], dtype=tf.float32)
            mu = self.actor(state)[0]  # returns batch size of 1, want scalar

        mu_prime = mu + np.random.normal(scale=self.noise)

        # Scale to [0, 180] range and clip values to be within [0, 180]
        mu_prime = (mu_prime + 1) / 2 * 180
        mu_prime = np.clip(mu_prime, 0, 180)  # Ensure values are within [0, 180]

        # Optional: round values to 2 decimal places (or to nearest whole number)
        mu_prime = np.round(mu_prime, 2)

        self.time_step += 1
        return mu_prime

    def choose_greedy(self, observation):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        mu = self.actor(state)[0]  # returns batch size of 1, want scalar

        mu_prime = mu + np.random.normal(scale=self.noise)

        mu_prime = tf.clip_by_value(mu_prime, self.min_action, self.max_action)

        self.time_step += 1
        return mu_prime

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, new_states, dones = self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        new_states = tf.convert_to_tensor(new_states, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            target_actions = self.target_actor(new_states)
            target_actions = target_actions + tf.clip_by_value(np.random.normal(scale=0.2), -0.5, 0.5)

            target_actions = tf.clip_by_value(target_actions, self.min_action, self.max_action)

            q1_ = self.target_critic_1(new_states, target_actions)
            q2_ = self.target_critic_2(new_states, target_actions)

            # Shape is [batch_size, 1] want to collapse to [batch_size]
            q1_ = tf.squeeze(q1_, 1)
            q2_ = tf.squeeze(q2_, 1)

            q1 = tf.squeeze(self.critic_1(states, actions), 1)
            q2 = tf.squeeze(self.critic_2(states, actions), 1)

            critic_value_ = tf.math.minimum(q1_, q2_)

            target = rewards + self.gamma*critic_value_*(1-dones)

            critic_1_loss = keras.losses.MSE(target, q1)
            critic_2_loss = keras.losses.MSE(target, q2)

        critic_1_gradient = tape.gradient(critic_1_loss, self.critic_1.trainable_variables)
        critic_2_gradient = tape.gradient(critic_2_loss, self.critic_2.trainable_variables)

        self.critic_1.optimizer.apply_gradients(zip(critic_1_gradient, self.critic_1.trainable_variables))
        self.critic_2.optimizer.apply_gradients(zip(critic_2_gradient, self.critic_2.trainable_variables))

        self.learn_step_cntr += 1

        if self.learn_step_cntr % self.update_actor_iter != 0:
            return

        with tf.GradientTape() as tape:
            new_actions = self.actor(states)
            critic_1_value = self.critic_1(states, new_actions)
            actor_loss = -tf.math.reduce_mean(critic_1_value)

        actor_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_gradient, self.actor.trainable_variables))

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i] * (1-tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic_1.weights
        for i, weight in enumerate(self.critic_1.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_critic_1.set_weights(weights)

        weights = []
        targets = self.target_critic_2.weights
        for i, weight in enumerate(self.critic_2.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_critic_2.set_weights(weights)


    def save_models(self):
        print('... saving models ...')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.critic_1.save_weights(self.critic_1.checkpoint_file)
        self.critic_2.save_weights(self.critic_2.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.target_critic_1.save_weights(self.target_critic_1.checkpoint_file)
        self.target_critic_2.save_weights(self.target_critic_2.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.actor.load_weights(self.actor.checkpoint_file)
        self.critic_1.load_weights(self.critic_1.checkpoint_file)
        self.critic_2.load_weights(self.critic_2.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.target_critic_1.load_weights(self.target_critic_1.checkpoint_file)
        self.target_critic_2.load_weights(self.target_critic_2.checkpoint_file)

agent = Agent(alpha=0.001,
              beta=0.001,
              input_dims=(6,),
              tau=0.005,
              batch_size=128,
              layer1_size=400,
              layer2_size=400,
              n_actions=10)


def calculate_reward(state, next_state):
    reward = 0

    # Extract relevant data for clarity
    distance_to_face = next_state[3] if next_state[3] is not None else state[3] or 0
    previous_distance_to_face = state[3] if state[3] is not None else 0
    current_orientation = [val if val is not None else 0 for val in next_state[0:3]]  # Replace None with 0
    previous_orientation = [val if val is not None else 0 for val in state[0:3]]
    left_foot_contact = next_state[4]
    right_foot_contact = next_state[5]

    # Reward for reducing distance to target
    if distance_to_face < previous_distance_to_face:
        reward += 1  # Encourage moving closer to the target

    # Penalize for being too far from target
    if distance_to_face > previous_distance_to_face:
        reward -= 0.5

    # Stability reward based on orientation (minimize orientation change)
    orientation_change = sum(abs(curr - prev) for curr, prev in zip(current_orientation, previous_orientation))
    reward -= orientation_change * 0.1  # Small penalty for deviation to encourage balance

    # Ground contact reward (encourage alternating ground contact)
    if left_foot_contact and right_foot_contact:
        reward -= 0.5  # Penalize if both feet are off or both feet are on
    elif left_foot_contact or right_foot_contact:
        reward += 0.5  # Encourage stable single foot contact during movement

    # Additional penalties for extreme angles if robot falls over
    if abs(current_orientation[0]) > 180 or abs(current_orientation[1]) > 180:
        reward -= 5  # Penalize falls or extreme angles in x, y orientations

    return reward


while True:
    try:
        ser0 = serial.Serial('COM6', 115200, timeout=1.0)
        print("Successfully connected to Serial 0")
        break
    except serial.SerialException:
        print("Could not connect to Serial 0. Trying again")
        time.sleep(1)

while True:
    try:
        ser1 = serial.Serial('COM3', 115200, timeout=1.0)
        print("Successfully connected to Serial 1")
        break
    except serial.SerialException:
        print("Could not connect to Serial 1. Trying again")
        time.sleep(1)

time.sleep(3)
ser0.reset_input_buffer()
ser1.reset_input_buffer()

score_history = []
reward_threshold = -200

print("Serial OK")

try:
    def get_sensor_data():
        # Variables
        x = None
        y = None
        z = None
        d = None
        p1 = None
        p2 = None

        # Face Distance Data
        success, img = cap.read(1)
        img, faces = detector.findFaceMesh(img, draw=False)

        if faces:
            face = faces[0]
            pointLeft = face[145]
            pointRight = face[374]

            w, _ = detector.findDistance(pointLeft, pointRight)
            W = 6.3

            # Finding distance
            f = 643
            d = (W * f) / w
        #cv2.waitKey(1)

        # Get X,Y,Z coordinates
        if ser0.in_waiting > 0:
            msg = ser0.readline().decode("utf-8").rstrip()
            message_list = msg.split()

            for i in range(len(message_list)):
                if message_list[i].isdigit():
                    if message_list[i - 1] == "X=":
                        x = int(message_list[i])
                    elif message_list[i - 1] == "Y=":
                        y = int(message_list[i])
                    elif message_list[i - 1] == "Z=":
                        z = int(message_list[i])


        # Get pressure plate data
        if ser1.in_waiting > 0:
            msg1 = ser1.readline().decode("utf-8").rstrip()
            if msg1 == "p1 & p2":
                p1 = True
                p2 = True
            elif msg1 == "p1":
                p1 = True
                p2 = False
            elif msg1 == "p2":
                p1 = False
                p2 = True
            else:
                p1 = False
                p2 = False

        return [x, y, z, d, p1, p2]


    best_score = 0
    reward_threshold = 200
    epochs = []
    score_hist = []

    # ---------------------------------------------------- For Training -------------------------------------------------- #
    while True:
        state = get_sensor_data()
        score = 0
        reward = None
        step = 0

        while True:  # Continuously train
            step += 1
            print(state)
            action = agent.choose_action(np.array(state))
            actioncmd = "S:" + ",".join(map(str, action)) + "\n"
            ser1.write(actioncmd.encode('utf-8'))
            print(action)

            # Get next state after applying action
            next_state = get_sensor_data()

            # Calculate reward
            reward = calculate_reward(state, next_state)

            # Store the transition in memory
            agent.remember(state, action, reward, next_state, done=False)

            # Train the agent with the stored data
            agent.learn()

            # Track the score (you can remove this if you don't need it)
            score += reward
            state = next_state

            print(f"Score: {score}")  # Track score if needed
            if reward < -999 or reward > 999:
                score_hist.append(score)
                epochs.append(step)
                score = 0
                step = 0



            # Optional: Save models periodically or based on other conditions
            if score > reward_threshold:
                print("New best score!")
                # agent.save_models()  # Uncomment to save the model periodically

            tf.keras.backend.clear_session()  # Clearing the session can help with memory management
            time.sleep(0.5)


except KeyboardInterrupt:
    print("Close Serial communication")
    ser0.close()
    ser1.close()