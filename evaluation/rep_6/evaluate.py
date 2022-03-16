
from keras.layers import Conv1D,Dense, Activation
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from joblib import dump, load
import sys



def setEvaluationFilesPath(video):
    return '../evaluationFiles/'+video+'/rep_6/'





MODEL_NAME='GreenABR'
if len(sys.argv)<2 :
    print('Please run the script with one of the test videos (tos,bbb,doc). i.e. python evaluate.py tos')
    quit()
else:
    VIDEO=sys.argv[1]
    if VIDEO not in ['tos','bbb','doc']:
        print('You must use one of the test videos, [tos,bbb,doc]')
        quit()
S_DIM=7
A_DIM = 6

VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 44
M_IN_K = 1000.0
M_IN_N = 1000000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000

REWARD_FOLDER='reward_logs/'
TEST_LOG_FOLDER = 'test_results/'
TEST_TRACES = '../test_sim_traces/'
EVAL_FILE_PATH=setEvaluationFilesPath(VIDEO)
PHONE_VMAF = pd.read_csv(EVAL_FILE_PATH+'vmaf_phone.csv')
REGULAR_VMAF = pd.read_csv(EVAL_FILE_PATH+'vmaf_phone.csv')
POWER_ATTRIBUTES= pd.read_csv('../power_attributes.csv')
POWER_MES=pd.read_csv(EVAL_FILE_PATH+'power_measurements.csv')

BITRATE_MAX=12000.0
FILE_SIZE_MAX=1775324.0
QUALITY_MAX=100.0
MOTION_MAX=20.15
PIXEL_RATE_MAX=2073600.0
POWER_MAX=1690.0



def load_trace(cooked_trace_folder=TEST_TRACES):
    cooked_files = os.listdir(cooked_trace_folder)
    if '.DS_Store' in cooked_files:
        cooked_files.remove('.DS_Store')
    all_cooked_time = []
    all_cooked_bw = []
    all_file_names = []
    for cooked_file in cooked_files:
        file_path = cooked_trace_folder + cooked_file
        cooked_time = []
        cooked_bw = []
        # print file_path
        with open(file_path, 'r') as f:
            for line in f:
                parse = line.split()
                if len(parse)>0:
#                     print(parse[0], parse[1])
                    cooked_time.append(float(parse[0]))
                    cooked_bw.append(float(parse[1]))
        all_cooked_time.append(cooked_time)
        all_cooked_bw.append(cooked_bw)
        all_file_names.append(cooked_file)

    return all_cooked_time, all_cooked_bw, all_file_names




# power model parameters for network power 
p_alpha = 210
p_betha = 28

SEGMENT_SIZE = 4.0
power_threshold = 2500
byte_to_KB = 1000
KB_to_MB=1000.0

def Estimate_Network_Power_Consumption(thr, chunk_file_size):
    return (chunk_file_size * (p_alpha*1/thr+p_betha))

MIN_ENERGY=1000
NORMALIZATION_SCALAR=1000
SCALING_FACTOR=-1.2
def Calculate_Energy_Penalty(energy_chunk):
    penalty= ((energy_chunk-MIN_ENERGY)/NORMALIZATION_SCALAR)**2/SCALING_FACTOR
    return min(0,penalty)


def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(7, input_dim=5, kernel_initializer='normal', activation='relu'))
    model.add(Dense(20, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model



class Environment:
    def __init__(self, all_cooked_time, all_cooked_bw, random_seed=RANDOM_SEED):
        self.MILLISECONDS_IN_SECOND = 1000.0
        self.B_IN_MB = 1000000.0
        self.BITS_IN_BYTE = 8.0
        self.RANDOM_SEED = 42
        self.VIDEO_CHUNCK_LEN = 4000.0  # millisec, every time add this amount to buffer
        self.BITRATE_LEVELS = 6
        self.TOTAL_VIDEO_CHUNCK = 44
        self.BUFFER_THRESH = 60000.0  # millisec, max buffer limit
        self.DRAIN_BUFFER_SLEEP_TIME = 500.0  # millisec
        self.PACKET_PAYLOAD_PORTION = 0.95
        self.LINK_RTT = 80  # millisec
        self.PACKET_SIZE = 1500  # bytes
        self.VIDEO_SIZE_FILE = EVAL_FILE_PATH+'video_size_'
        assert len(all_cooked_time) == len(all_cooked_bw)

        self.local_power_model=load('../power_model.joblib')

        np.random.seed(random_seed)

        self.all_cooked_time = all_cooked_time
        self.all_cooked_bw = all_cooked_bw

        self.video_chunk_counter = 0
        self.buffer_size = 0

        # pick a random trace file
        self.trace_idx = 0
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]

        # randomize the start point of the trace
        # note: trace file starts with time 0
        self.mahimahi_start_ptr = 1
        self.mahimahi_ptr = 1
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        self.video_size = {}  # in bytes
        for bitrate in range(self.BITRATE_LEVELS):
            self.video_size[bitrate] = []
            with open(self.VIDEO_SIZE_FILE + str(bitrate)) as f:
                for line in f:
                    self.video_size[bitrate].append(int(line.split()[0]))

  


    def reset(self):
        self.video_chunk_counter = 0
        self.buffer_size = 0

        # pick a random trace file
        # self.trace_idx = 0
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]

        # randomize the start point of the trace
        # note: trace file starts with time 0
        self.mahimahi_ptr = 1
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        self.video_size = {}  # in bytes
        for bitrate in range(self.BITRATE_LEVELS):
            self.video_size[bitrate] = []
            with open(self.VIDEO_SIZE_FILE + str(bitrate)) as f:
                for line in f:
                    self.video_size[bitrate].append(int(line.split()[0]))

        return np.zeros(S_DIM)

    def get_video_chunk(self, quality):

        assert quality >= 0
        assert quality < self.BITRATE_LEVELS

        video_chunk_size = self.video_size[quality][self.video_chunk_counter]

        # use the delivery opportunity in mahimahi
        delay = 0.0  # in ms
        video_chunk_counter_sent = 0  # in bytes

        while True:  # download video chunk over mahimahi
            throughput = self.cooked_bw[self.mahimahi_ptr] * self.B_IN_MB / self.BITS_IN_BYTE
            duration = self.cooked_time[self.mahimahi_ptr] - self.last_mahimahi_time

            packet_payload = throughput * duration * self.PACKET_PAYLOAD_PORTION

            if video_chunk_counter_sent + packet_payload > video_chunk_size:

                fractional_time = (video_chunk_size - video_chunk_counter_sent) /throughput / self.PACKET_PAYLOAD_PORTION
                delay += fractional_time
                self.last_mahimahi_time += fractional_time
                assert(self.last_mahimahi_time <= self.cooked_time[self.mahimahi_ptr])
                break

            video_chunk_counter_sent += packet_payload
            delay += duration
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
            self.mahimahi_ptr += 1

            if self.mahimahi_ptr >= len(self.cooked_bw):
            # loop back in the beginning
            # note: trace file starts with time 0
                self.mahimahi_ptr = 1
                self.last_mahimahi_time = 0

        delay *= self.MILLISECONDS_IN_SECOND
        delay += self.LINK_RTT

    # add a multiplicative noise to the delay
        # delay *= np.random.uniform(self.NOISE_LOW, self.NOISE_HIGH)

        # rebuffer time
        rebuf = np.maximum(delay - self.buffer_size, 0.0)

        # update the buffer
        self.buffer_size = np.maximum(self.buffer_size - delay, 0.0)

        # add in the new chunk
        self.buffer_size += self.VIDEO_CHUNCK_LEN

        # sleep if buffer gets too large
        sleep_time = 0
        if self.buffer_size > self.BUFFER_THRESH:
          # exceed the buffer limit
          # we need to skip some network bandwidth here
          # but do not add up the delay
            drain_buffer_time = self.buffer_size - self.BUFFER_THRESH
            sleep_time = np.ceil(drain_buffer_time / self.DRAIN_BUFFER_SLEEP_TIME) * self.DRAIN_BUFFER_SLEEP_TIME
            self.buffer_size -= sleep_time

            while True:
                duration = self.cooked_time[self.mahimahi_ptr] - self.last_mahimahi_time
                if duration > sleep_time / self.MILLISECONDS_IN_SECOND:
                    self.last_mahimahi_time += sleep_time / self.MILLISECONDS_IN_SECOND
                    break
                sleep_time -= duration * self.MILLISECONDS_IN_SECOND
                self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
                self.mahimahi_ptr += 1

                if self.mahimahi_ptr >= len(self.cooked_bw):
                  # loop back in the beginning
                  # note: trace file starts with time 0
                    self.mahimahi_ptr = 1
                    self.last_mahimahi_time = 0

        # the "last buffer size" return to the controller
        # Note: in old version of dash the lowest buffer is 0.
        # In the new version the buffer always have at least
        # one chunk of video
        return_buffer_size = self.buffer_size

        self.video_chunk_counter += 1
        video_chunk_remain = self.TOTAL_VIDEO_CHUNCK - self.video_chunk_counter

        end_of_video = False
        if self.video_chunk_counter >= self.TOTAL_VIDEO_CHUNCK:
            end_of_video = True
            self.buffer_size = 0
            self.video_chunk_counter = 0

            # pick a random trace file
            self.trace_idx += 1
            if self.trace_idx >= len(self.all_cooked_time):
                self.trace_idx = 0  
            self.cooked_time = self.all_cooked_time[self.trace_idx]
            self.cooked_bw = self.all_cooked_bw[self.trace_idx]

            # randomize the start point of the video
            # note: trace file starts with time 0
            self.mahimahi_ptr = self.mahimahi_start_ptr
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        next_video_chunk_sizes = []
        for i in range(self.BITRATE_LEVELS):
            next_video_chunk_sizes.append(self.video_size[i][self.video_chunk_counter])

        return self.video_chunk_counter,delay,sleep_time,return_buffer_size / self.MILLISECONDS_IN_SECOND, rebuf / self.MILLISECONDS_IN_SECOND, video_chunk_size,next_video_chunk_sizes, end_of_video, video_chunk_remain
    def normalize_parameters(self,bitrate,t_sec):
        time=self.video_chunk_counter*SEGMENT_SIZE+t_sec
        b_n= VIDEO_BIT_RATE[bitrate] / BITRATE_MAX
        d=POWER_ATTRIBUTES[(POWER_ATTRIBUTES['Bitrate']==VIDEO_BIT_RATE[bitrate]) & 
                         (POWER_ATTRIBUTES['Time']==time)]
        f_n=d['FileSize']/FILE_SIZE_MAX
        q_n= d['Quality']/QUALITY_MAX
        m_n=d['Motion']/MOTION_MAX
        p_n=d['PixelRate']/PIXEL_RATE_MAX
        return np.reshape(np.array([b_n,f_n,q_n,m_n,p_n]),(1,5))
        
    
    def calculate_local_energy(self, bitrate):
        total_energy=0
        for i in range(int(SEGMENT_SIZE)):
            pars=self.normalize_parameters(bitrate,i)
            pred=self.local_power_model.predict(pars)
            power=pred[0]*POWER_MAX # equal to energy as it is for 1 second
            total_energy+=power
        return total_energy


    def step(self, bit_rate, last_bitrate):
        video_chunk_counter,delay, sleep_time, buffer_size, rebuf, video_chunk_size, next_video_chunk_sizes, end_of_video, video_chunk_remain = self.get_video_chunk(bit_rate)

        throughput=float(video_chunk_size) / float(delay) / M_IN_K
        new_state= np.zeros(S_DIM)
        new_state[0]=VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))
        new_state[1]=buffer_size / BUFFER_NORM_FACTOR
        new_state[2]=throughput
        new_state[3]=float(delay) / M_IN_K / BUFFER_NORM_FACTOR
        new_state[4]=np.minimum(video_chunk_remain,CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)


        log=np.zeros(6)
        log[0]=self.TOTAL_VIDEO_CHUNCK if video_chunk_counter==0 else video_chunk_counter
        log[1]=delay
        log[2]=sleep_time
        log[3]=buffer_size
        log[4]=rebuf
        log[5]=video_chunk_size

        quality=(PHONE_VMAF['VMAF_' + str(bit_rate+1)][log[0]-1])
        new_state[6]=quality

        estimated_energy=(POWER_MES['P_' + str(bit_rate+1)][log[0]-1])
        new_state[5]= estimated_energy


        reward=0
        quality=quality/20.0
        quality_reward=quality

        if log[0]==1:
            rebuffer_penalty=0.0
        else:
            rebuffer_penalty= REBUF_PENALTY*rebuf

        if log[0]==1:
            smooth_penalty=0.0
        else:
            smooth_penalty=SMOOTH_PENALTY* np.abs((PHONE_VMAF['VMAF_' + str(bit_rate+1)][log[0]-1]/20.0)- (PHONE_VMAF['VMAF_' + str(last_bitrate+1)][log[0]-2]/20.0))

        reward=quality_reward - rebuffer_penalty - smooth_penalty

        energy_penalty=Calculate_Energy_Penalty(estimated_energy)

        return new_state, reward, end_of_video, estimated_energy, video_chunk_size, quality, rebuffer_penalty, smooth_penalty,energy_penalty, log



class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.int8)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        # store one hot encoding of actions, if appropriate
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal



def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):
    model = Sequential([
                Dense(fc1_dims, input_shape=(input_dims,)),
                Activation('relu'),
                Dense(fc2_dims),
                Activation('relu'),
                Dense(128), 
                Activation('relu'), 
                Dense(64),
                Activation('relu'),
                Dense(n_actions)])

    model.compile(optimizer=Adam(lr=lr), loss='mse')
    return model



class DDQNAgent(object):
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size,
                 input_dims,epsilon_dec=0.996,  epsilon_end=0.01,
                 mem_size=50000, fname=MODEL_NAME, replace_target=100):
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = MODEL_NAME+".h5"
        self.replace_target = replace_target
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions,
                                   discrete=True)
        self.q_eval = build_dqn(alpha, n_actions, input_dims,256, 256)
        self.q_target = build_dqn(alpha, n_actions, input_dims, 256,256)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):
        state = state[np.newaxis, :]
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)

        return action

    def learn(self):
        if self.memory.mem_cntr > self.batch_size:
            state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

            action_values = np.array(self.action_space, dtype=np.int8)
            action_indices = np.dot(action, action_values)

            q_next = self.q_target.predict(new_state)
            q_eval = self.q_eval.predict(new_state)
            q_pred = self.q_eval.predict(state)

            max_actions = np.argmax(q_eval, axis=1)

            q_target = q_pred

            batch_index = np.arange(self.batch_size, dtype=np.int32)

            q_target[batch_index, action_indices] = reward + self.gamma*q_next[batch_index, max_actions.astype(int)]*done

            _ = self.q_eval.fit(state, q_target, verbose=0)

            self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min
            if self.memory.mem_cntr % self.replace_target == 0:
                self.update_network_parameters()

    def update_network_parameters(self):
        self.q_target.set_weights(self.q_eval.get_weights())

    def save_model(self):
        self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval = load_model(self.model_file)
        # if we are in evaluation mode we want to use the best weights for
        # q_target
        if self.epsilon == 0.0:
            self.update_network_parameters()


# In[175]:


def main():
    REWARD_MODEL=MODEL_NAME
    
    
    all_cooked_time, all_cooked_bw, all_file_names = load_trace(TEST_TRACES)

    input_dims=7

    env=Environment(all_cooked_time, all_cooked_bw, RANDOM_SEED)
    ddqn_agent = DDQNAgent(alpha=0.001, gamma=0.99, n_actions=A_DIM, epsilon=0.0,batch_size=64,input_dims=input_dims)

    ddqn_agent.load_model()

    log_path = TEST_LOG_FOLDER+VIDEO+'/log_'+REWARD_MODEL+'_reward_' + all_file_names[env.trace_idx]
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, 'w')

    reward_log_path = 'reward_logs/'+VIDEO+'/log_'+REWARD_MODEL+'_reward_' + all_file_names[env.trace_idx]
    os.makedirs(os.path.dirname(reward_log_path), exist_ok=True)
    reward_log_file = open(reward_log_path, 'w')

    trace_log_path='reward_logs/'+VIDEO+"_log_"+REWARD_MODEL
    os.makedirs(os.path.dirname(trace_log_path), exist_ok=True)
    trace_log=open(trace_log_path, 'w')

    video_count=0
    for video_count in range(len(all_file_names)):
        done = False
        score = 0
        total_energy=0
        total_data=0
        last_bit_rate=1
        total_quality=0
        total_rebuffer_pen=0
        total_rebuffer_time=0
        total_smooth_pen=0
        total_smooth_time=0
        total_energy_penalty=0
        time_stamp=0
        first_chunk=True
        log_path =  TEST_LOG_FOLDER+VIDEO+'/log_'+REWARD_MODEL+'_reward_' + all_file_names[env.trace_idx]
        log_file = open(log_path, 'w')
        log_file.write('video_chunk' + '\t' +
                           'bitrate' + '\t' +
                           'buffer_size' + '\t' +
                           'rebuf' + '\t' +
                           'video_chunk_size' + '\t' +
                           'delay' + '\t' +
                           'phone_vmaf' + '\t' +
                           'regular_vmaf' + '\t' +
                           'energy' + '\t' +
                           'reward' + '\n'
                           )

        observation=env.reset()
        while not done:
            if first_chunk:
                action=0
            else :
                action = ddqn_agent.choose_action(observation)
            observation_, reward, done, energy, data,quality, rebuffer_penalty, smooth_penalty, energy_penalty, log= env.step(action,last_bit_rate)
            first_chunk=False
            score += reward
            total_energy+=energy
            total_data+=data
            total_quality+=quality
            total_rebuffer_pen=+rebuffer_penalty
            if rebuffer_penalty > 0:
                total_rebuffer_time+=1
            total_smooth_pen+=smooth_penalty
            if smooth_penalty > 0:
                total_smooth_time +=1
            total_energy_penalty+=energy_penalty

            time_stamp+=log[1]
            time_stamp+=log[2]
            # print("the chunk counter is ", str(log[0]))
            log_file.write(str(log[0]-1) + '\t' +
                           str(VIDEO_BIT_RATE[action]) + '\t' +
                           str(log[3]) + '\t' +
                           str(log[4]) + '\t' +
                           str(log[5]) + '\t' +
                           str(log[1]) + '\t' +
                           str(PHONE_VMAF['VMAF_' + str(action+1)][log[0]-1]) + '\t' +
                           str(REGULAR_VMAF['VMAF_' + str(action+1)][log[0]-1]) + '\t' +
                           str(energy) + '\t' +
                           str(reward) + '\n'
                           )

            log_file.flush()

            reward_log_file.write(str(log[0]) + '\t' +
                           str(time_stamp) + '\t' +
                           str(VIDEO_BIT_RATE[action]) + '\t' +
                           str(quality) + '\t' +
                           str(rebuffer_penalty) + '\t' +
                           str(smooth_penalty) + '\t' +
                           str(energy_penalty) + '\t' +
                           str(PHONE_VMAF['VMAF_' + str(action+1)][log[0]-1]) + '\t' +
                           str(REGULAR_VMAF['VMAF_' + str(action+1)][log[0]-1]) + '\t' +
                           str(energy) + '\t' +
                           str(reward) + '\n'
                           )

            reward_log_file.flush()
            observation = observation_
            last_bit_rate=action

        log_file.write('\n')
        log_file.close()
        reward_log_file.write('\n')
        reward_log_file.close()
        print('Completed trace ', all_file_names[env.trace_idx])
        print('Trace: ', video_count,'score: %.2f' % score)
        trace_log.write(str(video_count+1)+'\t'+
                        str(all_file_names[env.trace_idx]+'\t')+
                        str(score)+'\t'+
                        str(total_energy)+'\t'+
                        str(total_data)+'\t'+
                        str(total_quality)+'\t'+
                        str(total_rebuffer_pen)+'\t'+
                        str(total_rebuffer_time)+'\t'+
                        str(total_smooth_pen)+'\t'+
                        str(total_smooth_time)+'\t'+
                        str(total_energy_penalty)+'\n')
        trace_log.flush()
        video_count+=1
        reward_log_path = 'reward_logs/'+VIDEO+'/log_'+REWARD_MODEL+'_reward_' + all_file_names[env.trace_idx]
        reward_log_file = open(reward_log_path, 'w')

    trace_log.write('\n')
    trace_log.close()    



if __name__ == "__main__":
    main()