import pandas as pd 
import numpy as np 
import os 
import matplotlib.pyplot as plt




# simplified version of the GreenABR training QoE model 
# it does not include the energy component and nonlinear parts
def calc_score_gabr(l):
    data=pd.read_csv('./streaming_logs/'+l)
    qoe_a=[]
    for i in range(len(data)):
        qoe=data['vmaf'].iloc[i]/20-data['rebuffering_duration'].iloc[i]*4.3
        if i>0:
            qoe=1.0*(qoe-(data['vmaf'].iloc[i]-data['vmaf'].iloc[i-1])/20)
        qoe_a.append(qoe)
    return np.mean(qoe_a)


def evaluateTrainGABR():
	calculated_logs=[]
	for l in os.listdir('./streaming_logs/'):
	    calculated_logs.append([l,calc_score_gabr(l)])
	calc_data=pd.DataFrame(calculated_logs, columns=['streaming_log','qoe'])
	calc_data=calc_data.sort_values(by='streaming_log')
	merged_data=pd.merge(mos_data, calc_data, on='streaming_log')
	cr=merged_data[['mos','qoe']].corr(method='spearman')
	return np.round(cr.iloc[0][1], decimals=4)


# calculates the QoE based on Comyco QoE model 
def calculate_QOE_comyco(vmaf, last_vmaf, rebuf):
    qoe= vmaf*0.8469 - rebuf*28.7959
    if vmaf > last_vmaf:
        qoe=qoe+0.2979*(vmaf-last_vmaf)
    else:
        qoe=qoe-1.0610*(last_vmaf-vmaf)
    return qoe
def calc_score_cmc(l):
    data=pd.read_csv('./streaming_logs/'+l)
    qoe_a=[]
    for i in range(len(data)):
        if i>0:
            qoe_a.append(calculate_QOE_comyco(data['vmaf'].iloc[i],data['vmaf'].iloc[i-1],data['rebuffering_duration'].iloc[i]))
        else:
            qoe_a.append(calculate_QOE_comyco(data['vmaf'].iloc[i],data['vmaf'].iloc[i],data['rebuffering_duration'].iloc[i]))             
    return np.mean(qoe_a)


def evaluateComyco():
	calculated_logs=[]
	for l in os.listdir('./streaming_logs/'):
	    calculated_logs.append([l,calc_score_cmc(l)])
	calc_data=pd.DataFrame(calculated_logs, columns=['streaming_log','qoe'])
	calc_data=calc_data.sort_values(by='streaming_log')
	merged_data=pd.merge(mos_data, calc_data, on='streaming_log')
	cr=merged_data[['mos','qoe']].corr(method='spearman')
	return np.round(cr.iloc[0][1], decimals=4)

# calculate the QoE based on Pensieve model 
def calc_score_pen(l):
    data=pd.read_csv('./streaming_logs/'+l)
    qoe_a=[]
    for i in range(len(data)):
        qoe=data['video_bitrate'].iloc[i]/1000-data['rebuffering_duration'].iloc[i]*4.3
        if i>0:
            qoe=qoe-(data['video_bitrate'].iloc[i]-data['video_bitrate'].iloc[i-1])/1000
        qoe_a.append(qoe)
    return np.mean(qoe_a)

def evaluatePensieve():
	calculated_logs=[]
	for l in os.listdir('./streaming_logs/'):
	    calculated_logs.append([l,calc_score_pen(l)])
	calc_data=pd.DataFrame(calculated_logs, columns=['streaming_log','qoe'])
	calc_data=calc_data.sort_values(by='streaming_log')
	merged_data=pd.merge(mos_data, calc_data, on='streaming_log')
	cr=merged_data[['mos','qoe']].corr(method='spearman')
	return np.round(cr.iloc[0][1], decimals=4)


# calculate QoE based on the sample model in Waterloo Streaming dataset
def calc_score_wsqoe(l):
    data=pd.read_csv('./streaming_logs/'+l)
    qoe_a=[]
    pr=data['rebuffering_duration'].sum()/data['chunk_duration'].sum()
    b_=data['video_bitrate'].mean()
    total_bs=0
    total_bs_num=0
    for i in range(len(data)):
        if i>0:
            if data['video_bitrate'].iloc[i] != data['video_bitrate'].iloc[i-1]:
                total_bs=total_bs+np.abs(data['video_bitrate'].iloc[i]-data['video_bitrate'].iloc[i-1])
                total_bs_num=total_bs_num+1
    bs_=total_bs/total_bs_num
    return -63.1*pr+0.0079*b_+0.0010*bs_+49.7

def evaluateWSQOE():
	calculated_logs=[]
	for l in os.listdir('./streaming_logs/'):
	    calculated_logs.append([l,calc_score_wsqoe(l)])
	calc_data=pd.DataFrame(calculated_logs, columns=['streaming_log','qoe'])
	calc_data=calc_data.sort_values(by='streaming_log')
	merged_data=pd.merge(mos_data, calc_data, on='streaming_log')
	cr=merged_data[['mos','qoe']].corr(method='spearman')
	return np.round(cr.iloc[0][1], decimals=4)


# calculates QoE based on proposed standard QoE model used for the evaluation section
def calc_score_proposed(l):
    data=pd.read_csv('./streaming_logs/'+l)
    total_vmaf=data['vmaf'].sum()
    total_rebuf=data['rebuffering_duration'].sum()
    total_rebuf_count=0
    total_vmaf_change=0
    total_smooth_count=0
    for i in range(len(data)):
        if i>0:
            total_vmaf_change=total_vmaf_change+np.abs(data['vmaf'].iloc[i]-data['vmaf'].iloc[i-1])
            if np.abs(data['vmaf'].iloc[i]-data['vmaf'].iloc[i-1]) > 20:
                total_smooth_count=total_smooth_count+(np.abs(data['vmaf'].iloc[i]-data['vmaf'].iloc[i-1]))//20
            if data['rebuffering_duration'].iloc[i]>0:
                total_rebuf_count+=1
                
#     0.07713539 -1.24971639 -2.87757412 -0.04938335 -1.436473
    return 0.07713539*total_vmaf-1.24971639*total_rebuf -2.87757412*total_rebuf_count -0.04938335*total_vmaf_change -1.436473*total_smooth_count

def evaluateProposedQoE():
	calculated_logs=[]
	for l in os.listdir('./streaming_logs/'):
	    calculated_logs.append([l,calc_score_proposed(l)])
	calc_data=pd.DataFrame(calculated_logs, columns=['streaming_log','qoe'])
	calc_data=calc_data.sort_values(by='streaming_log')
	merged_data=pd.merge(mos_data, calc_data, on='streaming_log')
	cr=merged_data[['mos','qoe']].corr(method='spearman')
	return np.round(cr.iloc[0][1], decimals=4)


mos_data=pd.read_csv('data.csv')[['streaming_log','mos']]

train_model_score=evaluateTrainGABR()
pensieve_model_score=evaluatePensieve()
comyco_model_score=evaluateComyco()
wsqoe_model_score=evaluateWSQOE()
proposed_model_score=evaluateProposedQoE()


print ('{:<20s}{:>20s}{:>20s}{:>20s}{:>20s}'.format('Training QoE', 'Pensieve', 'Comyco','WSQOE','Proposed QoE'))

print ('{:<20s}{:>20s}{:>20s}{:>20s}{:>20s}'.format(str(train_model_score), str(pensieve_model_score), str(comyco_model_score),str(wsqoe_model_score),str(proposed_model_score)))
