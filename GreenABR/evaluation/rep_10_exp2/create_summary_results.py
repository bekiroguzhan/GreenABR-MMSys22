
import pandas as pd 
import os
import numpy as np


SUMMARY_RESULTS='summaryResults/'
NUM_BINS = 100
BITS_IN_BYTE = 8.0
MILLISEC_IN_SEC = 1000.0
M_IN_B = 1000000.0
VIDEO_LEN = 44
K_IN_M = 1000.0
K_IN_B=1000.0
REBUF_P = 4.3
SMOOTH_P = 1

POWER_RANGE= 648 #Difference between max and min avg power
BASE_POWER_XCOVER=1800.0 
BASE_POWER_GALAXY=1016.0 #Power consumption without streaming 



########################################
###### This part is added for network power estimation 
p_alpha = 210
p_betha = 28

SEGMENT_SIZE = 4.0
power_threshold = 1500
byte_to_KB = 1000
KB_to_MB=1000.0

def Estimate_Network_Power_Consumption(thr, chunk_file_size):
    return (chunk_file_size * (p_alpha*1/thr+p_betha))
########################################################




# QOE calculation for the entire session based on the standard model 
def calculateQOE(total_vmaf, total_rebuf, total_rebuf_count, total_vmaf_change, total_smooth_count):
    return 0.07713539*total_vmaf-1.24971639*total_rebuf -2.87757412*total_rebuf_count -0.04938335*total_vmaf_change -1.436473*total_smooth_count


def getStutData(video):
    if video=='tos':
        return pd.read_csv('../EvaluationFiles/stuttering_data/tos_stut_summary.csv')
    elif video=='bbb':
        return pd.read_csv('../EvaluationFiles/stuttering_data/bbb_stut_summary.csv')
    elif video=='doc':
        return pd.read_csv('../EvaluationFiles/stuttering_data/doc_stut_summary.csv')
    else:
        return None

# aggregates the result for each streaming session 
def CreateSummaryResults(video, model):
   log_dir='./test_results/'+video+'/'
   energy_g_dir='../powerMeasurementFiles/'+'exp2_galaxy_'+video.upper()+'_streaming.csv'
   energy_x_dir='../powerMeasurementFiles/'+'exp2_xcover_'+video.upper()+'_streaming.csv'
   en_g=pd.read_csv(energy_g_dir)
   en_x=pd.read_csv(energy_x_dir)
   stut_data=getStutData(video)
   # summary_data=pd.DataFrame(columns=['Trace','QoE','Bitrate','Throughput','Buffer_Size','VMAF','Rebuffer_Time','Rebuffer_Count','Smoothness_Change','Data','Power_G','Power_X'])
   traces=[]
   for l in os.listdir(log_dir):
       if l.split('_', maxsplit=3)[1]==model:
           traces.append(l.split('_', maxsplit=3)[3])
   
   traces.sort()
#     print(traces)

   summary_data=pd.DataFrame(index=traces,columns=['Efficiency_Avg','Efficiency_X','Efficiency_G','QoE','Bitrate','Throughput','Buffer_Size','VMAF','Rebuffer_Time','Rebuffer_Count','Smoothness_Change','Data','Energy_Avg','Energy_G','Energy_X','Stuttering'])

   for l in os.listdir(log_dir):
       if l.split('_', maxsplit=3)[1]==model:
           trace=l.split('_', maxsplit=3)[3]
           with open(log_dir + l, 'r') as f:
               last_vmaf=0
               rebuffer_c=0   # rebuffer events counter
               bit_rate = []
               buffer_size=[]
               rebuffer=[]
               data_size = []
               throughput =[]
               vmaf =[]
               energy_g = []
               energy_x = []

               smooth=[] #smoothness change counter 
               vmaf_change=[]
               stuttering=[]

               for line in f:
                   parse = line.split()
                   if len(parse) <= 1:
                       break
                   if parse[0]=='video_chunk':
                       continue
                   bit_rate.append(int(parse[1]))
                   buffer_size.append(float(parse[2]))
                   rebuffer.append(float(parse[3]))
                   if (float(parse[3])>0) & (float(parse[0])>0):
                       rebuffer_c=rebuffer_c+1
                   data_size.append(float(parse[4]))
                   thr=float(parse[4]) / float(parse[5])* BITS_IN_BYTE * MILLISEC_IN_SEC / M_IN_B
                   throughput.append(thr)
                   vmaf.append(float(parse[6]))

                   stuttering.append(stut_data[parse[1]][float(parse[0])])
                   
                   smooth_change=0.0
                   if float(parse[0])>1:
                       smooth_change= (np.abs(float(parse[6])-last_vmaf))//20.0
                       vmaf_change.append(np.abs(float(parse[6])-last_vmaf))

                   smooth.append(smooth_change)
                   last_vmaf=float(parse[6])
                   ###################################################################################
                   ########## this part is updated for power 
                   if (float(parse[0])>0) & (float(parse[0])<float(VIDEO_LEN)):
                       network_energy=Estimate_Network_Power_Consumption(thr,float(parse[4])/1000000.0) # network energy

                       local_energy_g=en_g[parse[1]][float(parse[0])] # measured local playback energy  
                       rebuf_energy_g=BASE_POWER_GALAXY*float(parse[3]) # additional energy due to rebuffering
                       total_energy_g=local_energy_g+network_energy+rebuf_energy_g# total energy
                       energy_g.append(total_energy_g)

                       local_energy_x=en_x[parse[1]][float(parse[0])] # measured local playback energy  
                       rebuf_energy_x=BASE_POWER_XCOVER*float(parse[3]) # additional energy due to rebuffering
                       total_energy_x=local_energy_x+network_energy+rebuf_energy_x# total energy
                       energy_x.append(total_energy_x)

                   else:
                       measured_power=0.0
                       energy_g.append(measured_power)
                       energy_x.append(measured_power)
                   
                   
               qoe=calculateQOE(np.sum(vmaf), np.sum(rebuffer), rebuffer_c, np.sum(vmaf_change), np.sum(smooth))
               eff_avg=(qoe/np.sum(energy_x)*1000+qoe/np.sum(energy_g)*1000)/2
               en_avg=(np.sum(energy_x)+np.sum(energy_g))/2
               summary_data.loc[trace]=[eff_avg,qoe/np.sum(energy_x)*1000,qoe/np.sum(energy_g)*1000,qoe, np.sum(bit_rate), np.sum(throughput), np.sum(buffer_size), np.sum(vmaf),np.sum(rebuffer), rebuffer_c, np.sum(smooth), np.sum(data_size),en_avg, np.sum(energy_g), np.sum(energy_x), np.sum(stuttering)]
               
   summary_data.to_csv(SUMMARY_RESULTS+ video+'_'+model+'_summaryResults.csv')




# create average results over all test videos 
def createAvgFiles():
    videos=['tos','bbb','doc']
    models=['BB','Bolae','DD','Dynamic','Throughput','GreenABR','Pensieve']
    for m in models:
        d_t=pd.read_csv('./summaryResults/tos_'+m+'_summaryResults.csv')
        d_d=pd.read_csv('./summaryResults/doc_'+m+'_summaryResults.csv')
        d_b=pd.read_csv('./summaryResults/bbb_'+m+'_summaryResults.csv')

        df_concat = pd.concat((d_t, d_d,d_b))
        by_row_index = df_concat.groupby(df_concat.index)
        df_means = by_row_index.mean()
        df_means.to_csv('./summaryResults/'+m+'_avg.csv')




def main():
    videos=['tos','bbb','doc']
    models=['BB','Bolae','DD','Dynamic','Throughput','GreenABR','Pensieve']
    for v in videos:
        for m in models:
            CreateSummaryResults(v,m)
    createAvgFiles()



if __name__ == '__main__':
    main()




