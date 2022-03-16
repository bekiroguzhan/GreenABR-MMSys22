import os
import re

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict


def rgb_to_hex(rr, gg, bb):
    rgb = (rr, gg, bb)
    return '#%02x%02x%02x' % rgb

colors = [
            rgb_to_hex(44, 160, 44), #green
            rgb_to_hex(255, 127, 14), #orange
            # rgb_to_hex(148, 103, 189), #lavender
            rgb_to_hex(128,0,128), #purple
            rgb_to_hex(214, 39, 40), #red
            
            rgb_to_hex(140, 86, 75), #brown
            rgb_to_hex(31, 119, 180), #blue
            rgb_to_hex(0, 0, 0), #black
            rgb_to_hex(227, 119, 194), #pink
            rgb_to_hex(127, 127, 127), #grey
            rgb_to_hex(188, 189, 34), #oliva
            rgb_to_hex(23, 190, 207)] #cyan

linestyles = OrderedDict(
    [('solid',               (0, ())),
     ('loosely dotted',      (0, (1, 10))),
     ('dotted',              (0, (1, 5))),
     ('densely dotted',      (0, (1, 1))),
     ('loosely dashed',      (0, (5, 10))),
     ('dashed',              (0, (5, 5))),
     ('densely dashed',      (0, (5, 1))),
     ('loosely dashdotted',  (0, (3, 10, 1, 10))),
     ('dashdotted',          (0, (3, 5, 1, 5))),
     ('densely dashdotted',  (0, (3, 1, 1, 1))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])

patternstyles = [ "/" , "\\" , "|" , "-" , "+" , "x", "o", "O", ".", "*" ]
# patternstyles = [ "/" , "\\" , "|" , "-" , "x", "."]

lwOther=3.5
lwGABR=4.0
markersize=10
lineStyles = {
    'BB':     {'ls': ':',  'lw':lwOther, 'marker': '^'},
    'Bolae':    {'ls': linestyles['densely dotted'],  'lw':lwOther, 'marker': 'o'},
    'DD':       {'ls': '-.', 'lw':lwOther, 'marker': '^'},
    'Dynamic':  {'ls': '-.', 'lw':lwOther, 'marker': 'o'},
    'Throughput':       {'ls': linestyles['densely dashdotted'], 'lw':lwOther, 'marker': 'd'},
    'Pensieve':      {'ls': '--', 'lw':lwOther, 'marker': 'None'},
    'GABR-E':   {'ls': linestyles['densely dashdotdotted'],  'lw':lwGABR,  'marker': '^'}
    }


font = {
        'family': 'Arial',
        'color':  'black',
        # 'weight': 'bold',
        'size': 24,
        }

graph_font_dict={
    'family': 'Arial',
    'color':'black',
    # 'weight':'bold',
    'size':22
}

legend_font_dict={
    'family':'Arial',
    # 'color':'black',
    # 'weight':'bold',
    'size': 18
}

ANNOTATE_SIZE=18
ARROW_THICKNESS=1.0
ARROW_COLOR='black'
ANNOTATE_COLOR='black'
TICKS_FONT_SIZE=20
LEGEND_FONT_SIZE=12
NUM_BINS = 100

figSizeW=10
figSizeH=7

# PLOT_FOLDER='./plots/' # rebuffer penalty = 12.0
PLOT_FOLDER='./plots/' # rebuffer penalty = 4.3

videos=['tos','bbb','doc']
# SCHEMES=['GABR-E','GABR-Q','Pensieve','BB','Bolae','DD','Dynamic','Throughput']
SCHEMES=['GreenABR','Pensieve','BB','Bolae','DD','Dynamic','Throughput']
SCHEMEFULLNAMES = ['GreenABR','Pensieve','Bola','Bolae', 'DynamicDash','Dynamic ABR' ,'Throughput-based']
# SUMMARY_DIR='./summaryResults/'  # rebuffer penalty 12.0
SUMMARY_DIR='./summaryResults/' # rebuffer penalty 4.3
VIDEO_LEN=44.0

def getRewards(scheme, video):
    f_path=SUMMARY_DIR+video+'_'+scheme+'_summaryResults.csv'
    data=pd.read_csv(f_path)
    return np.reshape(np.array(data['QoE'])/VIDEO_LEN,-1)

# data=getRewards('GABR-E','tos')

def arrowed_spines(fig, ax):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    # removing the default axis on all sides:
    for side in ['bottom','right','top','left']:
        ax.spines[side].set_visible(False)
    # removing the axis ticks
    # plt.xticks([]) # labels
    # plt.yticks([])
    # ax.xaxis.set_ticks_position('none') # tick markers
    # ax.yaxis.set_ticks_position('none')
    # get width and height of axes object to compute
    # matching arrowhead length and width
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height
    # manual arrowhead width and length
    hw = 1./50.*(ymax-ymin)
    hl = 1./50.*(xmax-xmin)
    lw = 1. # axis line width
    ohg = 0.3 # arrow overhang
    # compute matching arrowhead length and width
    yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width
    yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height
    # draw x and y axis
    ax.arrow(xmin, 0, xmax-xmin, 0., fc='k', ec='k', lw = lw, head_width=hw, head_length=hl, overhang = ohg, length_includes_head= True, clip_on = False)
    ax.arrow(xmin, ymin, 0., ymax-ymin, fc='k', ec='k', lw = lw, head_width=yhw, head_length=yhl, overhang = ohg, length_includes_head= True, clip_on = False)


def getAllResults(video, scheme):
    f_path=SUMMARY_DIR+video+'_'+scheme+'_summaryResults.csv'
    data=pd.read_csv(f_path)
    return np.reshape(np.array(data['Efficiency_Avg']),-1),np.reshape(np.array(data['Energy_Avg']),-1),np.reshape(np.array(data['Efficiency_X']),-1),np.reshape(np.array(data['Efficiency_G']),-1),np.reshape(np.array(data['QoE']),-1), np.reshape(np.array(data['Rebuffer_Time']),-1),\
            np.reshape(np.array(data['Smoothness_Change']),-1), np.reshape(np.array(data['Data']),-1),\
            np.reshape(np.array(data['Energy_G']),-1),np.reshape(np.array(data['Energy_X']),-1)

def draw_all_normalized():
    eff_a={}
    eff_x={}
    eff_g={}
    qoe={}
    rebuf={}
    smooth={}
    data={}
    en_a={}
    energy_g={}
    energy_x={}
    for s in SCHEMES:
        qoe[s]=[]
        rebuf[s]=[]
        smooth[s]=[]
        data[s]=[]
        energy_g[s]=[]
        energy_x[s]=[]
        eff_x[s]=[]
        eff_g[s]=[]
        eff_a[s]=[]
        en_a[s]=[]

    for v in videos:
        en_a_n=0
        eff_a_n=0
        eff_x_n=0
        eff_g_n=0
        qoe_n=0
        rebuf_n=0
        smooth_n=0
        data_n=0
        energy_g_n=0
        energy_x_n=0
        for s in SCHEMES:
            eff_a[s],en_a[s],eff_x[s],eff_g[s],qoe[s],rebuf[s],smooth[s],data[s], energy_g[s], energy_x[s]=getAllResults(v,s)

            eff_a_n=max(eff_a_n, np.mean(eff_a[s]))
            en_a_n=max(en_a_n, np.mean(en_a[s]))
            eff_x_n=max(eff_x_n, np.mean(eff_x[s]))
            eff_g_n=max(eff_g_n, np.mean(eff_g[s]))
            qoe_n=max(qoe_n, np.mean(qoe[s]))
            rebuf_n=max(rebuf_n, np.mean(rebuf[s]))
            smooth_n=max(smooth_n, np.mean(smooth[s]))
            data_n=max(data_n, np.mean(data[s]))
            energy_g_n=max(energy_g_n, np.mean(energy_g[s]))
            energy_x_n=max(energy_x_n, np.mean(energy_x[s]))
            

        schemedata = {}
        for scheme in SCHEMES:
            totalEff_a=np.round(np.mean(eff_a[scheme])/eff_a_n, decimals=2)
            totalEn_a=np.round(np.mean(en_a[scheme])/en_a_n, decimals=2)
            totalEff_x = np.round(np.mean(eff_x[scheme])/eff_x_n, decimals=2)
            totalEff_g = np.round(np.mean(eff_g[scheme])/eff_g_n, decimals=2)
            totalQoE = np.round(np.mean(qoe[scheme])/qoe_n, decimals=2)
            rebufferTime = np.round(np.mean(rebuf[scheme])/rebuf_n, decimals=2)
            smoothChange = np.round(np.mean(smooth[scheme])/smooth_n, decimals=2)
            total_energy_g = np.round(np.mean(energy_g[scheme])/energy_g_n, decimals=2)
            total_energy_x = np.round(np.mean(energy_x[scheme])/energy_x_n, decimals=2)
            dataUsage=np.round(np.mean(data[scheme])/data_n, decimals=2)
            graph_data = [totalEff_a, totalEn_a, totalEff_g,totalEff_x,totalQoE, rebufferTime, smoothChange, total_energy_g,total_energy_x,dataUsage]
            graph_data_a = [totalEff_a,totalQoE, rebufferTime, smoothChange, totalEn_a,dataUsage]
            schemedata.setdefault(scheme, graph_data_a)

        groupnum = len(schemedata[SCHEMES[0]])

        fig, ax = plt.subplots(figsize=(18, 4))
        bar_width = 0.07

        # Set position of bar on X axis
        br = np.arange(groupnum)
        for i, scheme in enumerate(SCHEMES):
            if i == 0:
                plt.bar(br, schemedata[scheme], bar_width, color=colors[i], edgecolor='w', lw=1.)
            else:
                plt.bar(br, schemedata[scheme], bar_width, color=colors[i], edgecolor='w', hatch=patternstyles[i]*3, lw=1.)
            brnew = [x + bar_width for x in br]
            br = [x for x in brnew]
    #     plt.legend(SCHEMEFULLNAMES, loc=2, prop=legend_font_dict, frameon=False)
    #     draw bar outlines
        br = np.arange(groupnum)
        for i, scheme in enumerate(SCHEMES):
            plt.bar(br, schemedata[scheme], bar_width, color='none', edgecolor='k', lw=2., zorder=1)
            brnew = [x + bar_width for x in br]
            br = [x for x in brnew]
        ax.legend(SCHEMEFULLNAMES, bbox_to_anchor=(0,1.02,1,1),loc="lower left",ncol=len(SCHEMES)//2+1,mode="expand", prop=legend_font_dict, facecolor='white', framealpha=1)
        # Adding Xticks
        plt.ylabel('Normalized average', fontdict=graph_font_dict)
        plt.yticks(fontsize=TICKS_FONT_SIZE)
        plt.xticks(fontsize=TICKS_FONT_SIZE)
        plt.yticks((0, 0.2, 0.4, 0.6, 0.8, 1))
        xlabels = ['Energy\nEfficiency','QoE', 'Rebuffer\nTime', 'Smoothness\nChange', 'Energy\nConsumption', 'Data\nUsage']
        # xlabels_new = [re.sub("(.{10})", "\\1\n", label, 0, re.DOTALL) for label in xlabels]
        plt.xticks([r + (len(SCHEMES) / 2 - 0.5) * bar_width for r in range(6)], xlabels)
        ax = plt.gca()
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(False)
        # ax.minorticks_on()
        ax.tick_params(which='major', length=10, width=2, direction='in')
        ax.tick_params(axis='both', which='major', pad=15)
        plt.tight_layout()
        plt.savefig(PLOT_FOLDER+v+'_'+'normalized_bar_results.png', dpi=300)
        # plt.show()


draw_all_normalized()

def getAllResultsAvg(scheme):
    f_path=SUMMARY_DIR+scheme+'_avg.csv'
    data=pd.read_csv(f_path)
    return np.reshape(np.array(data['Efficiency_Avg']),-1),np.reshape(np.array(data['Energy_Avg']),-1),np.reshape(np.array(data['Efficiency_X']),-1),np.reshape(np.array(data['Efficiency_G']),-1),np.reshape(np.array(data['QoE']),-1), np.reshape(np.array(data['Rebuffer_Time']),-1),\
            np.reshape(np.array(data['Smoothness_Change']),-1), np.reshape(np.array(data['Data']),-1),\
            np.reshape(np.array(data['Energy_G']),-1),np.reshape(np.array(data['Energy_X']),-1)

def draw_all_normalized_avg():
    
    normalized_all=[]
    eff_a={}
    en_a={}
    eff_x={}
    eff_g={}
    qoe={}
    rebuf={}
    smooth={}
    data={}
    energy_g={}
    energy_x={}
    for s in SCHEMES:
        eff_a[s]=[]
        en_a[s]=[]
        eff_x[s]=[]
        eff_g[s]=[]
        qoe[s]=[]
        rebuf[s]=[]
        smooth[s]=[]
        data[s]=[]
        energy_g[s]=[]
        energy_x[s]=[]

    eff_a_n=0
    en_a_n=0
    eff_x_n=0
    eff_g_n=0
    qoe_n=0
    rebuf_n=0
    smooth_n=0
    data_n=0
    energy_g_n=0
    energy_x_n=0
    for s in SCHEMES:
        eff_a[s],en_a[s],eff_x[s],eff_g[s],qoe[s],rebuf[s],smooth[s],data[s], energy_g[s], energy_x[s]=getAllResultsAvg(s)
        # eff[s]=np.divide(qoe[s],energy_x[s])
        eff_a_n=max(eff_a_n, np.mean(eff_a[s]))
        en_a_n=max(en_a_n, np.mean(en_a[s]))
        eff_x_n=max(eff_x_n, np.mean(eff_x[s]))
        eff_g_n=max(eff_g_n, np.mean(eff_g[s]))
        qoe_n=max(qoe_n, np.mean(qoe[s]))
        rebuf_n=max(rebuf_n, np.mean(rebuf[s]))
        smooth_n=max(smooth_n, np.mean(smooth[s]))
        data_n=max(data_n, np.mean(data[s]))
        energy_g_n=max(energy_g_n, np.mean(energy_g[s]))
        energy_x_n=max(energy_x_n, np.mean(energy_x[s]))
        

    schemedata = {}
    for scheme in SCHEMES:
        totalEff_a= np.round(np.mean(eff_a[scheme])/eff_a_n, decimals=2)
        totalEn_a= np.round(np.mean(en_a[scheme])/en_a_n, decimals=2)
        totalEff_x= np.round(np.mean(eff_x[scheme])/eff_x_n, decimals=2)
        totalEff_g= np.round(np.mean(eff_g[scheme])/eff_g_n, decimals=2)
        totalQoE = np.round(np.mean(qoe[scheme])/qoe_n, decimals=2)
        rebufferTime = np.round(np.mean(rebuf[scheme])/rebuf_n, decimals=2)
        smoothChange = np.round(np.mean(smooth[scheme])/smooth_n, decimals=2)
        total_energy_g = np.round(np.mean(energy_g[scheme])/energy_g_n, decimals=2)
        total_energy_x = np.round(np.mean(energy_x[scheme])/energy_x_n, decimals=2)
        dataUsage=np.round(np.mean(data[scheme])/data_n, decimals=2)
        graph_data = [totalEff_a,totalQoE, rebufferTime, smoothChange, totalEn_a,dataUsage]
        normalized_all.append([scheme,totalEff_a, totalEn_a,totalEff_x,totalEff_g,totalQoE, rebufferTime, smoothChange, total_energy_g,total_energy_x,dataUsage])

        schemedata.setdefault(scheme, graph_data)

    groupnum = len(schemedata[SCHEMES[0]])
    normalized_all_data=pd.DataFrame(normalized_all,columns=['Scheme','Avg Efficiency','Avg Energy','Efficiency_X','Efficiency_G','QoE','Rebuffer','Smooth','Energy_G','Energy_X','Data'])
    normalized_all_data.to_csv('./summaryResults/all_normalized.csv')

    fig, ax = plt.subplots(figsize=(18, 4))
    bar_width = 0.08

    # Set position of bar on X axis
    br = np.arange(groupnum)
    for i, scheme in enumerate(SCHEMES):
        if i == 0:
            plt.bar(br, schemedata[scheme], bar_width, color=colors[i], edgecolor='w', lw=1.)
        else:
            plt.bar(br, schemedata[scheme], bar_width, color=colors[i], edgecolor='w', hatch=patternstyles[i]*3, lw=1.)
        brnew = [x + bar_width for x in br]
        br = [x for x in brnew]
#     plt.legend(SCHEMEFULLNAMES, loc=2, prop=legend_font_dict, frameon=False)
#     draw bar outlines
    br = np.arange(groupnum)
    for i, scheme in enumerate(SCHEMES):
        plt.bar(br, schemedata[scheme], bar_width, color='none', edgecolor='k', lw=2., zorder=1)
        brnew = [x + bar_width for x in br]
        br = [x for x in brnew]
    ax.legend(SCHEMEFULLNAMES, bbox_to_anchor=(0,1.02,1,1),loc="lower left",ncol=len(SCHEMES)//2+1,mode="expand", prop=legend_font_dict, facecolor='white', framealpha=1)
    # Adding Xticks
    plt.ylabel('Normalized average', fontdict=graph_font_dict)
    plt.yticks(fontsize=TICKS_FONT_SIZE)
    plt.xticks(fontsize=TICKS_FONT_SIZE)
    # plt.yticks((0, 1))
    plt.yticks((0, 0.2, 0.4, 0.6, 0.8, 1))
    xlabels = ['Energy\nEfficiency','QoE', 'Rebuffer\nTime', 'Smoothness\nChange', 'Energy\nConsumption','Data\nUsage']
    # xlabels_new = [re.sub("(.{10})", "\\1\n", label, 0, re.DOTALL) for label in xlabels]
    plt.xticks([r + (len(SCHEMES)/2-0.5)*bar_width for r in range(6)], xlabels)
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.minorticks_on()
    ax.tick_params(which='major', length=10, width=2, direction='in')
    ax.tick_params(axis='both', which='major', pad=15)
    plt.tight_layout()
    plt.savefig(PLOT_FOLDER+'avg'+'_'+'normalized_bar_results.png', dpi=300)
    plt.show()


draw_all_normalized_avg()