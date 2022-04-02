import os
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


PLOT_FOLDER='./plots/' # rebuffer penalty = 4.3

videos=['tos','bbb','doc']
SCHEMES=['GreenABR','Pensieve','BB','Bolae','DD','Dynamic','Throughput']
SCHEMEFULLNAMES = ['GreenABR','Pensieve','Bola','Bolae', 'DynamicDash','Dynamic ABR' ,'Throughput-based']
SUMMARY_DIR='./summaryResults/'
VIDEO_LEN=44.0

def getStutsAvg(scheme):
    f_path = SUMMARY_DIR+scheme+'_avg.csv'
    data = pd.read_csv(f_path)
    return np.reshape(np.array(data['Stuttering']), -1)

def draw_stuts_avg():
    stut = {}
    for s in SCHEMES:
        stut[s] = []
    for s in SCHEMES:
        stut[s] = getStutsAvg(s)

    schemedata = {}
    stddata = {}
    for scheme in SCHEMES:
        totalStut = np.round(np.mean(stut[scheme]), decimals=2)
        stds = np.round(np.std(stut[scheme]), decimals=2)
        graph_data = [totalStut]
        schemedata.setdefault(scheme, graph_data)
        stddata.setdefault(scheme, stds)

    groupnum = len(schemedata[SCHEMES[0]])

    fig, ax = plt.subplots(figsize=(10, 4))
    bar_width = 0.07

    # Set position of bar on X axis
    br = np.arange(groupnum)
    for i, scheme in enumerate(SCHEMES):
        if i == 0:
            plt.bar(br, schemedata[scheme], bar_width/2, color=colors[i], edgecolor='w', lw=1.)
            # plt.bar(br, schemedata[scheme], bar_width, yerr=stddata[scheme], color=colors[i], edgecolor='w', lw=1.)
        else:
            plt.bar(br, schemedata[scheme], bar_width/2,  color=colors[i], edgecolor='w', hatch=patternstyles[i] * 3, lw=1.)
        brnew = [x + bar_width for x in br]
        br = [x for x in brnew]
    #     plt.legend(SCHEMEFULLNAMES, loc=2, prop=legend_font_dict, frameon=False)
    ax.legend(SCHEMEFULLNAMES, bbox_to_anchor=(0, 1.02, 1, 1), loc="lower left", ncol=3, mode="expand",
              prop=legend_font_dict, facecolor='white', framealpha=1)
    #     draw bar outlines
    br = np.arange(groupnum)
    for i, scheme in enumerate(SCHEMES):
        plt.bar(br, schemedata[scheme], bar_width/2, color='none', edgecolor='k', lw=2., zorder=1)
        brnew = [x + bar_width for x in br]
        br = [x for x in brnew]
    #     ax.legend(SCHEMEFULLNAMES,loc="lower left", prop=legend_font_dict, facecolor='white', framealpha=1)

    # Adding Xticks
    plt.xlabel('ABR approaches', fontdict=graph_font_dict, labelpad=10)
    plt.ylabel('Capacity Violations', fontdict=graph_font_dict)
    plt.yticks(fontsize=TICKS_FONT_SIZE)
   
    ax = plt.gca()
    #     ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dotted', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.tick_params(which='major', length=10, width=2, direction='in')
    ax.tick_params(axis='both', which='major', pad=15)
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    # ax.axes.xaxis.set_visible(False)
    plt.tight_layout()
    plt.savefig(PLOT_FOLDER + 'capacity_violations.png', dpi=300)
    plt.show()

draw_stuts_avg()
