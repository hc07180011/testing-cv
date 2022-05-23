import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

def graphKmeans(dataSet, labels, unique_labels):
    # f = go.FigureWidget()
    # f.layout.hovermode = 'closest'
    # f.layout.hoverdistance = -1
 
    #filter rows of original data
    # filtered_label0 = dataSet[labels == 0]
    
    #plotting the results
    for i in unique_labels:
        plt.scatter(dataSet[labels == i , 0] , dataSet[labels == i , 1] , label = i)
    plt.legend()
    plt.savefig('myresult.svg')
    plt.savefig('myresult.png')
    plt.savefig('myresult.pdf')
    return