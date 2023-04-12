# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 17:24:07 2023

@author: cta4r

This is the program to visualize 

"""


import numpy as np
import pandas as pd
import streamlit as st

from datetime import datetime

from pykeri.scidata.matprop import MatProp
# from pykeri.thermoelectrics.TEProp import TEProp
# from pykeri.thermoelectrics.TEProp_xls import TEProp as TEProp_xls
from pykeri.thermoelectrics.TEProp_df import TEProp as TEProp_df
from pykeri.thermoelectrics.solver1d.leg import Leg
from pykeri.thermoelectrics.solver1d.environment import Environment
from pykeri.thermoelectrics.solver1d.device import Device

from pykeri.byungkiryu import byungkiryu_util as br

from library.tematdb_util import get_Ts_TEPZT, get_ZTs_interp
from library.tematdb_util import draw_mat_teps, tep_generator_from_excel_files
from library.dev_performance import set_singleleg_device, run_pykeri, draw_dev_perf

from matplotlib import pyplot as plt

formattedDate, yyyymmdd, HHMMSS = br.now_string()




# if (1):

def draw_mat_ZT_errors(mat, label_db="", label_sampleid="", label_doi=""):
    suptitle = "{} {} {}".format(label_db, label_sampleid, label_doi)
    
    figsize = (8,8)
    
    fig1, axs1  = plt.subplots(2,2, figsize=figsize)
    (ax1, ax2), (ax3, ax4) = axs1
    fig1.suptitle(suptitle)

    # (ax5, ax6), (ax7, ax8) = axs2
    # fig2, axs2  = plt.subplots(2,2, figsize=figsize, sharex=True)
    # fig2.suptitle(suptitle)

    alpha_raw   = np.array( mat.Seebeck.raw_data() ).T
    rho_raw     = np.array( mat.elec_resi.raw_data() ).T    
    kappa_raw   = np.array( mat.thrm_cond.raw_data() ).T
    ZT_raw   = np.array( mat.ZT.raw_data() ).T

    sigma_raw   = rho_raw.copy()
    sigma_raw[1] = (1/rho_raw[1]).copy()    

    autoTc = mat.min_raw_T
    autoTh = mat.max_raw_T
    
    Ts_TEP, Ts_ZT, Ts_TEPZT = get_Ts_TEPZT(mat)
    TcZT, ThZT = min(Ts_ZT), max(Ts_ZT)
    TcTEP, ThTEP = min(Ts_TEP), max(Ts_TEP)
    TcTEPZT, ThTEPZT = min(Ts_TEPZT), max(Ts_TEPZT)
    
    
    ZT_raw_on_Ts_TEPZT, ZT_TEP_on_Ts_TEPZT = get_ZTs_interp(mat, Ts_TEPZT)
    ZT_raw_on_Ts_TEP,   ZT_TEP_on_Ts_TEP   = get_ZTs_interp(mat, Ts_TEP)
    ZT_raw_on_Ts_ZT,    ZT_TEP_on_Ts_ZT    = get_ZTs_interp(mat, Ts_ZT)
    
    ZT_error_on_Ts_TEPZT = ZT_raw_on_Ts_TEPZT - ZT_TEP_on_Ts_TEPZT
    
    peakZT_raw_on_Ts_ZT  = np.max(ZT_raw_on_Ts_ZT)
    peakZT_TEP_on_Ts_TEP = np.max(ZT_TEP_on_Ts_TEP)
    ZT_dev_max = np.max( np.abs(ZT_error_on_Ts_TEPZT) )*1.2
        
    import math
    T_plot_min = min(autoTc,min(ZT_raw[0]))
    T_plot_max = min(autoTh,max(ZT_raw[0]))
    T_plot_min = math.floor(T_plot_min/100)*100-50
    T_plot_max = math.ceil(T_plot_max/100)*100+50
    
    ax = ax1
    tep_title = r'Dimensionless Figure of Meirt: $ZT$'
    ax.plot( Ts_TEP, ZT_TEP_on_Ts_TEP, label="Calculated $ZT$ from TEP",color='C0')
    ax.plot( Ts_ZT, ZT_raw_on_Ts_ZT, color='C1')
    ax.scatter( ZT_raw[0], ZT_raw[1], label="Raw $ZT$ by author",color='C1')       
    ax.set_title(tep_title)
    ax.set_xlim(T_plot_min,T_plot_max)
    ax.set_ylim(-0.3, max( max(ZT_raw_on_Ts_ZT), max(ZT_raw_on_Ts_TEP))+.3)
    ax.legend()    
    ax.set_ylabel(r"$ZT$ [1]")
    ax.set_xlabel("Temperature [K]")
    
     
    ax = ax2
    tep_title = r'$ZT$-$ZT$ plot'
    ax.plot( [-0.1,max(ZT_raw_on_Ts_TEPZT)+0.1], [-0.1,max(ZT_raw_on_Ts_TEPZT)+0.1], 
            linestyle='--',color='C0',)
    ax.scatter( ZT_TEP_on_Ts_TEPZT, ZT_raw_on_Ts_TEPZT,color='C1')
    ax.set_xlabel('Calculated ZT from TEP')
    ax.set_ylabel('Raw ZT by author')
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")    
    ax.set_title(tep_title)


    ax = ax3
    tep_title = r'Deviation in $ZT$ ($\delta$-ZT)'
    ax.plot( Ts_TEPZT, ZT_raw_on_Ts_TEPZT - ZT_TEP_on_Ts_TEPZT , label=r"$\delta$-$ZT$",color='C1')
    ax.plot( Ts_TEPZT, Ts_TEPZT*0 , linestyle='--',color='C0',)
    ax.set_ylim(-ZT_dev_max,+ZT_dev_max)    
    ax.set_title(tep_title)
    ax.set_xlim(T_plot_min,T_plot_max)
    ax.legend()    
    ax.set_ylabel(r"$\delta$-$ZT$ [1]")
    ax.set_xlabel("Temperature [K]")
    
    
    import scipy.stats as stats
    ax = ax4
    # ax = ax2
    tep_title = r'Q-Q plot of $\delta$-ZT'
    x = np.array(ZT_error_on_Ts_TEPZT)
    res4 = stats.probplot(x,dist=stats.norm, plot=ax, rvalue=True)
    ax.set_title(r"Q-Q plot of Peak-ZT deviation")
    ax.set_ylabel(r"$\delta$-$ZT$")
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")    
    
    fig1.tight_layout()
    
    return fig1


if __name__ == "__main__":
    db_mode = 'teMatDb'
    sampleid = 1
    
    file_tematdb_metadata_csv = "_tematdb_metadata_v1.0.0-20230407_brjcsjp.xlsx"
    file_tematdb_db_csv  =  "tematdb_v1.0.0_completeTEPset.csv"
    df_db_meta = pd.read_excel("../"+file_tematdb_metadata_csv, sheet_name='list', )
    df_db_csv = pd.read_csv("../data_csv/"+file_tematdb_db_csv)
    
    df_db_meta_sampleid = df_db_meta[ df_db_meta.sampleid == sampleid]
    doi = df_db_meta_sampleid.DOI.iloc[0]
    label_db = "DB: {}".format(db_mode)
    label_sampleid = "sampleid: {}".format(sampleid)
    label_doi = '[DOI: {}]'.format(doi)    
    
    
    interp_opt = {MatProp.OPT_INTERP:MatProp.INTERP_LINEAR,\
                  MatProp.OPT_EXTEND_LEFT_TO:1,          # ok to 0 Kelvin
                  MatProp.OPT_EXTEND_RIGHT_BY:2000}        # ok to +50 Kelvin from the raw data
    df_db_csv_sampleid = df_db_csv[ df_db_csv.sampleid == sampleid]
    df_alpha = df_db_csv_sampleid[ df_db_csv_sampleid.tepname == 'alpha']
    df_rho   = df_db_csv_sampleid[ df_db_csv_sampleid.tepname == 'rho'  ]
    df_kappa = df_db_csv_sampleid[ df_db_csv_sampleid.tepname == 'kappa']
    df_ZT    = df_db_csv_sampleid[ df_db_csv_sampleid.tepname == 'ZT'   ]
    try:
        mat = TEProp_df.load_from_df(df_alpha, df_rho, df_kappa, df_ZT, mat_name='test')
        mat.set_interp_opt(interp_opt)
        TF_mat_complete = True
    except:
        TF_mat_complete = False

    fig = draw_mat_ZT_errors(mat, label_db=label_db, label_sampleid=label_sampleid, label_doi=label_doi)
    