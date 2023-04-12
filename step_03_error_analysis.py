# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 17:24:07 2023

@author: cta4r

This is the program to visualize 

"""



import numpy as np
import pandas as pd
# import streamlit as st

from datetime import datetime
from matplotlib import pyplot as plt

from pykeri.scidata.matprop import MatProp
from pykeri.thermoelectrics.TEProp import TEProp
from pykeri.thermoelectrics.TEProp_xls import TEProp as TEProp_xls
from pykeri.thermoelectrics.solver1d.leg import Leg
from pykeri.thermoelectrics.solver1d.environment import Environment
from pykeri.thermoelectrics.solver1d.device import Device

from pykeri.byungkiryu import byungkiryu_util as br
from library.tematdb_util import draw_mat_teps, tep_generator_from_excel_files
from library.tematdb_util import err_norm_calc, err_norm_calc2
from library.tematdb_util import get_Ts_TEPZT, get_ZTs_interp


formattedDate, yyyymmdd, HHMMSS = br.now_string()



def get_error_dict(Ts_TEPZT, ZT_raw_on_TEPZT, ZT_TEP_on_TEPZT):        
    dZT_interp = ZT_raw_on_TEPZT - ZT_TEP_on_TEPZT
    err = dZT_interp    
    doi, doilink = get_doi(sampleid)    
    sampleid_errLnorm_dict = {}
    sampleid_errLnorm_dict['sampleid'] = sampleid
    sampleid_errLnorm_dict['doi']  = doi
    sampleid_errLnorm_dict['doilink']  = doilink
    sampleid_errLnorm_dict['L1']   = err_norm_calc2(err,n_norm=1)
    sampleid_errLnorm_dict['L2']   = err_norm_calc2(err,n_norm=2)
    sampleid_errLnorm_dict['L3']   = err_norm_calc2(err,n_norm=3)
    sampleid_errLnorm_dict['L4']   = err_norm_calc2(err,n_norm=4)
    sampleid_errLnorm_dict['Linf'] = np.max( np.abs(err) )
    sampleid_errLnorm_dict['errMax'] = np.max(err)
    sampleid_errLnorm_dict['errMin'] = np.min(err)
    return sampleid_errLnorm_dict

def error_analysis(sampleid):
    interp_opt = {MatProp.OPT_INTERP:MatProp.INTERP_LINEAR,\
              MatProp.OPT_EXTEND_LEFT_TO:1,          # ok to 0 Kelvin
              MatProp.OPT_EXTEND_RIGHT_BY:2000}        # ok to +50 Kelvin from the raw data
    TF_mat_complete, mat = tep_generator_from_excel_files(sampleid, interp_opt)
    
    # for sampleid in sampleid_list:
    sampleid_errLnorm_dict = dict()
    if not TF_mat_complete:
        sampleid_errLnorm_dict['sampleid'] = sampleid
        sampleid_errLnorm_dict['TF_mat_complete'] = False
    else:
        Ts_TEP, Ts_ZT, Ts_TEPZT          = get_Ts_TEPZT(mat)
        ZT_raw_on_Ts_TEPZT, ZT_TEP_on_Ts_TEPZT = get_ZTs_interp(mat, Ts_TEPZT)
        ZT_raw_on_Ts_TEP,   ZT_TEP_on_Ts_TEP   = get_ZTs_interp(mat, Ts_TEP)
        ZT_raw_on_Ts_ZT,    ZT_TEP_on_Ts_ZT    = get_ZTs_interp(mat, Ts_ZT)
            
        sampleid_errLnorm_dict = get_error_dict(Ts_TEPZT, ZT_raw_on_Ts_TEPZT, ZT_TEP_on_Ts_TEPZT)
        sampleid_errLnorm_dict['TF_mat_complete'] = True

        avgZT_raw_on_Ts_ZT  = np.mean(ZT_raw_on_Ts_ZT)
        avgZT_TEP_on_Ts_TEP = np.mean(ZT_TEP_on_Ts_TEP)
        
        peakZT_raw_on_Ts_ZT  = np.max(ZT_raw_on_Ts_ZT)
        peakZT_TEP_on_Ts_TEP = np.max(ZT_TEP_on_Ts_TEP)
        
        sampleid_errLnorm_dict['avgZT_raw_on_Ts_ZT'] = avgZT_raw_on_Ts_ZT
        sampleid_errLnorm_dict['avgZT_TEP_on_Ts_TEP'] = avgZT_TEP_on_Ts_TEP
        sampleid_errLnorm_dict['peakZT_raw_on_Ts_ZT'] = peakZT_raw_on_Ts_ZT
        sampleid_errLnorm_dict['peakZT_TEP_on_Ts_TEP'] = peakZT_TEP_on_Ts_TEP
        
        sampleid_errLnorm_dict['dpeakZT'] = peakZT_raw_on_Ts_ZT - peakZT_TEP_on_Ts_TEP
        sampleid_errLnorm_dict['davgZT']  = avgZT_raw_on_Ts_ZT  - avgZT_TEP_on_Ts_TEP
        
        
    return sampleid_errLnorm_dict    

def get_doi(sampleid):
    df_db_meta_sampleid = df_db_meta[ df_db_meta.sampleid == sampleid]
    doi = df_db_meta_sampleid.DOI.iloc[0]   
    doilink = "https://doi.org/{}".format(doi)
    return doi, doilink



## choose DB
## Read mat tep excel
interp_opt = {MatProp.OPT_INTERP:MatProp.INTERP_LINEAR,\
          MatProp.OPT_EXTEND_LEFT_TO:1,          # ok to 0 Kelvin
          MatProp.OPT_EXTEND_RIGHT_BY:2000}        # ok to +50 Kelvin from the raw data

## Read mat meta excel
db_mode = 'teMatDb'
file_db_meta = "_tematdb_metadata_v1.1.0-20230412_brjcsjp.xlsx"
df_db_meta = pd.read_excel("./"+file_db_meta, sheet_name='list', )
    
    
# sampleid_errLnorm_dict = error_analysis(sampleid=2)
# print(sampleid_errLnorm_dict)
   

sampleid_list = list(df_db_meta.sampleid.unique())
# sampleid_list = sampleid_list[0:5]
sampleid_errLnorm_dict_list = []
for sampleid in sampleid_list:
    sampleid_errLnorm_dict = error_analysis(sampleid)
    sampleid_errLnorm_dict_list.append(sampleid_errLnorm_dict)
    
    if (sampleid_errLnorm_dict['TF_mat_complete']):
        peakZT_raw_on_Ts_ZT  = sampleid_errLnorm_dict['peakZT_raw_on_Ts_ZT']
        peakZT_TEP_on_Ts_TEP = sampleid_errLnorm_dict['peakZT_TEP_on_Ts_TEP']
        dpeakZT = peakZT_raw_on_Ts_ZT-peakZT_TEP_on_Ts_TEP
        print("{:03d} {:5.1f} {:5.1f} {:5.1f}".format(sampleid, peakZT_raw_on_Ts_ZT, peakZT_TEP_on_Ts_TEP, dpeakZT ))
    del sampleid_errLnorm_dict

df_err = pd.DataFrame(sampleid_errLnorm_dict_list)
df_err.to_csv("./data_error_analysis/"+"error_{}.csv".format(formattedDate), index=False)
df_err.to_csv("./data_error_analysis/"+"error.csv", index=False)
    


    # avgZT_raw





