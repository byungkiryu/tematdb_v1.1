# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 15:32:28 2022

@author: byungkiryu

This code generates a set of extended TEPs from raw digitized TEPs
Here, I used the teMatDb v1.0.


"""

import os
import numpy as np
import pandas as pd
from datetime import datetime

from pykeri.scidata.matprop import MatProp
from pykeri.thermoelectrics.TEProp import TEProp
from pykeri.thermoelectrics.TEProp_xls import TEProp as TEProp_xls
from pykeri.thermoelectrics.solver1d.leg import Leg
from pykeri.thermoelectrics.solver1d.environment import Environment
from pykeri.thermoelectrics.solver1d.device import Device

from pykeri.byungkiryu import byungkiryu_util as br

formattedDate, yyyymmdd, HHMMSS = br.now_string()


## read single TEP csv file to get dataframe for TEPs, alpha, rho, kappa, ZT
def get_df_from_tematdb(teMatDb_file):
    df_raw_tep = pd.read_csv(teMatDb_file)    
    df_raw_alpha = df_raw_tep[ df_raw_tep.tepname == 'alpha']
    df_raw_rho   = df_raw_tep[ df_raw_tep.tepname == 'rho']
    df_raw_kappa = df_raw_tep[ df_raw_tep.tepname == 'kappa']
    df_raw_ZT    = df_raw_tep[ df_raw_tep.tepname == 'ZT']    
    return df_raw_tep, df_raw_alpha, df_raw_rho, df_raw_kappa, df_raw_ZT

def tep_generator_from_csv_files(sampleid, df_alpha, df_rho, df_kappa, df_ZT, interp_opt):
    temp_grid = df_alpha[df_alpha.sampleid == sampleid].Temperature
    tep_grid = df_alpha[df_alpha.sampleid == sampleid].tepvalue    
    Seebeck_raw_data = tuple((a_elem,b_elem) for (a_elem,b_elem) in zip(temp_grid,tep_grid))
    
    temp_grid = df_rho[df_rho.sampleid == sampleid].Temperature
    tep_grid = df_rho[df_rho.sampleid == sampleid].tepvalue
    elec_resi_raw_data = tuple((a_elem,b_elem) for (a_elem,b_elem) in zip(temp_grid,tep_grid))

    temp_grid = df_kappa[df_kappa.sampleid == sampleid].Temperature
    tep_grid = df_kappa[df_kappa.sampleid == sampleid].tepvalue
    thrm_cond_raw_data = tuple((a_elem,b_elem) for (a_elem,b_elem) in zip(temp_grid,tep_grid))  
   
    temp_grid = df_ZT[df_ZT.sampleid == sampleid].Temperature
    tep_grid = df_ZT[df_ZT.sampleid == sampleid].tepvalue
    # thrm_cond_raw_data = tuple((a_elem,b_elem) for (a_elem,b_elem) in zip(temp_grid,tep_grid))   
   
    mat = TEProp.from_raw_data(elec_resi_raw_data, Seebeck_raw_data, thrm_cond_raw_data, name="tematdb_{:5d}".format(sampleid))    
    mat.set_interp_opt(interp_opt)
    return mat 


def tep_generator_from_excel_files(sampleid, interp_opt):    
    version = "v1.0.0"
    DIR_tematdb = "./data_excel/"    
    
    # filename1 = "_tematdb_tep_excel_{:s}_{:05d}-{:05d}_confirmed_220606.xlsx".format(version,1,50)
    # filename2 = "_tematdb_tep_excel_{:s}_{:05d}-{:05d}_confirmed_220606.xlsx".format(version,51,100)
    # filename3 = "_tematdb_tep_excel_{:s}_{:05d}-{:05d}_confirmed_220606.xlsx".format(version,101,150)
    # filename4 = "_tematdb_tep_excel_{:s}_{:05d}-{:05d}_confirmed_220606.xlsx".format(version,151,200)
    # filename5 = "_tematdb_tep_excel_{:s}_{:05d}-{:05d}_confirmed_220606.xlsx".format(version,201,250)
    # filename6 = "_tematdb_tep_excel_{:s}_{:05d}-{:05d}_confirmed_220606.xlsx".format(version,251,300)
    # filename7 = "_tematdb_tep_excel_{:s}_{:05d}-{:05d}_confirmed_220606.xlsx".format(version,301,350)
    # filename8 = "_tematdb_tep_excel_{:s}_{:05d}-{:05d}_confirmed_220606.xlsx".format(version,351,400)
    # filename9 = "_tematdb_tep_excel_{:s}_{:05d}-{:05d}_confirmed_220606.xlsx".format(version,401,450)
    
    # files = [filename1, filename2, filename3, filename4, filename5, filename6,
    #           filename7, filename8, filename9 ]
    
    files = os.listdir('./data_excel/')
    
    fileindex = int((sampleid-1)/50)
    filename = files[fileindex]
    sheetname = "#{:05d}".format(sampleid)        
    
    try:
        mat = TEProp_xls.from_dict({'xls_filename': DIR_tematdb+filename,
                                'sheetname': sheetname, 'color': (sampleid/255, 0/255, 0/255)} ) 
        TF_mat_complete = True     
        mat.set_interp_opt(interp_opt)
        print(sampleid)
        return TF_mat_complete, mat
    except:
        print(filename, sampleid, 'data set is incompelete or empty')
        TF_mat_complete = False
        mat = False        
        return TF_mat_complete, mat



def get_df_tep_sample_given_dTs(sampleid, dTs):
    interp_opt = {MatProp.OPT_INTERP:MatProp.INTERP_LINEAR,\
                  MatProp.OPT_EXTEND_LEFT_TO:1,          # ok to 0 Kelvin
                  MatProp.OPT_EXTEND_RIGHT_BY:2000}        # ok to +50 Kelvin from the raw data

    TF_mat_complete, mat = tep_generator_from_excel_files(sampleid, interp_opt)
        
    try:
        # print( TF_mat_complete)
        # TF_mat_complete, mat = tep_generator_from_excel_files(sampleid, interp_opt)
        autoTc = mat.min_raw_T
        autoTh = mat.max_raw_T
        
        ztTc, ztTh = mat.ZT.raw_interval()
        
        # dTs = np.arange(5,2001,5)
        # dT_extended = 
        
        df = pd.DataFrame()
        alpha = mat.Seebeck(dTs)
        rho   = mat.elec_resi(dTs)
        kappa = mat.thrm_cond(dTs)
        raw_ZT = mat.ZT(dTs)
        
        sigma = 1/rho
        PF    = alpha*alpha*sigma
        Z     = PF/kappa
        ZT    = Z*dTs
        RK    = rho*kappa
        Lorenz= RK/dTs
        
        df['sampleid'] = [sampleid]*len(dTs)
        df['Temperature'] = dTs.copy()
        df['alpha'] = alpha
        df['rho']   = rho
        df['kappa'] = kappa
        
        df['sigma'] = sigma
        # df['PF']    = PF
        # df['Z']     = Z
        df['ZT_tep_reevaluated']    = ZT
        # df['RK']    = RK
        # df['Lorenz']= Lorenz
        
        autoTcTh_buffer_dT = 0
        # df['autoTc'] = autoTc
        # df['autoTh'] = autoTh
        df['autoTcTh_buffer_dT'] = autoTcTh_buffer_dT
        df['is_Temp_in_autoTcTh_range'] = (df.Temperature >= autoTc-autoTcTh_buffer_dT) & (df.Temperature <= autoTh+autoTcTh_buffer_dT)
        
        df['ZT_author_declared'] = raw_ZT
        df['is_Temp_in_ZT_author_declared']  = (df.Temperature > ztTc) & (df.Temperature < ztTh)
        
        df_extended_tep_sample = df.copy()
        # df_autoTcTh_tep_sample = df[ df.TF_Temp_is_in_autoTcTh_range ].copy()
        # df_ztTcTh_tep_sample   = df[ df.TF_Temp_is_in_rawZT_range ].copy()
        
        return df_extended_tep_sample
    except:
        df_extended_tep_sample = pd.DataFrame()
        return df_extended_tep_sample
    # df_autoTcTh_tep.reset_index(inplace=True)
    
    # return df_extended_tep_sample, df_autoTcTh_tep_sample


interp_opt = {MatProp.OPT_INTERP:MatProp.INTERP_LINEAR,\
              MatProp.OPT_EXTEND_LEFT_TO:1,          # ok to 0 Kelvin
              MatProp.OPT_EXTEND_RIGHT_BY:2000}        # ok to +50 Kelvin from the raw data


## read single TEP csv file 
## to get dataframe for TEPs, alpha, rho, kappa, ZT
teMatDb_csv_file = "./data_csv/"+"tematdb_v1.1.0_completeTEPset.csv"
df_raw_tep, df_raw_alpha, df_raw_rho, df_raw_kappa, df_raw_ZT = get_df_from_tematdb(teMatDb_csv_file)


##
## sampleid list from dataframe of single csv tep file
sampleid_list = [1,2]
sampleid_list = df_raw_tep.sampleid.unique().tolist()

##
## list that will be used for dataframe of TEPs for given dTs
## where dTs is a temperature list for raw alpha T points
df_tep_list = []
for sampleid in sampleid_list:
  
    df_raw_alpha_sampleid = df_raw_alpha[ df_raw_alpha.sampleid == sampleid] 
    mat = tep_generator_from_csv_files(sampleid, df_raw_alpha, df_raw_rho, df_raw_kappa, df_raw_ZT, interp_opt)
    autoTc = mat.min_raw_T
    autoTh = mat.max_raw_T
    
    dTs = list(df_raw_alpha_sampleid.Temperature)
    # dTs = dTs[ dTs > autoTc-1]
    # dTs = dTs[ dTs < autoTh+1]    
    
    if len(dTs)>0:
        df_tep_sample_given_dTs = get_df_tep_sample_given_dTs(sampleid, dTs)
        df_tep_sample_given_dTs['autoTc'] = autoTc
        df_tep_sample_given_dTs['autoTh'] = autoTh
        df_tep_list.append(df_tep_sample_given_dTs.copy())


df_TEP_given_Ts = pd.concat( df_tep_list, ignore_index= True)


version = "v1.1.0"
dbname = 'tematdb'
versionshort  = 'tematdb_{:s}_TEPatTsOfAlphas'.format(version)
versionprefix = 'tematdb_{:s}_TEPatTsOfAlphas_convertedOn_{}_'.format(version,formattedDate)

datetimeupdate =  datetime.now()
sampleid_min = df_TEP_given_Ts.sampleid.min()
sampleid_max = df_TEP_given_Ts.sampleid.max()

versiontype ="range"
versionlabel = versionprefix+'_{:s}_{:.0f}_to_{:.0f}'.format(versiontype,sampleid_min, sampleid_max)

df_TEP_given_Ts['dbname']  = dbname
df_TEP_given_Ts['version'] = version
df_TEP_given_Ts['versionlabel'] = versionlabel
df_TEP_given_Ts['update']  = datetimeupdate
df_TEP_given_Ts['pykeri_compatible'] = True
df_TEP_given_Ts['interpolation_method'] = "piecewise_liniear"
df_TEP_given_Ts['extrapolation_method'] = "constant"

df_TEP_given_Ts.to_csv( "./data_csv/"+versionlabel+'.csv',index=False )
df_TEP_given_Ts.to_csv( "./data_csv/"+versionshort+'.csv',index=False )







