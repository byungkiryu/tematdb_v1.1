# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 17:24:07 2023

@author: cta4r

This is the program to visualize 

"""


import math
import numpy as np
import pandas as pd
import streamlit as st

from matplotlib import pyplot as plt
import scipy.stats as stats  

from datetime import datetime

from pykeri.scidata.matprop import MatProp
# from pykeri.thermoelectrics.TEProp import TEProp
from pykeri.thermoelectrics.TEProp_xls import TEProp as TEProp_xls
from pykeri.thermoelectrics.TEProp_df import TEProp as TEProp_df
from pykeri.thermoelectrics.solver1d.leg import Leg
from pykeri.thermoelectrics.solver1d.environment import Environment
from pykeri.thermoelectrics.solver1d.device import Device

from pykeri.byungkiryu import byungkiryu_util as br


        
from library.tematdb_util import get_Ts_TEPZT
from library.tematdb_util import draw_mat_teps, tep_generator_from_excel_files
from library.draw_ZT_errors_with_mat import draw_mat_ZT_errors
from library.dev_performance import set_singleleg_device, run_pykeri, draw_dev_perf

formattedDate, yyyymmdd, HHMMSS = br.now_string()

st.set_page_config(page_title="teMatDb")


st.title("teMatDb")
st.subheader(":blue[t]hermo:blue[e]lectric :blue[Mat]erial :blue[D]ata:blue[b]ase")
st.markdown("- High quality thermoelectric (TE) database (DB), teMatDb (ver1.1.1)")
st.markdown("- That can be used for data analytics, machine learning and AI")


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Material Property", 
                                                 "Data Self-Consistency", 
                                                 "Data Distribution",
                                                 "Theory & Equations", 
                                                 "Link", 
                                                 "Contact"])


with tab3:  
    st.header(":red[tab1c to be made]")    

with tab4:        
    st.header(":blue[Thermoelectric Data]")
    with st.expander("See details.."):
        st.subheader(":red[Data Sources]")  
        st.subheader(":red[Thermoelectric Materials to Digital Data]")  
        st.subheader(":red[Thermoelectric Device Data]")  
        
    st.header(":blue[teMatDb]")
    with st.expander("See details.."):
        st.subheader(":red[Data digitization and teMatDb]")
        st.subheader(":red[Metadata in teMatDb]")
        st.subheader(":red[Possible Source of Errors]")
        st.subheader(":red[Error Analysis Methods]")
        
    st.header(":blue[Theory]")
    with st.expander("See details.."):
        st.subheader(":red[Introduction to Thermoelectricity]")
        st.subheader(":red[Thermoelectric Transport]")
        st.subheader(":red[Thermoelectric Materials]")
        st.subheader(":red[Thermoelectric Device and Modules for Power Generation]")
        st.subheader(":red[Thermoelectric Efficiency]")  
    
    st.header(":blue[Advanced Efficiency Theory and Calculations]")
    with st.expander("See details.."):
        st.subheader(":red[Thermoelectric Conversion Efficiency with Temperature-dependent Thermoelectric Properties]")
        st.subheader(":red[Thermoelectric Differential Equations]")
        st.subheader(":red[Single Parameter Theory I: ZT in Constant-Property Model]")
        st.subheader(":red[Single Parameter Theory II: ZT in Constant-Seebeck coefficient Model]")
        st.subheader(":red[Three Thermoelectric Degrees of Freedom]")
        st.subheader(":red[Thermoelectric Inegtral Equations in the One-dimensional (1-D)]")
        st.subheader(":red[Thermoelectric Efficiency Solving Algorithm]")
        st.subheader(":red[Efficiency Solver: pykeri 2019]")
        st.subheader(":red[High-dimensional analysis: Optimal Leg Aspect Ratio]")
        st.subheader(":red[Dimensional Mapping to 1-D]")

    st.header(":blue[Advanced Device Design Theory]")
    with st.expander("See details.."):
        st.subheader(":red[Thermoelectric Algebra]")
        st.subheader(":red[Thermoelectric Inequality]")
        st.subheader(":red[Thermoelectric Circuit Model]")
        st.subheader(":red[Contact Resistances]")
        
    st.header(":blue[Thermoelectric Efficiency Map]")
    with st.expander("See details.."):
        st.subheader(":red[Best Thermoelectric Efficiency of Ever-Explored Materials], as of 2021-Nov-24")
        
        
        
    st.header(":blue[References]")
    # st.markdown("[1] Chung, Jaywan, and Byungki Ryu. “Nonlocal Problems Arising in Thermoelectrics.” Mathematical Problems in Engineering 2014 (December 15, 2014): e909078. https://doi.org/10.1155/2014/909078.")
    st.markdown("[1] Ryu, Byungki, Jaywan Chung, Eun-Ae Choi, Pawel Ziolkowski, Eckhard Müller, and SuDong Park. “Counterintuitive Example on Relation between ZT and Thermoelectric Efficiency.” Applied Physics Letters 116, no. 19 (May 13, 2020): 193903. https://doi.org/10.1063/5.0003749.")
    st.markdown("[2] Chung, Jaywan, Byungki Ryu, and SuDong Park. “Dimension Reduction of Thermoelectric Properties Using Barycentric Polynomial Interpolation at Chebyshev Nodes.” Scientific Reports 10, no. 1 (August 10, 2020): 13456. https://doi.org/10.1038/s41598-020-70320-7.")
    st.markdown("[3] Ryu, Byungki, Jaywan Chung, and SuDong Park. “Thermoelectric Degrees of Freedom Determining Thermoelectric Efficiency.” iScience 24, no. 9 (September 24, 2021): 102934. https://doi.org/10.1016/j.isci.2021.102934.")
    st.markdown("[4] Chung, Jaywan, Byungki Ryu, and Hyowon Seo. “Unique Temperature Distribution and Explicit Efficiency Formula for One-Dimensional Thermoelectric Generators under Constant Seebeck Coefficients.” Nonlinear Analysis: Real World Applications 68 (December 1, 2022): 103649. https://doi.org/10.1016/j.nonrwa.2022.103649.")
    st.markdown("[5] Ryu, Byungki, Jaywan Chung, Masaya Kumagai, Tomoya Mato, Yuki Ando, Sakiko Gunji, Atsumi Tanaka, et al. “Best Thermoelectric Efficiency of Ever-Explored Materials.” iScience 26, no. 4 (April 21, 2023): 106494. https://doi.org/10.1016/j.isci.2023.106494.")
    st.write("(Style: Chicago Manual of Style 17th edition (full note))")

with tab5:
    st.subheader("Thermoelectric Power Generator Web Simulator Lite ver.0.53a")
    st.write(":blue[https://tes.keri.re.kr/]")
    st.write(":blue[https://github.com/jaywan-chung/teg-sim-lite]")
    
    st.subheader("Korea Electrotechnology Research Institute (KERI)")
    st.write(":blue[https://www.keri.re.kr/]")
   
    st.subheader("Alloy Design DB (v0.33)")
    st.write(":blue[https://byungkiryu-alloydesigndb-demo-v0-33-main-v0-33-u86ejf.streamlit.app/]")
    
with tab6:   
    st.header(":blue[KERI Thermoelectric Science TEAM]")
    with st.expander("See Members:", expanded=False):   
        st.subheader("SuDong Park, Dr. (박수동)")
        st.subheader("Byungki Ryu, Dr. (류병기)")
        st.markdown("byungkiryu at keri.re.kr")
        st.subheader("Jaywan Chung, Dr. (정재환)")
        st.subheader("Jongho Park, Mr. (박종호)")
        st.subheader("Ji-Hee Son, Ms. (손지희)")
        st.subheader("Jeongin Jang, Miss (장정인)")
        st.subheader("Sungjin Park, Dr. (박성진)")
    
    st.header(":blue[Visit KERI. How to Come?]")
    with st.expander("See Maps:", expanded=False):        
        st.markdown("")
        st.image("./image/southkorea_map_screenshot.png")    
        st.caption("(South Korea Map) Changwon-si")
        st.image("./image/changwon_map_screenshot.png")    
        st.caption("(Changwon Map) Korea Electrotechnology Research Institute (KERI)")
        st.image("./image/map_2023.jpg")   
        st.caption("(KERI map) Office and Lab, building #3 and #5")
    
    st.header(":blue[QR Code]")
    with st.expander("See QR code (v1.1.1):", expanded=False):            
        st.image("./image/"+"qrcode_tematdb-v1-1-main-v1-1-1-abc.streamlit.app.png")
        st.subheader("https://tematdb-v1-1-main-v1-1-1-abc.streamlit.app/")
    with st.expander("See QR code (v1.1.0):", expanded=False):            
        st.image("./image/"+"qrcode_tematdb-v1-1-main-v1-1-0-abc.streamlit.app.png")
        st.subheader("https://tematdb-v1-1-main-v1-1-0-abc.streamlit.app/")
    with st.expander("See QR code (v1.0.2):", expanded=False):            
        st.image("./image/"+"qrcode_tematdb-v1-0-0-main-v1-0-2-abc.streamlit.app.png")
        # st.header("https://qrco.de/bds6GG/")
    with st.expander("See QR code (v1.0.1):", expanded=False):            
        st.image("./image/"+"qrcode_tematdb-v1-0-0-main-v1-0-1-abc.streamlit.app.png")
        # st.header("https://qrco.de/bds6GG/")
    with st.expander("See QR code (v1.0.0):", expanded=False):            
        st.image("./image/"+"qrcode_tematdb-v1-0-0-main-v1-0-0-abc.streamlit.app.png")


    
###############
###############
###############
## Sidebar, choose DB, choose mat
with st.sidebar:
    st.subheader(":red[Select TE Mat. DB]")
    options =['teMatDb','teMatDb_expt','Starrydata2']
    db_mode = st.radio( 'Select :red[Thermoelectric DB] :',
        options=options, index=0,
        label_visibility="collapsed")    
    
    if (db_mode == 'teMatDb'):
        file_tematdb_metadata_csv = "_tematdb_metadata_v1.1.0-20230412_brjcsjp.xlsx"
        file_tematdb_db_csv       =  "tematdb_v1.1.0_completeTEPset.csv"
        
        df_db_meta0 = pd.read_excel("./"+file_tematdb_metadata_csv, sheet_name='list', )
        df_db_meta = df_db_meta0
        df_db_meta.index = list(df_db_meta.sampleid.copy())
        
        df_db_csv = pd.read_csv("./data_csv/"+file_tematdb_db_csv)        
        
        df_db_extended_csv = pd.read_csv("./data_csv/"+"tematdb_v1.1.0_extendedTEPset.csv")
        df_db_extended_csv = df_db_extended_csv[ df_db_extended_csv.is_Temp_in_autoTcTh_range ]
        df_db_extended_csv = df_db_extended_csv[ df_db_extended_csv.is_Temp_in_ZT_author_declared ] 
        
        file_tematdb_error_csv = "error.csv"
        df_db_error0 = pd.read_csv("./data_error_analysis/"+file_tematdb_error_csv)
        err_cols = ['sampleid','JOURNAL', 'YEAR', 
               'Corresponding_author_main', 'Corresponding_author_institute',
               'Corresponding_author_email', 'figure_number_of_targetZT',
               'label_of_targetZT_in_figure', 'figure_label_description',]
        # df_db_error0['Linf_over_avgZT']  = df_db_error0.Linf / df_db_error0.avgZT_TEP_on_Ts_TEP
        # df_db_error0['Linf_over_peakZT'] = df_db_error0.Linf / df_db_error0.peakZT_TEP_on_Ts_TEP
        df_db_error = pd.merge( df_db_error0, df_db_meta[err_cols], on='sampleid', how='left')
        df_db_error.index = list(df_db_error.sampleid.copy())        
         
        ## choose sampleid
        st.subheader(":red[Select sampleid]")
        option_sampleid = list(df_db_meta['sampleid'].unique())
        sampleid = st.selectbox('Select or type sampleid:',
            option_sampleid, index=0,
            label_visibility="collapsed")   
        
        df_db_meta_sampleid = df_db_meta[ df_db_meta['sampleid'] == sampleid]
        doi = df_db_meta_sampleid.DOI.iloc[0]
        link_doi = '[DOI: {}](http://www.doi.org/{})'.format(doi,doi)
        st.markdown(link_doi, unsafe_allow_html=True)
        corrauthor = df_db_meta_sampleid.Corresponding_author_main.iloc[0]
        corrinstitute  = df_db_meta_sampleid.Corresponding_author_institute.iloc[0] 
        corremail  = df_db_meta_sampleid.Corresponding_author_email.iloc[0] 
        st.markdown("First Author: :red[need to update]")
        st.markdown("Correspondence: {}".format(corrauthor)) 
        st.markdown("Institute: {}".format(corrinstitute))         
        
        interp_opt = {MatProp.OPT_INTERP:MatProp.INTERP_LINEAR,\
                      MatProp.OPT_EXTEND_LEFT_TO:1,          # ok to 0 Kelvin
                      MatProp.OPT_EXTEND_RIGHT_BY:2000}        # ok to +50 Kelvin from the raw data
        TF_mat_complete, mat = tep_generator_from_excel_files(sampleid, interp_opt)
    
        label_db = "DB: {}".format(db_mode)
        label_sampleid = "sampleid: {}".format(sampleid)
        label_doi = '[DOI: {}]'.format(doi)    
        
        st.subheader(":red[Data Filter]")
        
        with st.form("Data Error Filter Criteria"):
            st.markdown("Minimum value: 0.1. The data larger than ths will be filtered out")
            cri_cols = ['davgZT', 'dpeakZT','Linf',
                        'Linf_over_avgZT','Linf_over_peakZT']
            cri_vals_def = [0.1, 0.1, 0.1, 0.3, 0.3]
            cri_vals0 = st.number_input('N for criteria: {} > N'.format(cri_cols[0]),
                                          min_value = 0.01, value=cri_vals_def[0],step=0.05)
            cri_vals1 = st.number_input('N for criteria: {} > N'.format(cri_cols[1]),
                                          min_value = 0.01, value=cri_vals_def[1],step=0.05)
            cri_vals2 = st.number_input('N for criteria: {} > N'.format(cri_cols[2]),
                                          min_value = 0.01, value=cri_vals_def[2],step=0.05)        
            cri_vals3 = st.number_input('N for criteria: {} > N'.format(cri_cols[3]),
                                          min_value = 0.01, value=cri_vals_def[3],step=0.05)  
            cri_vals4 = st.number_input('N for criteria: {} > N'.format(cri_cols[4]),
                                          min_value = 0.01, value=cri_vals_def[4],step=0.05)  
            submitted = st.form_submit_button("Submit criteria")
            
            if submitted:                   
                cri_vals = [cri_vals0, cri_vals1, cri_vals2, cri_vals3, cri_vals4]
            else:
                cri_vals = cri_vals_def

    elif (db_mode == 'Starrydata2'):
        pass
    else:
        pass

###############
###############
###############
## Material data for given sampleid
with tab1:  
    ## Read mat, TEP
    # st.header("[db_mode  = :blue[{}]]".format(db_mode) )
    st.header(":blue[I. DB MetaData Table]")
    with st.expander("See material metadata:", expanded=True):
        st.write(df_db_meta)                

    st.header(":blue[II. Material Data] ")
    st.subheader(":red[[db_mode  = :blue[{}]]]".format(db_mode) + ":red[[sampleid = :blue[{}]]]".format( sampleid))
    
    st.subheader(":red[Material Summary]")
    st.markdown(link_doi, unsafe_allow_html=True)
    
    colnames = ['sampleid', 'DOI', 'JOURNAL', 'YEAR',
                'GROUP', 'BASEMAT','Composition_by_element', 'Composition_detailed',
                'mat_dimension(bulk, film, 1D, 2D)', 'SC/PC', 'Reaction', 'Milling',
                'SINTERING', 'PostProcessing']
    for colname in colnames:
        st.markdown("{}: :blue[{}]".format(colname, df_db_meta_sampleid[colname].iloc[0]))
    
    st.subheader(":red[Material Information]")
    st.write(df_db_meta_sampleid)

 
    ## print material(mat) tep
    st.subheader(":red[Transport Properties] (Table)")
    df_db_csv_sampleid = df_db_csv[ df_db_csv['sampleid'] == sampleid]
    if not TF_mat_complete:
        st.write(':red[TEP is invalid because TEP set is incomplete..]')    
    if TF_mat_complete:
        with st.expander("See rawdata of material TEPs:"):
            st.write(df_db_csv_sampleid)
    
    ## draw material(mat) tep
    st.subheader(":red[Transport Properties] (interpolated) (Figure)")
    if not TF_mat_complete:
        st.write(':red[TEP is invalid because TEP set is incomplete..]')    
    if TF_mat_complete:        
        fig1, fig2 = draw_mat_teps(mat, 
                                   label_db=label_db, 
                                   label_sampleid=label_sampleid, 
                                   label_doi=label_doi
                                   )
        with st.expander("See TEP curves:", expanded=True):        
            st.pyplot(fig1)   
            st.caption("Figure. Thermoelectric Properties of :blue[sampleid={}] in :blue[{}].".format(sampleid,db_mode))
        with st.expander("See rho, PF, RK, Loenz curves:", expanded=True):        
            st.pyplot(fig2)        
            st.caption("Figure. Extended Thermoelectric Properties of :blue[sampleid={}] in :blue[{}].".format(sampleid,db_mode))
        with st.expander("See Interpolation schemes:", expanded=False):  
            ZT_raw   = np.array( mat.ZT.raw_data() ).T
            autoTc = mat.min_raw_T
            autoTh = mat.max_raw_T    
            TcZT = min(ZT_raw[0])
            ThZT = max(ZT_raw[0])    
            TcTEPZT = max(autoTc, TcZT,1)
            ThTEPZT = min(autoTh, ThZT)            
            st.markdown(":blue[TEP interpolation interval (Tc, Th) = ({:6.2f} K, {:6.2f} K)]  \n".format(autoTc,autoTh) \
                        + ":red[ZT interpolation interval (Tc, Th) = ({:6.2f} K, {:6.2f} K)]  \n".format(TcZT,ThZT) \
                        + ":green[Union of TEP and ZT interpolation interval (Tc, Th) = ({:6.2f} K, {:6.2f} K)]".format(TcZT,ThZT))

        
    ## Digitized data quality using error analysis
    st.header(":blue[III. ZT Self-consistency Analyzer]")
    st.subheader(":red[[db_mode  = :blue[{}]]]".format(db_mode) + ":red[[sampleid = :blue[{}]]]".format( sampleid))
    st.subheader(":red[Material Error (or Noise) Statistics]")
    if not TF_mat_complete:
        st.write(':red[TEP is invalid because TEP set is incomplete..]')    
    if TF_mat_complete:   
        try:
            df_db_error_sampleid = df_db_error[ df_db_error['sampleid'] == sampleid]
        except:
            st.markdown('Yet, no error reports')
        else:            
            st.write(df_db_error_sampleid)
            
            if ( np.abs( df_db_error_sampleid.dpeakZT.iloc[0] ) > 0.1 ):
                st.markdown("**:red[Warning: peak ZT mismatch is larger than 0.1]**")
            else:
                st.markdown("**:blue[Self-consistent: peak ZT mismatch is smaller than, 0.1]**")
            st.markdown(":black[Raw peak-ZT (from published figure data): peak-ZT raw = {:6.2f}]".format(df_db_error_sampleid.peakZT_raw_on_Ts_ZT.iloc[0]))
            st.markdown(":black[Calculated peak-ZT (from TEP interpolated): peak-ZT TEP = {:6.2f}]".format(df_db_error_sampleid.peakZT_TEP_on_Ts_TEP.iloc[0])) 
                
            if ( np.abs( df_db_error_sampleid.davgZT.iloc[0] ) > 0.1 ):
                st.markdown("**:red[Warning: average ZT mismatch is larger than 0.1]**")
            else:
                st.markdown("**:blue[Self-consistent: average ZT mismatch is smaller than, 0.1]**")
            st.markdown(":black[Raw avg-ZT (from published figure data): avg-ZT raw = {:6.2f}]".format(df_db_error_sampleid.avgZT_raw_on_Ts_ZT.iloc[0]))
            st.markdown(":black[Calculated avg-ZT reevaulated (from TEP interpolated): avg-ZT TEP = {:6.2f}]".format(df_db_error_sampleid.avgZT_TEP_on_Ts_TEP.iloc[0])) 
        
            with st.expander("How to calculateSee Lp errors and etc...:", expanded=False):        
                st.markdown(":red[error was calculated blah blbah using following equations (to be filled later)]")
                
            fig3 = draw_mat_ZT_errors(mat, label_db=label_db, label_sampleid=label_sampleid, label_doi=label_doi)
            st.pyplot(fig3)        
            st.caption("Figure. ZT error analysis of :blue[sampleid={}] in :blue[{}].".format(sampleid,db_mode))
            
          
    st.header(":blue[IV. Material Performance]")
    st.subheader(":red[[db_mode  = :blue[{}]]]".format(db_mode) + ":red[[sampleid = :blue[{}]]]".format( sampleid))
    if not TF_mat_complete:
        st.write(':red[TEP is invalid because TEP set is incomplete..]')    
    if TF_mat_complete:
        # st.write(':blue[TEP is valid..]')     
        
        # autoTc, autoTh = float(mat.min_raw_T), float(mat.max_raw_T)
        limitTc = float( math.floor(autoTc/50)*50.0 )
        limitTh = float( math.ceil(autoTh/50)*50.0 )
        
        st.markdown(":blue[TEP interpolation:] piecewise-linear manner.  \n "
                    +":blue[TEP extrapolation:] constant-extrapolated.  \n "    
                    +":blue[TEP interpolation at the interval (Tc, Th) = ({:6.2f} K, {:6.2f} K)]".format(autoTc,autoTh))
        
        
        st.subheader(":red[Singleleg Device Spec.]")
        with st.form("my_form"):
            st.write("Valid temperature range considering Ts of TEPs: Tc > {:6.2f} K, Th < {:6.2f} K".format(autoTc, autoTh))
            
            col1, col2 = st.columns([2,2])
            with col1:
                Tc = st.number_input(r'$T_c$: cold side temperature in [K]',
                                min_value = limitTc, max_value = limitTh, step = 25.0,
                                value = limitTc+25 )
                Th = st.number_input(r'$T_h$: hot side temperature in [K]',
                                min_value = limitTc, max_value = limitTh, step = 25.0,
                                value = limitTh-25 )
            with col2:
                leg_length_in_mm = st.number_input('leg_length in mm',
                                                     min_value = 0.5, max_value = 10.0, step = 0.5,
                                                     value = 3.0)
                leg_area_in_mmsq = st.number_input('leg_area in mm x mm',
                                                     min_value = 0.25, max_value = 25.0, step = 1.0,
                                                     value = 9.0)
                N_leg = st.number_input('number of legs',
                                                      min_value = 1, max_value = 1, step = 1,
                                                      value = 1, disabled=True)
            
            submitted = st.form_submit_button("Calculate Material Performance (singleleg device).")
            if submitted:                   
                leg_length = leg_length_in_mm *1e-3
                leg_area   = leg_area_in_mmsq *1e-6
                N_leg      = 1  
            else:
                leg_length = 3 *1e-3
                leg_area   = 9 *1e-6
                N_leg      = 1 

        dev = set_singleleg_device(mat, leg_length,leg_area,N_leg,Th,Tc)            
        # dev = set_singleleg_device(df_db_csv,sampleid,leg_length,leg_area,N_leg,Th,Tc)
        df_dev_run_currents_result, df_dev_run_powMax_result, df_dev_run_etaOpt_result = run_pykeri(dev, sampleid,leg_length,leg_area,N_leg,Th,Tc)
        fig4 = draw_dev_perf(df_dev_run_currents_result, df_dev_run_powMax_result, df_dev_run_etaOpt_result,
                             label_db, label_sampleid, label_doi)
        st.pyplot(fig4)
        with st.expander("See Material performance curves:", expanded=True):        
            st.write("currents performances")
            st.write(df_dev_run_currents_result)
            st.write("Power max performances")
            st.write(df_dev_run_powMax_result)
            st.write("Optimal efficiency performances")
            st.write(df_dev_run_etaOpt_result)
            
###############
###############
###############
## DB Stat
with tab2:     

    st.header(":blue[DataFrame for Error Table]")

    st.write(df_db_error)
    
    st.header(":blue[Error Analysis based on ZT self-consistency]")
    with st.expander("See analysis:", expanded=True): 
        # cri_cols = ['davgZT', 'dpeakZT','Linf']
        # cri_vals = [0.10, 0.10, 0.10]
        
        
        
        df_db_error_criteria_list = []
        error_criteria_list = []
        sampleid_list_df_db_error_criteria = []
        df4_db_error_filtered = df_db_error.copy()
        df5_db_error_anomaly = df_db_error.copy()
        for cri_col, cri_val in zip(cri_cols, cri_vals):   
            error_criteria = np.abs( df_db_error[cri_col] ) > cri_val 
            error_criteria_list.append(error_criteria.copy())
            df_db_error_criteria = df_db_error[ error_criteria ].copy()
            df_db_error_criteria.sort_values(by=cri_col,ascending=False, inplace=True)
            df_db_error_criteria.set_index('sampleid', inplace=True, drop=False)
            
            df4_db_error_filtered = df4_db_error_filtered[ np.abs( df4_db_error_filtered[cri_col] ) < cri_val ].copy()
            df5_db_error_anomaly  = df5_db_error_anomaly[ np.abs( df5_db_error_anomaly[cri_col] ) < cri_val ].copy()
            
            cri_str = ":red[Noisy samples: {} > {:.2f}]".format(cri_col, cri_val)
            st.subheader(cri_str)
            st.markdown("There are :red[{}] noisy-cases.".format(len(df_db_error_criteria)) )
            st.write(df_db_error_criteria)
            df_db_error_criteria_list.append(df_db_error_criteria)
            sampleid_list = df_db_error_criteria['sampleid'].unique().tolist()
            st.write(sampleid_list)
            sampleid_list_df_db_error_criteria = sampleid_list_df_db_error_criteria + sampleid_list
            del df_db_error_criteria
    
    st.header(":blue[DB Before Filtering]")
    with st.expander("See plots:", expanded=True):   
        df1 = df_db_extended_csv
        df2 = df_db_error
        
        st.subheader(":red[ZT Error Correlation]")
        def draw_ZT_error_correlation(df_db_error):
            df_to_plotly = df_db_error[ df_db_error.Linf > 0].copy()
            hover_data = ['sampleid','doi','Corresponding_author_main','davgZT','dpeakZT','L2','L3','errMax','errMin']
            
            import plotly.express as px    
            fig = px.scatter(
                df_to_plotly,
                x='peakZT_TEP_on_Ts_TEP',
                y='peakZT_raw_on_Ts_ZT',
                size='Linf',
                color='peakZT_raw_on_Ts_ZT',        
                hover_name="sampleid",
                hover_data=hover_data
                )
            figa = fig
            st.plotly_chart(figa)
            st.caption("Peak ZT bias plot: ZT_raw from author publication vs. ZT_TEP from TEP reevalulation. Size is Linf. Color is peakZT_raw_on_Ts_ZT.")
        
            fig = px.scatter(
                df_to_plotly,
                x='avgZT_TEP_on_Ts_TEP',
                y='avgZT_raw_on_Ts_ZT',
                size='Linf',
                color='peakZT_raw_on_Ts_ZT',        
                hover_name="sampleid",
                hover_data=hover_data
                )
            figa = fig
            st.plotly_chart(figa)
            st.caption("ZT average bias plot: ZT_raw from author publication vs. ZT_TEP from TEP reevalulation. Size is Linf. Color is peakZT_raw_on_Ts_ZT.")
            
            figc = px.scatter(
                df_to_plotly,
                x='davgZT',
                y='dpeakZT',
                size='Linf',
                color='peakZT_raw_on_Ts_ZT', 
                hover_name="sampleid",
                hover_data=hover_data
                )
            st.plotly_chart(figc)    
            st.caption("ZT deviation correlation between d(peakZT) and d(avgZT). Size is Linf. Color is peakZT_raw_on_Ts_ZT.")
                
            fig = px.scatter(
                df_to_plotly,
                x='L2',
                y='dpeakZT',
                size='Linf',
                color='peakZT_raw_on_Ts_ZT',   
                hover_name="sampleid",
                hover_data=hover_data
                )
            figb = fig
            st.plotly_chart(figb)
            st.caption("Error Effect on ZT bias. Size is Linf. Color is peakZT_raw_on_Ts_ZT. Size is Linf. Color is peakZT_raw_on_Ts_ZT.")
            
            fig = px.scatter(
                df_to_plotly,
                x='L3',
                y='dpeakZT',
                size='Linf',
                color='peakZT_raw_on_Ts_ZT',   
                hover_name="sampleid",
                hover_data=hover_data
                )
            figb = fig
            st.plotly_chart(figb)
            st.caption("Error Effect on ZT bias")    
            
            fig = px.scatter(
                df_to_plotly,
                x='Linf',
                y='dpeakZT',
                # size='Linf',
                color='peakZT_raw_on_Ts_ZT',   
                hover_name="sampleid",
                hover_data=hover_data
                )
            figb = fig
            st.plotly_chart(figb)
            st.caption("Error Effect on ZT bias")           
        draw_ZT_error_correlation(df2)
        

    def draw3QQ(df1,df2): 
        figsize = (13,3) 
        # fig, axs = plt.subplots(1,3,figsize=figsize, sharex=True, constrained_layout=True )  
        fig, axs = plt.subplots(1,3,figsize=figsize, sharex=True,  )  
        fig.subplots_adjust(wspace=0.4, hspace=0.2)
        (ax1, ax2, ax3) = axs 
        
        ax = ax1
        tep_title = r'Q-Q plot of $ZT$ deviation'
        X = df1.ZT_author_declared - df1.ZT_tep_reevaluated
        stats.probplot(X,dist=stats.norm, plot=ax, rvalue=True)     
        ax.set_ylabel(r"$\delta$($ZT$)")
        Xlim = max( np.abs(X) ) 
        Xlim2 = (math.ceil(Xlim/0.1)+0.0) *0.1
        # Xrange = np.arange(-Xlim2, Xlim2+0.05,0.1)
        Xlim3 = Xlim2 *1.1
        ax.set_ylim(-Xlim3,Xlim3)
        ax.set_xlim(-4.5,4.5)
        # ax.legend(loc=2)
        ax.set_title(tep_title)


        ax = ax2
        tep_title = r'Q-Q plot of avg-$ZT$ deviation'
        X = df2.dropna(axis=0,subset='davgZT').davgZT
        stats.probplot(X,dist=stats.norm, plot=ax, rvalue=True)    
        ax.set_ylabel(r"$\delta$(avg-$ZT$)")    
        Xlim = max( np.abs(X) ) 
        Xlim2 = (math.ceil(Xlim/0.1)+0.0) *0.1
        # Xrange = np.arange(-Xlim2, Xlim2+0.05,0.1)
        Xlim3 = Xlim2 *1.1
        ax.set_ylim(-Xlim3,Xlim3)
        ax.set_xlim(-4.5,4.5)
        ax.set_title(tep_title)
    
        ax = ax3
        tep_title = r'Q-Q plot of peak-$ZT$ deviation'
        X = df2.dropna(axis=0,subset='dpeakZT').dpeakZT
        stats.probplot(X,dist=stats.norm, plot=ax, rvalue=True)    
        ax.set_ylabel(r"$\delta$(peak-$ZT$)")    
        Xlim = max( np.abs(X) ) 
        Xlim2 = (math.ceil(Xlim/0.1)+0.0) *0.1
        # Xrange = np.arange(-Xlim2, Xlim2+0.05,0.1)
        Xlim3 = Xlim2 *1.1
        ax.set_ylim(-Xlim3,Xlim3)
        ax.set_xlim(-4.5,4.5)
        ax.set_title(tep_title)
        
        for ax in (ax1, ax2, ax3):
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')
            ax.xaxis.set_tick_params(which='both', direction='in', labelbottom=True)
            ax.yaxis.set_tick_params(which='both', direction='in', labelbottom=True)        
        return fig
    
    st.subheader(":red[QQ analysis]")
    fig_before_filter = draw3QQ(df1,df2)
    st.pyplot(fig_before_filter)

    st.header(":blue[DB After Filtering]")
    with st.expander("See plots:", expanded=True):   
        df4_db_error_filtered['notNoisy'] = True
        df3_df_db_extended_csv_filtered = pd.merge( df_db_extended_csv, df4_db_error_filtered[['sampleid','notNoisy']], on='sampleid', how='left')
        df3_df_db_extended_csv_filtered = df3_df_db_extended_csv_filtered[ df3_df_db_extended_csv_filtered.notNoisy == True ].copy()
        
        df3, df4 = df3_df_db_extended_csv_filtered, df4_db_error_filtered
        
        st.subheader(":red[ZT Error Correlation]")
        draw_ZT_error_correlation(df4)

    
    st.subheader(":red[QQ analysis]")
    fig_after_filter = draw3QQ(df3,df4)
    st.pyplot(fig_after_filter)      
