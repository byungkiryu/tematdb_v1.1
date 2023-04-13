# tematdb_v1.1

:blue[How to download and use.]
1) download the release zip
2) install the required libraries; see requirements.txt. you will need numpy, plotly, openpyxl, pandas, pyparsing, scipy, matplotlib
3) make environment and pip intall streamlit therein
4) set the path for pykeri in the root of release
5) command "streamlit run streamlit_app.py"

:blue[Ultrahigh quality thermoelctric db]
This teMatDb aims ultrahigh self-consistent transport properties for machine learning and transport mechanism analysis. Thermoelectric properties of each sample can be browsed. Using self-consistency criteria for database, one can list up the erroneous samples using the data filters and obtain the errorless sample lists also.

:blue[Here we have three data filters.]
1) average ZT filter for average ZT deviation between ZT direictly digitized from the figure and ZT calculated from TEP curves. This is sensitive to T range mismatch and ZT bias error.
2) peak ZT filter for peak ZT deviation between figure and TEP. This is sensitive to peak ZT bias error and unwanted extrapolation, mainly cuased by data digitization resolution.
3) ZT interpolation error for the intersected temperature range. This is sensitive to poor interpolation, maingly caused by phase transition and exponential behavior.
