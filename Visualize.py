from Preprocessing import ds
import plotly.express as px
import plotly.offline as pyo
fig = px.scatter(ds,x=0,y=1,color=ds.Class.astype(str))
# 0 is genuie and 1 is fraud
fig.show()
