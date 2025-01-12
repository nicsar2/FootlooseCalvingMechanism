import pandas as pd

def getData():
	data = []
	data.append({"name":"DR01", "lat":-77.767, "lon":178.346%360, "V":1023.36, "U":103.61, "sV":0.92, "sU":1.57})
	data.append({"name":"DR02", "lat":-77.824, "lon":-178.425%360,  "V":1088.51, "U":157.85, "sV":5.64, "sU":2.85})
	data.append({"name":"DR03", "lat":-78.263, "lon":-175.117%360, "V":993.02, "U":222.30, "sV":2.97, "sU":1.14})
	data = pd.DataFrame.from_dict(data)
	data.set_index("name", verify_integrity=True, inplace=True)
	return data


