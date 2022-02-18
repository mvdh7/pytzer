import PyCO2SYS as pyco2

result = pyco2.sys(
    # par1=df.alkalinity.to_numpy(),
    par1=2300,
    par2=2150,
    par1_type=1,  # 1 means alkalinity
    par2_type=2,  # 2 means DIC
    salinity=30,
    temperature=12.5,
)

pCO2 = result["pCO2"]


result_pH = pyco2.sys(
    par1=2300,
    par2=8.1,
    par1_type=1,
    par2_type=3,  # 3 means pH
    temperature=25,  # lab
    temperature_out=6,  # in the ocean
)

pCO2_in_the_ocean = result_pH["pCO2_out"]
pH_in_the_ocean = result_pH["pH_out"]

# https://pyco2sys.readthedocs.io/en/latest/
