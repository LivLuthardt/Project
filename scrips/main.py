from data_clean import*
from ellipse import*
from tangent import*
from copula import*
from clustering import*

data = data_clean()

data_clean = tangent_angles(data)

data_sorted = sort(data_clean)
print(data_sorted)

