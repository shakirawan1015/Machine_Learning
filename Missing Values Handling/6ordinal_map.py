import pandas as pd
import numpy as np

data = pd.DataFrame({
    "letters": ["a", "u", "c", "k", "g", "w", "b", "d", "e", "f", "h", "i"],
})

map_ordinal = {
    "a":0,"u":1,"c":2,"k":9,"g":6,"w":5,"b":6,"d":7,"e":2,"f":9,"h":1,"i":5
}

data["Encoded_letters"] = data["letters"].map(map_ordinal)

print("="*30,"\n  \tMap_Encoded Data\n",data)