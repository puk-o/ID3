import math

import numpy as np
import pandas as pd

from dataclasses import dataclass, field
from typing import Optional, Union

features = ["Shape", "Color", "Odor"]
label = "Edible"

train_set = [
    ["C", "B", 1, True],
    ["D", "B", 1, True],
    ["D", "W", 1, True],
    ["D", "W", 2, True],
    ["C", "B", 2, True],
    ["D", "B", 2, False],
    ["D", "G", 2, False],
    # ["C", "U", 2, False],
    ["C", "B", 3, False],
    ["C", "W", 3, False],
    ["D", "W", 3, False],
]

test_set = [
    ["D", "U", 1],
    ["C", "B", 2],
]

train_set = pd.DataFrame(train_set, columns=features + [label])

test_set = pd.DataFrame(test_set, columns=features)


#method counts the amount of values for each expression in a category
def count_values(df, category, crit1, crit2, crit3):
    first_count = 0
    second_count = 0
    third_count = 0
    for index, row in df.iterrows():
        if row[category] == crit1:
            first_count += 1
        elif row[category] == crit2:
            second_count += 1
        elif row[category] == crit3:
            third_count += 1

    return first_count, second_count, third_count


#method counts the true/false values for a given category and citerium
def count_values_true(df, category, crit):
    true_count = 0
    false_count = 0

    for index, row in df.iterrows():
        if row[category] == crit and row['Edible'] == True:
            true_count += 1
        elif row[category] == crit and row['Edible'] == False:
            false_count += 1

    return true_count, false_count


#method calculates the entropy with given values
def calc_entr(plus, neg, tot):
    if plus == 0:
        entr = 0 - (neg / tot) * math.log2(neg / tot)
    elif neg == 0:
        entr = -(plus/tot)*math.log2(plus/tot)- 0
    else:
        entr = -(plus/tot)*math.log2(plus/tot)-(neg/tot)*math.log2(neg/tot)
    return entr


#calculates the overall entropy with given entropy-values and weighting of the categories (2)
def calc_entr_final_2val(plus, neg, tot, v1, v2):
    entr = ((plus/tot)*v1)+((neg/tot)*v2)
    return entr


#calculates the overall entropy with given entropy-values and weighting of the categories (3)
def calc_entr_final_3val(plus, neg, num, tot, v1, v2, v3):
    entr = ((plus/tot)*v1)+((neg/tot)*v2)+((num/tot)*v3)
    return entr


#calculating the gain with the overall entropy and the given entropy of the category
def calc_infgain(total, new):
    gain = total - new
    return gain


#picking the category of the branch by checking for the highest value //not general yet
def pick_branch(shape, color, odor):
    branch = max(shape, color, odor)
    if branch == shape:
        return "Shape"
    elif branch == color:
        return "Color"
    elif branch == odor:
        return "Odor"

#-------------------FIRST BRANCH-------------------


#GesamtEntr berechnen
true_count, false_count, empty = count_values(train_set, "Edible", True, False, None)
total_entries = train_set.shape[0]
total_entr = calc_entr(true_count, false_count, total_entries)

#-----------------SHAPE-----------------#


#Anzahl der Einträge mit jeweil C und D für Shape berechnen
shaped_amount, shapec_amount, empty = count_values(train_set, "Shape", "D", "C", None)
#Anzahl der Einträge mit true/false jeweil sfür c/d berechnen
shaped_true, shaped_false = count_values_true(train_set, "Shape", "D")
entr_shaped = calc_entr(shaped_true, shaped_false, shaped_amount)
shapec_true, shapec_false = count_values_true(train_set, "Shape", "C")
entr_shapec = calc_entr(shapec_true, shapec_false, shapec_amount)

#Entropie für Shape
entr_shape = calc_entr_final_2val(shaped_amount, shapec_amount ,total_entries, entr_shaped, entr_shapec)

print(shaped_amount, shapec_amount ,total_entr, entr_shaped, entr_shapec)
print(entr_shape)

#-----------------COLOR-----------------#


#Anzahl der Einträge mit jeweil C und D für Shape berechnen
colorb_amount, colorw_amount, colorg_amount = count_values(train_set, "Color", "B", "W", "G")
#Anzahl der Einträge mit true/false jeweil sfür c/d berechnen
colorb_true, colorb_false = count_values_true(train_set, "Color", "B")
entr_colorb = calc_entr(colorb_true, colorb_false, colorb_amount)
colorw_true, colorw_false = count_values_true(train_set, "Color", "W")
entr_colorw = calc_entr(colorw_true, colorw_false, colorw_amount)
colorg_true, colorg_false = count_values_true(train_set, "Color", "G")
entr_colorg = calc_entr(colorg_true, colorg_false, colorg_amount)

#Entropie für Shape
entr_color = calc_entr_final_3val(colorg_amount, colorw_amount ,colorb_amount, total_entries, entr_colorg, entr_colorw, entr_colorb)

print(colorb_amount, colorw_amount, colorg_amount, entr_colorg, entr_colorw, entr_colorb)
print(entr_shape)

#-----------------Odor-----------------#


#Anzahl der Einträge mit jeweil C und D für Shape berechnen
odor1_amount, odor2_amount, odor3_amount = count_values(train_set, "Odor", 1, 2, 3)
#Anzahl der Einträge mit true/false jeweil sfür c/d berechnen
odor1_true, odor1_false = count_values_true(train_set, "Odor", 1)
entr_odor1 = calc_entr(odor1_true, odor1_false, odor1_amount)
odor2_true, odor2_false = count_values_true(train_set, "Odor", 2)
entr_odor2 = calc_entr(odor2_true, odor2_false, odor2_amount)
odor3_true, odor3_false = count_values_true(train_set, "Odor", 3)
entr_odor3 = calc_entr(odor3_true, odor3_false, odor3_amount)

#Entropie für Odor
entr_odor = calc_entr_final_3val(odor1_amount, odor2_amount ,odor3_amount, total_entries, entr_odor1, entr_odor2, entr_odor3)

#Information-Gain berechnen
gain_odor = calc_infgain(total_entr, entr_odor)
gain_shape = calc_infgain(total_entr, entr_shape)
gain_color = calc_infgain(total_entr, entr_color)


print(odor1_amount, odor2_amount, odor3_amount, entr_odor1, entr_odor2, entr_odor3)
print(entr_odor)
print("-------------------------------------------------------")
print("Finale Entropien:")
print("Shape:", entr_shape, " Color: ", entr_color, " Odor: ", entr_odor)
print("Information Gain: ")
print("Shape:", gain_shape, " Color: ", gain_color, " Odor: ", gain_odor)
print("-------------------------------------------------------")
#Pick highest information game for first branch
first_branch = pick_branch(gain_shape, gain_color, gain_odor)
print("Höchster Gain: ", first_branch)
#Using first_branch to block out entries from being used
"""-----------------------------------------------------------"""

#-------------------SECOND BRANCH-------------------

#Calculate GesamtEntr  of 2nd branch

true_count, false_count, empty = count_values(train_set, "Edible", True, False, "Odor")
total_entries = train_set.shape[0]
total_entr = calc_entr(true_count, false_count, total_entries)