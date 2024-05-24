import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from math import comb
import shap

mortality_auc = pd.read_csv("shapley/length_of_stay.csv")


def powerset(iterable,missing_el):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    xs = list(iterable)
    return [tuple(sorted(list(subset))) for i in range(0, len(xs) + 1) for subset in combinations(xs, i) if missing_el not in subset]



mortality_auc_types = mortality_auc[['AUC', 'viz','txt','tab','ts']].groupby(['viz','txt','tab','ts']).mean().reset_index()
#print(mortality_auc_types)
type_auc_dict = {}
colnames = list(mortality_auc_types.columns.values)
for row_count, row in mortality_auc_types.iterrows():
    set_comb = []
    for i in range(4):
        if row[i] == 1:
            set_comb.append(colnames[i])
    
    type_auc_dict[tuple(sorted(set_comb))] = row[4]
type_auc_dict[()] = 0.5
    
types = ['viz','ts', 'txt'] #'tab' 'txt'
types_shap_dict = {}
for tpe in types:
    type_shap = 0
    total_sets = powerset(types, tpe)
    for single_set in total_sets:
        single_list = list(single_set)
        single_list.append(tpe)
        type_shap += 1 / (len(types) * comb(len(types)-1, len(single_set))) * (type_auc_dict[tuple(sorted(single_list))]-type_auc_dict[single_set])
    types_shap_dict[tpe] = type_shap

shap_values = pd.DataFrame(types_shap_dict.items(), columns=['Type', 'Gain'])
print(shap_values)
#shap_values_types.to_csv("shapley/mortality_shapley.csv")
shap_vals = shap_values['Gain']
features = shap_values['Type']

total_gain = shap_values['Gain'].sum()

shap_values_sorted = shap_values.sort_values(by='Gain', ascending=True)

cumulative_sum = 0.5

SorT = 'Type'

# plt.figure(figsize=(10, 6))

# for idx, row in shap_values_sorted.iterrows():
#     plt.barh(row[SorT], row['Gain'], left=cumulative_sum, color='red' if row['Gain'] > 0 else 'blue', align="edge")
#     plt.text(cumulative_sum + row['Gain'], row[SorT], f"+{round(row['Gain'], 2)}", ha='left', va='center')
#     cumulative_sum += row['Gain']

# print(plt.barh)

# plt.axvline(x=0.5, color='black', linestyle='--', linewidth=0.5)  # Baseline
# plt.xlabel(SorT)
# plt.ylabel('Gain')
# plt.title('Shapley values')
# plt.grid(axis='x', linestyle='--', alpha=0.5)
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig('shap_fracture_no_txt.png')

pos_cum = 0.5
neg_cum = 0.5

for idx, row in shap_values_sorted.iterrows():
    if row['Gain'] > 0:
        plt.barh(row[SorT], row['Gain'], left=pos_cum, color='red' if row['Gain'] > 0 else 'royalblue', height=0.6)
        plt.text(pos_cum + row['Gain'], row[SorT], f"+{round(row['Gain'], 2)}", ha='left', va='center')
        pos_cum += row['Gain']
    if row['Gain'] < 0:
        plt.barh(row[SorT], row['Gain'], left=neg_cum, color='red' if row['Gain'] > 0 else 'royalblue', height=0.6)
        plt.text(neg_cum + row['Gain'], row[SorT], f"+{round(row['Gain'], 3)}", ha='left', va='center')
        neg_cum += row['Gain']

print(plt.barh)

plt.axvline(x=0.5, color='black', linestyle='--', linewidth=0.5)  # Baseline
plt.xlabel('Gain')
plt.ylabel('Modality')
plt.title('Shapley values')
plt.xlim(left=0.45, right=0.8)
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('shapley_imgs/length_of_stay.png')