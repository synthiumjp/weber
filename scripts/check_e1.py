import json
base = json.load(open(r'C:\weber\results\paradigm_a\llama_base\numerical\paradigm_a_analysis.json'))
inst = json.load(open(r'C:\weber\results\paradigm_a\llama_instruct\numerical\paradigm_a_analysis.json'))
print('Layer | Base Weber R2 | Inst Weber R2 | Base Weber rho | Inst Weber rho')
for layer_num in [5, 16, 20, 24, 28, 31]:
    lname = f'layer_{layer_num:02d}'
    b_r2 = base['layers'][lname]['cosine']['model_fits']['weber']['r2']
    i_r2 = inst['layers'][lname]['cosine']['model_fits']['weber']['r2']
    b_rho = base['layers'][lname]['cosine']['rsa']['weber']['rho']
    i_rho = inst['layers'][lname]['cosine']['rsa']['weber']['rho']
    print(f'  {layer_num:2d}  | {b_r2:.4f}        | {i_r2:.4f}        | {b_rho:.4f}         | {i_rho:.4f}')
