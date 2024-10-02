import json
import os

def check_in_answers(output,answers):
    for ans in answers:
        if ans.lower() in output.lower():
            return True
    return False

def eval(file_path):
    path = file_path
    files = os.listdir(path)
    files = [i for i in files if "json" in i]
    dics = []
    for file in files:
        data = json.load(open(os.path.join(path,file),'r'))
        dics.extend(data)

    cnt = 0
    for i in dics:
        if check_in_answers(i['model_output'],i['answers']):
            cnt += 1
    print(cnt/len(dics),len(dics))
    
for index in [1, 3, 5, 7, 10]:
    file_path = f"./mdqa_output/ablation_llama2_trained_40/output_folder_{index}"
    eval(file_path)