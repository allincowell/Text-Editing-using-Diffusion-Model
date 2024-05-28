
import os
import tqdm

def create_result_folder():
    files = os.listdir('inputs')
    data = []

    for file in files:
        command = 'mkdir {output_folder}'

        if os.path.isfile(os.path.join('inputs', file)) and file.endswith('.jpg'):
            os.system(command.format( output_folder = 'inputs/'+file[:-4]))
            # last_line = extract_last_line('hw3_opensmile/'+out_file).split(',')
            # values = [float(x) for x in last_line[1:-1]]
            # row.extend(values)
            # data.append(row)
    
    # output_df = pd.DataFrame(data = data, columns = opensmile_attributes)
    # return output_df

create_result_folder()