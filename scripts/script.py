import subprocess

command_lambda_clip = "python main.py -p {REPLACE_PROMPT} -i {IMAGE_PATH} --mask {OUTPUT_DIR}/{segment_file} --output_path {OUTPUT_DIR}/out{sub} --clip_guidance_lambda {lambda_clip}"
command_lambda_lpips = "python main.py -p {REPLACE_PROMPT} -i {IMAGE_PATH} --mask {OUTPUT_DIR}/{segment_file} --output_path {OUTPUT_DIR}/out{sub} --lpips_sim_lambda {lambda_lpips}"
command_lambda2 = "python main.py -p {REPLACE_PROMPT} -i {IMAGE_PATH} --mask {OUTPUT_DIR}/{segment_file} --output_path {OUTPUT_DIR}/out{sub} --l2_sim_lambda {lambda2}"


lambda_clips =[10 ** i for i in range(7)]
lambda_lpips =[10 ** i for i in range(7)]
lambda2s =[10 ** i for i in range(7)]
# /home/cgptuser01/Desktop/rahul/blended-diffusion/ablation_clip/000000050679.jpg
image_path = './ablation_lambda2/000000050679.jpg'
replace_prompt ='apple'
output_dir = './ablation_lambda2'
segment_file = 'best_segment.png'


# Loop through powers of 10 and call the program with varied argument
for i in range(7):
    lambda2 = lambda2s[i]
    sub = '_'+ str(i)
    command = command_lambda2.format(REPLACE_PROMPT=replace_prompt, IMAGE_PATH = image_path, OUTPUT_DIR=output_dir,segment_file =segment_file, sub= sub, lambda2 = lambda2)
    # Print the command being executed
    print("Executing command:", command) 
    # Execute the command using subprocess
    subprocess.run(command, shell=True)  # Adjust shell=True based on your use case