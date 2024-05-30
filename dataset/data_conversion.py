from datetime import datetime

# Function to convert Unix timestamp to conventional date and time format
def convert_unix_to_date_time(unix_timestamp):
    date_time = datetime.utcfromtimestamp(unix_timestamp)
    return date_time.strftime('timestamp=%d/%m/%Y,%H:%M:%S')

# Read the input file
input_file_path = 'CollegeMsg.txt'
output_file_path = 'ProcessedCollegeMsg.txt'

with open(input_file_path, 'r') as file:
    lines = file.readlines()

# Process each line and convert the timestamp
converted_lines = []
for line in lines:
    src, tgt, unix_timestamp = line.strip().split()
    formatted_date_time = convert_unix_to_date_time(int(unix_timestamp))
    converted_lines.append(f"{src} {tgt} {formatted_date_time}\n")

# Write the converted data to a new file
with open(output_file_path, 'w') as file:
    file.writelines(converted_lines)

print(f"Converted data has been written to {output_file_path}")