import chardet

input_file = 'dataset/ITKTTS-IDN/transkrip.txt'
output_file = 'dataset/ITKTTS-IDN/new_transkrip.txt'

with open(input_file, 'rb') as f:
    raw_data = f.read()

detected = chardet.detect(raw_data)
encoding = detected['encoding'] if detected['encoding'] else 'utf-8'  

with open(input_file, 'r', encoding=encoding) as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        parts = line.strip().split(':', 1)
        if len(parts) == 2:
            nomor, teks = parts
            nomor_padded = nomor.strip().zfill(4) 
            new_line = f"dataset/ITKTTS-IDN/utterance/ITKTTS001-{nomor_padded}.wav|{teks.strip()}\n"
            outfile.write(new_line)

print("File successfully converted!")
