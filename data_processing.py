def read_fasta_file(file_path):
    sequences = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()

    current_sequence = ""
    current_header = ""
    for line in lines:
        line = line.strip()
        if line.startswith('>'):
            if current_sequence:
                sequences[current_header] = current_sequence
            current_header = line[1:]
            current_sequence = ""
        else:
            current_sequence += line

    # Add the last sequence
    if current_sequence:
        sequences[current_header] = current_sequence

    return sequences

def extract_center_sequences(sequences, target_length):
    extracted_sequences = {}
    center_offset = (len(sequences[list(sequences.keys())[0]]) - target_length) // 2

    for header, sequence in sequences.items():
        center_sequence = sequence[center_offset:center_offset+target_length]
        extracted_sequences[header] = center_sequence

    return extracted_sequences

def write_fasta_file(file_path, sequences):
    with open(file_path, 'w') as file:
        for header, sequence in sequences.items():
            file.write(f'>{header}\n')
            file.write(sequence + '\n')


if __name__ == '__main__':

    input_file = r'dataset\G3PO\donor.fasta'
    output_lengths = [20, 40, 60, 100, 200, 300, 400]

    # Read the input fasta file
    sequences = read_fasta_file(input_file)

    # Process and generate output fasta files for each target length
    for length in output_lengths:
        extracted_sequences = extract_center_sequences(sequences, length)
        # output_file = f'sequences_acceptor_{length}.fasta'
        output_file = f'donor_{length}.fasta'
        write_fasta_file(output_file, extracted_sequences)

