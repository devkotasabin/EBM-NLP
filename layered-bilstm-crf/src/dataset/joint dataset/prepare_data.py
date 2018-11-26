import sys

FILENAME = sys.argv[1]
print FILENAME
OUTDIR = "joint_BIO"

# Go through the columns
# Prepare the data file
# Skip 2nd column i.e. the POS data

# Replace the normal tag with B- or I- tag
# Add a column with 'O' tag for nested LSTM implementation

prev_lbl1 = 'O'
prev_lbl2 = 'O'

label1_cpy = 'O'
label2_cpy = 'O'

with open(FILENAME) as f:
	with open(OUTDIR + '/' + FILENAME, 'w') as write_file:
		for line in f:
			if line.strip() != '':
				line = line.rstrip()
				token, pos, label1, label2 = line.split('\t')
				label1_cpy = label1
				label2_cpy = label2
				if label1 != 'O':
					if prev_lbl1 == 'O':
						label1 = 'B-' + label1
					else:
						label1 = 'I-' + label1

				if label2 != 'O':
					if prev_lbl2 == 'O':
						label2 = 'B-' + label2
					else:
						label2 = 'I-' + label2


				write_file.write('%s\t%s\t%s\tO\n' %(token, label1, label2))
			else:
				write_file.write(line)
				label1_cpy = 'O' 
				label2_cpy = 'O'

			prev_lbl1 = label1_cpy
			prev_lbl2 = label2_cpy
