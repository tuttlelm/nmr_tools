## xpk2sparky.py
## from Damien Wilburn
## USAGE: python xpk2sparky.py input.xpk output.list
## Only works for 2D HSQC-style spectra
## Only exports peak positions (intensities will differ between .nv and .uscf)

## Libraries
import os, sys

assert len(sys.argv) == 3, 'Improper number of inputs'
input_file, output_file = sys.argv[1:]
assert os.path.isfile(input_file), input_file+' does not exist'

fin = open(input_file)
xpk_lines = fin.read().splitlines()
fin.close()

fout = open(output_file,'w')
fout.write('      Assignment         w1         w2 \n\n')

header = [ x for x in xpk_lines[5].split(' ') if x != '' ]
# Determine heteronucleus
H, X = [ x.split('.')[0] for x in header if x[-2:] == '.L' ]

for xpk_line in xpk_lines[6:]:
    xpk_elems = [ x for x in xpk_line.split(' ') if x != '' ]
    elems = dict(zip(header,xpk_elems[1:]))

    # Determine proper sparky label
    if elems[H+'.L'].count('.') == 1 and elems[X+'.L'].count('.') == 1:
        header_H, nucleus_H = elems[H+'.L'][1:-1].split('.')
        header_X, nucleus_X = elems[X+'.L'][1:-1].split('.')

        if nucleus_H.upper() == 'HN': nucleus_H = 'H'
        if header_H == header_X:
            label = header_X+nucleus_X.upper()[-1]+'-'+nucleus_H.upper()[-1]
    else:
        labels = [elems[x+'.L'][1:-1] if elems[x+'.L'] != '{}' else '?' for x in [H,X] ]
        label = labels[0]+'-'+labels[1]

    fout.write( label.rjust(17) + 
                format(float(elems[X+'.P']),'.3f').rjust(11) +
                format(float(elems[H+'.P']), '.3f').rjust(11) + '\n' )
fout.close()
