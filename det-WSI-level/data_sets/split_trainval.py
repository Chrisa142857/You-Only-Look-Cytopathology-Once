with open('sfy1&2_all_slide_list.txt', 'r') as f:
	lines = f.read().split('\n')[:-1]
pt = 273
print(lines[pt-1])
lines = lines[:pt] + [''] + lines[pt:]
print(lines[pt])
t, v= '', ''
for i,l in enumerate(lines):
    if i % 7 == 0:
    	if l != '': v += l+'\n'
    else:
    	if l != '': t += l+'\n'
with open('sfy1&2_all_slide_list_MILtrain.txt','w') as f:
    f.write(t)

with open('sfy1&2_all_slide_list_MILval.txt','w') as f:
    f.write(v)
