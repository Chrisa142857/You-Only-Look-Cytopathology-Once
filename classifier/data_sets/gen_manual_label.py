with open('manualcheck_label_origindata.txt', 'r') as f:
 lines=f.read().split('\n')

ls = [l for l in lines if len(l.split(','))==3]
ls = [[l.split(',')[0],l.split(',')[-1]] for l in ls]

with open('manualcheck_121slide_list.txt','r') as f:
 slides=f.read().split('\n')[:-1]

new_l = ''
for l,label in ls:
 l=l.replace('.sdpc','')
 s=[l for s in slides if l in s]
 if len(s)!=1: print('Error', l, s)
 s=s[0]
 new_l+=label+'\n'

with open('manualcheck_121slide_label.txt', 'w') as f:
 f.write(new_l)
#################################################################
with open('manualcheck_121slide_list.txt', 'r') as f:
 lines=f.read().split('\n')[:-1]

with open('sfy_all_slide_list_train.txt', 'r') as f:
 oalls=f.read().split('\n')[:-1]

# with open('sfy_all_slide_list_val.txt', 'r') as f:
#  voalls=f.read().split('\n')[:-1]

# oalls = toalls+voalls
ls = [l.split('\\')[-1] for l in lines]
alls = [a.split('\\')[-1] for a in oalls]

new = ''
for a, oa in zip(alls, oalls):
 if a not in ls:
  new+=oa+'\n'

with open('sfy_all_slide_train_formanualcheck.txt', 'w') as f:
 f.write(new)


#####################################

