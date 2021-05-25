s1=''
s2=''
s3=''
s4=''
tag='test'
with open('sfyall_%s.txt'%tag,'r') as f: line=f.read().split('\n')[:-1]

for l in line:
 if 'sfy1' in l and '3DHisTech' in l: s1+=l+'\n'
 elif ('sfy2' in l or 'sfy3' in l) and ('3DHisTech' in l): s2+=l+'\n'
 elif 'SZSQ' in l: s3+=l+'\n'
 elif 'WNLO' in l: s4+=l+'\n'

with open('s1_%s.txt'%tag,'w') as f: f.write(s1)

with open('s2_%s.txt'%tag,'w') as f: f.write(s2)

with open('s3_%s.txt'%tag,'w') as f: f.write(s3)

with open('s4_%s.txt'%tag,'w') as f: f.write(s4)
