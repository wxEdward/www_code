f = open('output.txt','w')

a = 10
print('This is the first result:{}'.format(a),file=f)
print('Second:{}'.format(a+10),file=f)
f.close()
