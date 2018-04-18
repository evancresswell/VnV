import sys,os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def plotErrorTimeCourse( fileName,i,nx,orders,errorType ):
		sims_l2 = []
		l2s = []
		dts = []

		# i is the nx index
		for j,order in enumerate(orders):
			name1 = fileName+str(order)+'_'+ str(i) + ".out"
			with open(name1,'r') as f:
				dt = f.readline().splitlines()
				#print "dt is " + str(dt)
				f.close() 
			print np.loadtxt(name1,skiprows=1).T 
			l2s.append ( np.loadtxt(name1,skiprows=1).T)
			#sims_l2.append(l2[len(l2)-1])
			#dts.append(float(dt[0]))
			print len(l2s[0])
			time = np.linspace(0,10,len(l2s[0]))

			#debug plotting of error as function of time for each nx
			plt.plot(time, l2s[j], label=errorType+' Order '+str(order))

		#plt.ylim([0,1])
		plt.title(str(nx) + ' Points' )
		plt.xlabel('t')
		plt.ylabel('Error')
		plt.legend()
		plt.show()

def benchmark(filename, dxs, orders):

		sims_l2 = []
		dts = []
		numSim = len(nxs)

		#try:
		error = 2
		for order in orders:
			for i,nx in enumerate(nxs):
				name1 = filename+"_"+ str(i) + ".out"
				with open(name1,'r') as f:
					dt = f.readline().splitlines()
					print "dt is " + str(dt)
					f.close() 
				#print np.loadtxt(name1,skiprows=1).T 
				l2 = np.loadtxt(name1,skiprows=1).T
				sims_l2.append(l2[-1])
				print "Error is "+str(l2[-1])
				#print len(l2)
	
			#plot lines for refferences 
			#nxs = np.asarray(nxs[::-1])
			if(order==1):
				x1 = [1.e-2*a for a in dxs] 
				plt.loglog(dxs, x1,label='$1^{st}$ Order') 
			if(order==2):
				x2 = [1.e-2*a**2 for a in dxs]
				plt.loglog(dxs, x2,label='$2^{nd}$ Order') 
			if(order==3):
				x3 = [1.e-2*a**3 for a in dxs]
				plt.loglog(dxs, x3,label='$3^{rd}$ Order') 
	
			plt.loglog(dxs, sims_l2, label='L'+str(error)+' Order '+str(order),  marker='o')
		plt.legend()
		plt.savefig('benchmark.png')
		plt.show()


		#print "algo = %s %s" % (algorithm, algorithm)
		#plt.loglog(dts[1:], sims_l2[1:], label=algorithm, marker='o')
		
		#np.save("%s_norms.npy" % algorithm, np.asarray(sims_l2))
		#np.save("%s_dts.npy" % algorithm, np.asarray(dts))
		#print "%s norms are: " % algorithm, sims_l2
		#print "%s dts are: " % algorithm, dts
		#except:
		#print "EXCEPTION" 
		#pass


#---------------------------------------------------------------------------------------------------------------#
def plotAnim(i, order, nx):
		times = []
		masses = []
		solutions = []
		real_solutions = []


		infile = open('scalar_advec'+str(order)+'_sol'+str(i)+'.out','r')
		temp = infile.readlines()
		for line in temp:
			solutions.append([float(x) for x in line.split()])
		#print solutions	

		infile = open('scalar_advec'+str(order)+'_real'+str(i)+'.out','r')
		temp = infile.readlines()
		for line in temp:
			real_solutions.append([float(x) for x in line.split()])
		#print real_solutions	


		infile = open('scalar_advec'+str(order)+'_mass'+str(i)+'.out','r')
		temp = infile.readlines()
		for line in temp:
			masses.append(float(line))
		#print masses

		infile = open('scalar_advec'+str(order)+'_t'+str(i)+'.out','r')
		temp = infile.readlines()
		for line in temp:
			times.append(float(line))
		#print times


		#plot solution
		fig, ax = plt.subplots()
		fig.set_tight_layout(True)

		x = np.linspace(0,2*3.14159,len(solutions[0]))
		#ax.set_ylim((-1.1,1.1))
		line1, = ax.plot(x,solutions[0],'b-',label='Numerical Solution')
		line2, = ax.plot(x,real_solutions[0],linestyle='-.',color='r',label='Solution')

		def update(i):
			label = 't = {0}'.format(times[i])
			#print(label)
			# Update the line and the axes (with a new xlabel). Return a tuple of
			# "artists" that have to be redrawn for this frame.
			line2.set_ydata(real_solutions[i])
			line1.set_ydata(solutions[i])
			ax.set_xlabel(label)
			ax.legend()
			if(i%500):
				plt.savefig('solution_plot_t'+str(times[i])+'_nx'+str(nx)+'_o'+str(order)+'.png')
			return line, ax

		if __name__ == '__main__':
			# FuncAnimation will call the 'update' function for each frame; here
			# animating over 10 frames, with an interval of 200ms between frames.
			anim = FuncAnimation(fig, update, frames=np.arange(0, len(solutions)), interval=200)
			if len(sys.argv) > 1 and sys.argv[1] == 'save':
				anim.save('line.gif', dpi=80, writer='imagemagick')
			else:
				# plt.show() will just loop the animation forever.
				plt.show()

#---------------------------------------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------------------------------------#
def plotAnim_square(i,order,nx):
		times = []
		masses = []
		solutions = []
		real_solutions = []

		infile = open('scalar_advec'+str(order)+'_sol'+str(i)+'.out','r')
		temp = infile.readlines()
		for line in temp:
			solutions.append([float(x) for x in line.split()])
		#print solutions	

		infile = open('scalar_advec'+str(order)+'_real'+str(i)+'.out','r')
		temp = infile.readlines()
		for line in temp:
			real_solutions.append([float(x) for x in line.split()])
		#print real_solutions	


		infile = open('scalar_advec'+str(order)+'_mass'+str(i)+'.out','r')
		temp = infile.readlines()
		for line in temp:
			masses.append(float(line))
		#print masses

		infile = open('scalar_advec'+str(order)+'_t'+str(i)+'.out','r')
		temp = infile.readlines()
		for line in temp:
			times.append(float(line))
		#print times


		#plot solution
		fig, ax = plt.subplots()
		fig.set_tight_layout(True)

		x = np.linspace(0,2*3.14159,len(solutions[0]))
		ax.set_ylim((-.5,1.15))
		line1, = ax.plot(x,solutions[0],'b-')
		#line2, = ax.plot(x,real_solutions[0],'r-')

		def update(i):
			label = 't = {0}'.format(times[i])
			#print(label)
			# Update the line and the axes (with a new xlabel). Return a tuple of
			# "artists" that have to be redrawn for this frame.
			#line2.set_ydata(real_solutions[i])
			line1.set_ydata(solutions[i])
			#line.set_ydata(x)
			#line2.set_ydata(massesp[i])
			ax.set_xlabel(label)
			if(i%100):
				plt.savefig('solSquare_plot_t'+str(times[i])+'_nx'+str(nx)+'_o'+str(order)+'.png')
			return line, ax

		if __name__ == '__main__':
			# FuncAnimation will call the 'update' function for each frame; here
			# animating over 10 frames, with an interval of 200ms between frames.
			anim = FuncAnimation(fig, update, frames=np.arange(0, len(solutions)), interval=200)
			if len(sys.argv) > 1 and sys.argv[1] == 'save':
				anim.save('line.gif', dpi=80, writer='imagemagick')
			else:
				# plt.show() will just loop the animation forever.
				plt.show()

#---------------------------------------------------------------------------------------------------------------#


#---------------------------------------------------------------------------------------------------------------#


pi = math.pi

c = .5

orders = [1,2,3]
nxs = [10,20,40,80,160,320]
nxs = [16,32,64,128]
dxs = [(2.*pi)/a for a in nxs]
orders = [1,2,3]
#orders = [3]
#We want to read in and plot 

# remove previous simulation files
os.system("./prep.x")
os.system("ls")
for order in orders:
	errorType1 = 'L2'
	errorType2 = 'L1'
	fileName_error1 = "l2_error"+str(order)
	fileName_error2 = "l1_error"+str(order)
	print order
	for i, nx in enumerate(nxs):
		print nx
		cmd_1 = "./scalar_advec %(nx)s %(c)s %(order)s > o%(i)s" % locals()
		os.system(cmd_1)
		cmd_2 = "mv l2_error%(order)s.out l2_error%(order)s_%(i)s.out" % locals()
		os.system(cmd_2)
		cmd_3 = "mv scalar_advec%(order)s_mass.out scalar_advec%(order)s_mass%(i)s.out" % locals()
		os.system(cmd_3)
		cmd_4 = "mv scalar_advec%(order)s_sol.out scalar_advec%(order)s_sol%(i)s.out" % locals()
		os.system(cmd_4)
		cmd_5 = "mv scalar_advec%(order)s_t.out scalar_advec%(order)s_t%(i)s.out" % locals()
		os.system(cmd_5)
		cmd_6 = "mv l1_error%(order)s.out l1_error%(order)s_%(i)s.out" % locals()
		os.system(cmd_6)
		cmd_7 = "mv scalar_advec%(order)s_real.out scalar_advec%(order)s_real%(i)s.out" % locals()
		os.system(cmd_7)
		#plotAnim(i,order,nx) #old plotting for animation and benchmarking
		#plotAnim_square(i,order,nx) #old plotting for animation and benchmarking
	
for nx_index, nx in enumerate(nxs):
	errorFile1 = "l2_error"
	errorFile2 = "l1_error"
	plotErrorTimeCourse(errorFile1,nx_index,nxs[nx_index],orders,errorType1)
	plotErrorTimeCourse(errorFile2,nx_index,nxs[nx_index],orders,errorType2)
	
benchmark(fileName_error1, dxs,orders)

#---------------READ IN ERRORS---------------------#
sims_l2 = []
sims_l1 = []
dts1 = []
dts2 = []
numSim = len(nxs)

#try:
for i,nx in enumerate(nxs):
	name1 = fileName_error1+"_"+ str(i) + ".out"
	with open(name1,'r') as f:
		dt = f.readline().splitlines()
		print "dt is " + str(dt)
		f.close() 
	#print np.loadtxt(name1,skiprows=1).T 
	l2 = np.loadtxt(name1,skiprows=1).T
	sims_l2.append(l2[-1])
	print "l2 Error is "+str(l2[-1])
	#print len(l2)




for i,nx in enumerate(nxs):
	name2 = fileName_error2+'_'+ str(i) + ".out"
	with open(name2,'r') as f:
		dt = f.readline().splitlines()
		#print "dt is " + str(dt)
		f.close() 
	#print np.loadtxt(name1,skiprows=1).T 
	l1 = np.loadtxt(name1,skiprows=1).T
	sims_l1.append(l1[-1])
	#print "l1 Error is "+str(l1[-1])
	#print len(l2)


roc_l2 = []
roc_l1 = []
for i,l2_er in enumerate(sims_l2):
	if(i>0):
		p_l2 = np.log(sims_l2[i-1]/l2_er)/np.log(2)
		roc_l2.append(p_l2)	
		p_l1 = np.log(sims_l1[i-1]/sims_l1[i])/np.log(2)
		roc_l1.append(p_l1)	

print "L2 ERROR:"
print roc_l2
print "L1 ERROR:"
print roc_l1



