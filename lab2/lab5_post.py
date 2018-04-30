import sys,os
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from itertools import groupby

#---------------------------------------------------------------------------------------------------------------#
def plotErrorTimeCourse_multi( fileNames,errorTypes,i,nx,orders):
		sims_l2 = []
		sims_l1 = []
		l2s = []
		l1s = []
		dts = []
		matplotlib.rc('xtick',labelsize=50)
		matplotlib.rc('ytick',labelsize=50)

		# i is the nx index
		for k,fileName in enumerate(fileNames):
			for order in orders:
				name1 = fileName+str(order)+'_'+ str(i) + ".out"
				with open(name1,'r') as f:
					dt = f.readline().splitlines()
					#print "dt is " + str(dt)
					f.close() 
				print np.loadtxt(name1,skiprows=1).T 
				if(k==0):
					l2s.append ( np.loadtxt(name1,skiprows=1).T)
					print len(l2s[0])
				if(k==1):
					l1s.append ( np.loadtxt(name1,skiprows=1).T)
					print len(l1s[0])
				#sims_l2.append(l2[len(l2)-1])
				#dts.append(float(dt[0]))
				time = np.linspace(0,10,len(l2s[0]))
		
			
			#debug plotting of error as function of time for each nx
			if(k==0):
				for j,order in enumerate(orders):
					plt.plot(time, l2s[j], linewidth=10,label='L2: Order '+str(order),marker='o')
			if(k==1):
				for j,order in enumerate(orders):
					plt.plot(time, l1s[j],linewidth=10, label='L1: Order '+str(order),marker='*')

		#plt.ylim([0,1])
		#plt.title('$n_x$=' + str(nx)  )
		plt.title(str(nx) + ' Points',fontsize=60 )
		plt.ylim([0,.5])
		plt.xlabel('t',fontsize=50)
		plt.ylabel('Error',fontsize=50)
		plt.legend(prop={'size': 50})

		plt.show()

#---------------------------------------------------------------------------------------------------------------#
def benchmark_multiError(filenames,errors, dxs, orders):
			
		l2_orders = []
		l1_orders = []
		dts = []
		numSim = len(nxs)
		
		#try:
		for k,filename in enumerate(filenames):
			if(k==0):
				for j,order in enumerate(orders):
					sims_l2 = []
					for i,nx in enumerate(nxs):
						name1 = filename+str(order)+"_"+ str(i) + ".out"
						with open(name1,'r') as f:
							dt = f.readline().splitlines()
							print "dt is " + str(dt)
							f.close() 
						#print np.loadtxt(name1,skiprows=1).T 
						l2 = np.loadtxt(name1,skiprows=1).T
						sims_l2.append(l2[-1])
						print "Error is "+str(l2[-1])
						#print len(l2)
					#print "sims l2 +" +str(sims_l2)
					l2_orders.append(sims_l2)	
					#plot lines for refferences 
					#nxs = np.asarray(nxs[::-1])
					if(order==1):
						x1 = [1.e-2*a for a in dxs] 
						plt.loglog(dxs, x1,linewidth=5,label='$1^{st}$ Order',linestyle='-.') 
					if(order==2):
						x2 = [1.e-2*a**2 for a in dxs]
						plt.loglog(dxs, x2,linewidth=5,label='$2^{nd}$ Order',linestyle='-.') 
					if(order==3):
						x3 = [1.e-2*a**3 for a in dxs]
						plt.loglog(dxs, x3,linewidth=5,label='$3^{rd}$ Order',linestyle='-.') 
			if(k==1):
				for j,order in enumerate(orders):
					sims_l1 = []
					for i,nx in enumerate(nxs):
						name1 = filename+str(order)+"_"+ str(i) + ".out"
						with open(name1,'r') as f:
							dt = f.readline().splitlines()
							print "dt is " + str(dt)
							f.close() 
						#print np.loadtxt(name1,skiprows=1).T 
						l1 = np.loadtxt(name1,skiprows=1).T
						sims_l1.append(l1[-1])
						#print "Error is "+str(l1[-1])
						#print len(l2)
					#print "sims l2 +" +str(sims_l2)
					l1_orders.append(sims_l1)	
					#plot lines for refferences 
					#nxs = np.asarray(nxs[::-1])

			#debug plotting of error as function of time for each nx
			for j,order in enumerate(orders):
				#print dxs
				print l2_orders[j]	
				if(k==0):
					plt.plot(dxs, l2_orders[j],linewidth=10, label='L'+str(errors[k])+' Order '+str(order),marker = '*')
				if(k==1):
					plt.plot(dxs, l1_orders[j],linewidth=10, label='L'+str(errors[k])+' Order '+str(order),marker = 'o')
				
		for j,order in enumerate(orders):
			roc_l2 = []
			roc_l1 = []
			print 'ORDER ' +str(order)		
			sims_l2 = l2_orders[j]
			sims_l1 = l1_orders[j]
			print "L2 error "+str( sims_l2)
			print "L1 error "+ str(sims_l1)
			for pp,error in enumerate(errors):
				if(pp==0):
					for l ,l2_er in enumerate(sims_l2):
						if(l>0):
							p_l2 = np.log(sims_l2[l-1]/l2_er)/np.log(2)
							roc_l2.append(p_l2)	
				if(pp==1):
					for l ,l1_er in enumerate(sims_l1):
						if(l>0):
							p_l1 = np.log(sims_l1[l-1]/l1_er)/np.log(2)
							roc_l1.append(p_l1)	

			print "L2 ERROR:"
			print roc_l2
			print "L1 ERROR:"
			print roc_l1
			print 'Done Plotting Errors'

		plt.title('Order of Convergence',fontsize=60 )
		plt.xlabel('$dx$',fontsize=50)
		plt.ylabel('Error',fontsize=50)
		plt.legend(prop={'size': 50})
		plt.savefig('benchmark.png')
		plt.show()

#---------------------------------------------------------------------------------------------------------------#
def plotAnimations(nx_index, orders, nx):
		times = []
		masses = []
		real_solutions = []
		order_sols = []
		order_reals = []
		lines = []
		

	
		#infile = open('scalar_advec'+str(1)+'_mass'+str(i)+'.out','r')
		#temp = infile.readlines()
		#for line in temp:
	#		masses.append(float(line))
		#print masses

		infile = open('scalar_advec'+str(1)+'_t'+str(nx_index)+'.out','r')
		temp = infile.readlines()
		for line in temp:
			times.append(float(line))
		#print times
			
		
		for j, order in enumerate(orders):
			
			solutions = []
		
			infile = open('scalar_advec'+str(order)+'_real'+str(nx_index)+'.out','r')
			temp = infile.readlines()
			for line in temp:
				real_solutions.append([float(x) for x in line.split()])
			#print real_solutions	
			#order_reals.append(real_solutions)

			infile = open('scalar_advec'+str(order)+'_sol'+str(nx_index)+'.out','r')
			temp = infile.readlines()
			for line in temp:
				solutions.append([float(x) for x in line.split()])

			print "order " +str(order)+ " Solution hs length " +str(len(solutions[0]))
			#print solutions	
			order_sols.append(solutions)
	

			#plot solution
			fig, ax = plt.subplots()
			fig.set_tight_layout(True)
			#ax.set_ylim((-1.1,1.1))

		x = np.linspace(0,2*3.14159,len(solutions[0]))
		line1, = ax.plot(x,order_sols[0][0],'b-',label='Order 1')
		line2, = ax.plot(x,order_sols[1][0],'r-',label='Order 2')
		line3, = ax.plot(x,order_sols[2][0],'g-',label='Order 3')
		line4, = ax.plot(x,real_solutions[0],linestyle='-.',color='r',label='Solution')

		#lines.append( ax.plot(x,solutions[0],'b-',label='Order '+str(order)))

		for j,order in enumerate(orders):
			print order_sols[j][10]

	
		def update(i):
			label = 't = {0}'.format(times[i])
			#print(label)
			# Update the line and the axes (with a new xlabel). Return a tuple of
			# "artists" that have to be redrawn for this frame.
			line4.set_ydata(real_solutions[i])
			line1.set_ydata(order_sols[0][i])		
			line2.set_ydata(order_sols[1][i])		
			line3.set_ydata(order_sols[2][i])		
				#line[j].set_ydata(solutions[i])
			ax.set_xlabel(label)
			ax.legend()
			if(i % 2000):
				plt.savefig('solution_plot_i'+str(i)+'_t'+str(times[i])+'_nx'+str(nx)+'_o'+str(order)+'.png')
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
def plotAnimations(nx_index, orders, nx):
		times = []
		masses = []
		real_solutions = []
		order_sols = []
		order_reals = []
		lines = []
		

	
		#infile = open('scalar_advec'+str(1)+'_mass'+str(i)+'.out','r')
		#temp = infile.readlines()
		#for line in temp:
	#		masses.append(float(line))
		#print masses

		infile = open('scalar_advec'+str(1)+'_t'+str(nx_index)+'.out','r')
		temp = infile.readlines()
		for line in temp:
			times.append(float(line))
		#print times
			
		
		for j, order in enumerate(orders):
			
			solutions = []
		
			infile = open('scalar_advec'+str(order)+'_real'+str(nx_index)+'.out','r')
			temp = infile.readlines()
			for line in temp:
				real_solutions.append([float(x) for x in line.split()])
			#print real_solutions	
			#order_reals.append(real_solutions)

			infile = open('scalar_advec'+str(order)+'_sol'+str(nx_index)+'.out','r')
			temp = infile.readlines()
			for line in temp:
				solutions.append([float(x) for x in line.split()])

			print "order " +str(order)+ " Solution hs length " +str(len(solutions[0]))
			#print solutions	
			order_sols.append(solutions)
	

			#plot solution
			fig, ax = plt.subplots()
			fig.set_tight_layout(True)
			#ax.set_ylim((-1.1,1.1))

		x = np.linspace(0,2*3.14159,len(solutions[0]))
		for l,order in enumerate(orders):
			if(l==0):
				line1, = ax.plot(x,order_sols[0][0],'b-',label='Order 1')
			if(l==1):
				line2, = ax.plot(x,order_sols[1][0],'c-',label='Order 2')
			if(l==2):
				line3, = ax.plot(x,order_sols[2][0],'g-',label='Order 3')
		line4, = ax.plot(x,real_solutions[0],linestyle='-.',color='r',label='Solution')

		#lines.append( ax.plot(x,solutions[0],'b-',label='Order '+str(order)))

		plt.title('$n_x=$'+str(nx))
		for j,order in enumerate(orders):
			print order_sols[j][10]

	
		def update(i):
			label = 't = {0}'.format(times[i])
			#print(label)
			# Update the line and the axes (with a new xlabel). Return a tuple of
			# "artists" that have to be redrawn for this frame.
			line4.set_ydata(real_solutions[i])
			for l,order in enumerate(orders):
				if(l==0):
					line1.set_ydata(order_sols[0][i])		
				if(l==1):
					line2.set_ydata(order_sols[1][i])		
				if(l==2):
					line3.set_ydata(order_sols[2][i])		
				#line[j].set_ydata(solutions[i])
			ax.set_xlabel(label)
			ax.legend()
			if(i%1000):
				plt.savefig('solution_plot_t'+str(times[i])+'_nx'+str(nx)+'.png')
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
			if(i%1000):
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

def gen_data(nxs):
		"""
		Run lax and lax wendrof then read in time plots for post processing
		"""		
		# generate data

		os.system("./prep.x")
		os.system("ls")
		for i, nx in enumerate(nxs):
			cmd_1 = "./lax > o%(nx)s" % locals()
			os.system(cmd_1)
			cmd_2 = "mv advection_data.txt lax%(nx)s.txt" %locals()
			os.system(cmd_2)

		for i, nx in enumerate(nxs):
			name1 = "lax"+str(nx)+".txt"
			with open(name1,'r') as f:
				line = f.readline().splitlines()
				print "line " + str(line)
				f.close() 
			l2 = np.loadtxt(name1,skiprows=1).T
			print "loadtxt: "+str(l2)
			#sims_l2.append(l2[-1])


				
		
#---------------------------------------------------------------------------------------------------------------#


pi = math.pi

c = .5
#c = .6

nxs = [100]
dxs = [(2.*pi)/a for a in nxs]

gen_data(nxs)
#orders = [3]
#We want to read in and plot 


# remove previous simulation files

matplotlib.rc('xtick',labelsize=50)
matplotlib.rc('ytick',labelsize=50)

# Plot animations
#for i,nx in enumerate(nxs):	
	#plotAnimations(i,orders,nx) # plots solution for given order, nx




#-------------------- ERROR plotting --------------------------#

# Error time course for multiple simulations
#for nx_index, nx in enumerate(nxs):
#	plotErrorTimeCourse_multi(errorFiles,errorTypes,nx_index,nxs[nx_index],orders)
	
exit(0)

#benchmark_multiError(errorFiles,errorTypes, dxs,orders)












#---------------READ IN ERRORS---------------------#
sims_l2 = []
sims_l1 = []
dts1 = []
dts2 = []
numSim = len(nxs)
exit(0)
#try:
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



