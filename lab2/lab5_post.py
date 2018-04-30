import sys,os
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation
from itertools import groupby

#--------------------------------------------------------------------------------------------------------------#


def plotAnim(ts, x, us, real_sols):

		fig, ax = plt.subplots()
		fig.set_tight_layout(True)
		line1, = ax.plot(x,us[0],'b-',label='Numerical Solution')
		line2, = ax.plot(x,real_sols[0],linestyle='-.',color='r',label='Solution')
		ax.legend()

		def update(i):
			label = 't = {0}'.format(ts[i])
			line1.set_ydata(us[i])
			line2.set_ydata(real_sols[i])		
			ax.set_xlabel(label)
			ax.legend()
			if(i%10):
				plt.savefig('solution_plot_t'+str(ts[i])+'.png')
			#return line2, ax

		if __name__ == '__main__':
			# FuncAnimation will call the 'update' function for each frame; here
			# animating over 10 frames, with an interval of 200ms between frames.
			anim = FuncAnimation(fig, update, frames=np.arange(0, len(ts)), interval=200)
			if len(sys.argv) > 1 and sys.argv[1] == 'save':
				anim.save('line.gif', dpi=80, writer='imagemagick')
			else:
				# plt.show() will just loop the animation forever.
				plt.show()

#--------------------------------------------------------------------------------------------------------------#


def gen_data(nxs,k):
		"""
		Run lax and lax wendrof then read in time plots for post processing
		"""		
		# generate data
		nx_us = []
		nx_sols = []			
		real_sol = []
		real_sols = []
		cour = .5

		os.system("./prep.x")
		os.system("ls")

		for i, nx in enumerate(nxs):
			cmd_1 = "./lax %(nx)s %(cour)s > o%(nx)s" % locals()
			os.system(cmd_1)
			cmd_2 = "mv advection_data.txt lax%(nx)s.txt" %locals()
			os.system(cmd_2)

		for i, nx in enumerate(nxs):
			name1 = "lax"+str(nx)+".txt"
			x,t,u = np.loadtxt(name1,skiprows=1).T

			us = [u[i:i+nx] for i  in xrange(0,len(u),nx)]
			time = [t[i] for i in xrange(0,len(t),nx)]
			xs = [x[i:i+nx] for i in xrange(0,len(x),nx)]
			usout = us
			tout = time
			xout = xs[0]

			# create analytical solution corresponding to output
			del real_sols[:]
			vel = 1.	
			for  t in time:
				wrapped = 1
				#print "t = "+str(t)
				#print "window: [ " +  str((.25+t*vel)%1. )+", "+ str((.75+t*vel)%1. )+ "]"
				del real_sol[:]
				for j, x_val in enumerate(xs[0]):
					if(x_val<= (.75+t*vel)%1. and x_val>= (.25+t*vel)%1. ):
						real_sol.append( np.sin(2 * math.pi * ( x_val - (.25+t*vel)%1. ) )**2. + 1.)
						wrapped = 0
					else:
						real_sol.append(1.)
				if(wrapped):
					del real_sol[:]
					for j, x_val in enumerate(xs[0]):
						if(x_val<= (.75+t*vel)%1. or x_val>= (.25+t*vel)%1. ):
							real_sol.append( np.sin(2 * math.pi * ( x_val - (.25+t*vel)%1. ) )**2. + 1.)
							wrapped = 0
						else:
							real_sol.append(1.)
		
				real_sols.append(real_sol)
				rsout = real_sols
			#print real_sol
			#print x
			nx_us.append(us[-1])
			nx_sols.append(real_sols[-1])

		for i,us in enumerate(nx_us):
			print 'length of row '+str(i)+' of nx_us is '+ str(len(us))
			print 'length of row '+str(i)+' of nx_sols is '+ str(len(nx_sols[i]))
		print 'x vector has length ' + str(len(xout))
		print 'first analytical solution vector has length ' + str(len(rsout[0]))
		return usout,rsout, tout , xout, nx_us,nx_sols


				
		
#-------------------------------------------------------------------------------------------------------------#
def L1(u,sol,nx):
	l1 = 0.0
	for i,u_val in enumerate(u):
		l1 = l1 + abs(u_val-sol[i])
	l1 = l1 / nx
	return l1
#-------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------#
def L2(u,sol,nx):
	l2 = 0.0
	for i,u_val in enumerate(u):
		l2 = l2 + (u_val-sol[i])**2.
	l2 = l2 / nx
	l2 = l2**.5
	return l2
#-------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------#
def Linf(u,sol,nx):
	l2 = 0.0
	linf_error = [abs(u_val-sol[i]) for i,u_val in enumerate(u)]
	linf = max(linf_error)
	return linf
#-------------------------------------------------------------------------------------------------------------#


def benchmark_lax(nx_us,nx_sols,nxs):
		
		"""
		nx_us and nx_sols should be the final solution vector for different nxs
		"""	
		errors = []
		l2_error = []	
		l1_error = []	
		linf_error = []	

		for i, nx in enumerate(nxs):
			l2 = L2(nx_us[i],nx_sols[i],nx)
			l2_error.append(l2)
			l1 = L1(nx_us[i],nx_sols[i],nx)
			l1_error.append(l1)
			linf = Linf(nx_us[i],nx_sols[i],nx)
			linf_error.append(linf)
		errors.append(l2_error)
		errors.append(l1_error)
		errors.append(linf_error)

		x1 = [1.e-2*a for a in nxs] 
		plt.loglog(nxs, x1,linewidth=5,label='$1^{st}$ Order',linestyle='-.') 
		for i, error in enumerate(errors):
			if(i==0):
				plt.plot(nxs, errors[i],linewidth=10, label='$L_2$ Error',marker = '1')
			if(i==1):
				plt.plot(nxs, errors[i],linewidth=10, label='$L_1$ Error',marker = '2')
			if(i==2):
				plt.plot(nxs, errors[i],linewidth=10, label='$L_{\infinity}$ Error',marker = '*')

					
		roc_l2 = []
		roc_l1 = []
		roc_linf = []
		for j,error in enumerate(errors):
			if(j==0):
				for l ,l2_er in enumerate(error):
					if(l>0):
						p_l2 = np.log(error[l-1]/l2_er)/np.log(2)
						roc_l2.append(p_l2)	
			if(j==2):
				for l ,l1_er in enumerate(error):
					if(l>0):
						p_l1 = np.log(error[l-1]/l1_er)/np.log(2)
						roc_l1.append(p_l1)
			if(j==3):
				for l ,linf_er in enumerate(error):
					if(l>0):
						p_linf = np.log(error[l-1]/linf_er)/np.log(2)
						roc_linf.append(p_linf)
		print "L2 ERROR:"
		print roc_l2
		print "L1 ERROR:"
		print roc_l1
		print "L1 ERROR:"
		print roc_linf
		print 'Done Plotting Errors'

		plt.title('Order of Convergence',fontsize=60 )
		plt.xlabel('$dx$',fontsize=50)
		plt.ylabel('Error',fontsize=50)
		plt.legend(prop={'size': 50})
		plt.savefig('benchmark.png')
		plt.show()

#---------------------------------------------------------------------------------------------------------------#

matplotlib.rc('xtick',labelsize=50)
matplotlib.rc('ytick',labelsize=50)

pi = math.pi

c = .5

nxs = [8,16,32,64,128,256]

us, real_sols, time, x,nx_us,nx_sols = gen_data(nxs,1)


plotAnim(time, x, us,real_sols)

benchmark_lax(nx_us,nx_sols,nxs)


