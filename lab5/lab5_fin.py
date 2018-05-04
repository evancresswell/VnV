import sys,os
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation
from itertools import groupby

#--------------------------------------------------------------------------------------------------------------#


def plotAnim(ts, x, us1, us2, real_solutions):
		#plot solution
		fig, ax = plt.subplots()
		fig.set_tight_layout(True)




		line1, = ax.plot(x,us1[0],'b-',linewidth=5,label='Lax-Frd')
		line2, = ax.plot(x,us2[0],'g-',linewidth=5,label='Lax-Wen')
		line3, = ax.plot(x,real_solutions[0],linewidth=6,linestyle='-.',color='r',label='Analytical Solution')

		plt.title('Solution Plot',fontsize=60 )
		plt.xlabel('$t$',fontsize=60 )
		plt.ylabel('$u$',fontsize=60 )

	
		def update(i):
			label = 't = {0}'.format(ts[i])
			#print real_solutions[i]
			line3.set_ydata(real_solutions[i])
			line1.set_ydata(us1[i])		
			line2.set_ydata(us2[i])		
			ax.set_xlabel(label)
			ax.legend(fontsize = 50)
			if(i%10):
				plt.savefig('solution_plot_t'+str(ts[i])+'.png')
			return line1, ax

		if __name__ == '__main__':
			# FuncAnimation will call the 'update' function for each frame; here
			# animating over 10 frames, with an interval of 200ms between frames.
			anim = FuncAnimation(fig, update, frames=np.arange(0, len(us1)), interval=200)
			if len(sys.argv) > 1 and sys.argv[1] == 'save':
				anim.save('line.gif', dpi=80, writer='imagemagick')
			else:
				# plt.show() will just loop the animation forever.
				plt.show()

#--------------------------------------------------------------------------------------------------------------#


def gen_data_square(nxs,k):
		"""
		Run lax and lax wendrof then read in time plots for post processing
		"""		


		# ---------------Generate Data For Lax---------------------------------------#
		nx_us1 = []
		nx_sols = []			
		cour = .5

		os.system("./prep.x")
		#os.system("ls")

		for i, nx in enumerate(nxs):
			cmd_1 = "./lax %(nx)s %(cour)s > o%(nx)s" % locals()
			os.system(cmd_1)
			cmd_2 = "mv advection_data.txt lax%(nx)s.txt" %locals()
			os.system(cmd_2)

		for i, nx in enumerate(nxs):
			name1 = "lax"+str(nx)+".txt"
			x,t,u = np.loadtxt(name1,skiprows=1).T

			us1 = [u[i:i+nx] for i  in xrange(0,len(u),nx)]
			time = [t[i] for i in xrange(0,len(t),nx)]
			xs = [x[i:i+nx] for i in xrange(0,len(x),nx)]
			us1out = us1
			tout = time
			t_lax = time
			xout = xs[0]

			# create analytical solution corresponding to output
			real_sols = []
			vel = 1.	
			#print 'time: '+str(time)

			for  t in time:
				wrapped = 1
				#print "t = "+str(t)
				#print "window: [ " +  str((.25+t*vel)%1. )+", "+ str((.75+t*vel)%1. )+ "]"
				real_sol = []
				for j, x_val in enumerate(xs[0]):
					if(x_val<= (.75+t*vel)%1. and x_val>= (.25+t*vel)%1. ):
						real_sol.append( 2.)
						wrapped = 0
					else:
						real_sol.append(1.)
				if(wrapped):
					del real_sol[:]
					for j, x_val in enumerate(xs[0]):
						if(x_val<= (.75+t*vel)%1. or x_val>= (.25+t*vel)%1. ):
							real_sol.append( 2.)
							wrapped = 0
						else:
							real_sol.append(1.)
				#print real_sol
				real_sols.append(real_sol)
			rsout = real_sols
			nx_us1.append(us1[-1])
			nx_sols.append(real_sols[-1])

		#for i,us in enumerate(nx_us1):
		#	print 'length of row '+str(i)+' of nx_us1 is '+ str(len(us))
		#	print 'length of row '+str(i)+' of nx_sols is '+ str(len(nx_sols[i]))
		#print 'x vector has length ' + str(len(xout))
		#print 'first analytical solution vector has length ' + str(len(rsout[0]))
		# ---------------------------------------------------------------------------#

		# ---------------Generate Data For Lax---------------------------------------#
		# only need to generate the numerical solutions, time, x and real sol should be good
		nx_us2 = []
		cour = .5

		os.system("./prep.x")
		#os.system("ls")

		for i, nx in enumerate(nxs):
			cmd_1 = "./lax_wen %(nx)s %(cour)s > owen%(nx)s" % locals()
			os.system(cmd_1)
			cmd_2 = "mv advection_data.txt lax_wen%(nx)s.txt" %locals()
			os.system(cmd_2)

		for i, nx in enumerate(nxs):
			name1 = "lax_wen"+str(nx)+".txt"
			x,t,u = np.loadtxt(name1,skiprows=1).T

			us2 = [u[i:i+nx] for i  in xrange(0,len(u),nx)]
			t_wen = [t[i] for i in xrange(0,len(t),nx)]
			xs = [x[i:i+nx] for i in xrange(0,len(x),nx)]
			x_wen = xs[0]

			us2out = us2

			# create analytical solution corresponding to output
			nx_us2.append(us2[-1])
		# ---------------------------------------------------------------------------#
		print len(us2out)
		print len(us1out)
		print len(nx_us1)
		print len(nx_us2)
		print "Time for lax: " + str(t_lax)
		print "Time for wen: " + str(t_wen)
		print "x for lax: " + str(xout)
		print "x for wen: " + str(x_wen)


		return  tout, xout, us1out, us2out, rsout, nx_us1, nx_us2, nx_sols



#--------------------------------------------------------------------------------------------------------------#


#--------------------------------------------------------------------------------------------------------------#


def gen_data_sin(nxs,k,cour):
		"""
		Run lax and lax wendrof then read in time plots for post processing
		"""		


		# ---------------Generate Data For Lax---------------------------------------#
		nx_us1 = []
		nx_sols = []			

		os.system("./prep.x")
		#os.system("ls")

		for i, nx in enumerate(nxs):
			cmd_1 = "./lax %(nx)s %(cour)s > o%(nx)s" % locals()
			os.system(cmd_1)
			cmd_2 = "mv advection_data.txt lax%(nx)s.txt" %locals()
			os.system(cmd_2)

		for i, nx in enumerate(nxs):
			name1 = "lax"+str(nx)+".txt"
			x,t,u = np.loadtxt(name1,skiprows=1).T

			us1 = [u[i:i+nx] for i  in xrange(0,len(u),nx)]
			time = [t[i] for i in xrange(0,len(t),nx)]
			xs = [x[i:i+nx] for i in xrange(0,len(x),nx)]
			us1out = us1
			tout = time
			t_lax = time
			xout = xs[0]

			# create analytical solution corresponding to output
			real_sols = []
			vel = 1.	
			#print 'time: '+str(time)

			for  t in time:
				wrapped = 1
				#print "t = "+str(t)
				#print "window: [ " +  str((.25+t*vel)%1. )+", "+ str((.75+t*vel)%1. )+ "]"
				real_sol = []
				for j, x_val in enumerate(xs[0]):
					real_sol.append( np.sin(2 * math.pi *  (x_val -vel*t) ) + 1.)
				real_sols.append(real_sol)
			rsout = real_sols
			nx_us1.append(us1[-1])
			nx_sols.append(real_sols[-1])

		#for i,us in enumerate(nx_us1):
		#	print 'length of row '+str(i)+' of nx_us1 is '+ str(len(us))
		#	print 'length of row '+str(i)+' of nx_sols is '+ str(len(nx_sols[i]))
		#print 'x vector has length ' + str(len(xout))
		#print 'first analytical solution vector has length ' + str(len(rsout[0]))
		# ---------------------------------------------------------------------------#

		# ---------------Generate Data For Lax---------------------------------------#
		# only need to generate the numerical solutions, time, x and real sol should be good
		nx_us2 = []
		cour = .5

		os.system("./prep.x")
		#os.system("ls")

		for i, nx in enumerate(nxs):
			cmd_1 = "./lax_wen %(nx)s %(cour)s > owen%(nx)s" % locals()
			os.system(cmd_1)
			cmd_2 = "mv advection_data.txt lax_wen%(nx)s.txt" %locals()
			os.system(cmd_2)

		for i, nx in enumerate(nxs):
			name1 = "lax_wen"+str(nx)+".txt"
			x,t,u = np.loadtxt(name1,skiprows=1).T

			us2 = [u[i:i+nx] for i  in xrange(0,len(u),nx)]
			t_wen = [t[i] for i in xrange(0,len(t),nx)]
			xs = [x[i:i+nx] for i in xrange(0,len(x),nx)]
			x_wen = xs[0]

			us2out = us2

			# create analytical solution corresponding to output
			nx_us2.append(us2[-1])
		# ---------------------------------------------------------------------------#
		print len(us2out)
		print len(us1out)
		print len(nx_us1)
		print len(nx_us2)
		print "Time for lax: " + str(t_lax)
		print "Time for wen: " + str(t_wen)
		print "x for lax: " + str(xout)
		print "x for wen: " + str(x_wen)


		return  tout, xout, us1out, us2out, rsout, nx_us1, nx_us2, nx_sols



#--------------------------------------------------------------------------------------------------------------#


def gen_data(nxs,k,cour):
		"""
		Run lax and lax wendrof then read in time plots for post processing
		"""		


		# ---------------Generate Data For Lax---------------------------------------#
		nx_us1 = []
		nx_sols = []			
		xout = []

		os.system("./prep.x")
		#os.system("ls")

		for i, nx in enumerate(nxs):
			cmd_1 = "./lax %(nx)s %(cour)s > o%(nx)s" % locals()
			os.system(cmd_1)
			cmd_2 = "mv advection_data.txt lax%(nx)s.txt" %locals()
			os.system(cmd_2)

		for i, nx in enumerate(nxs):
			name1 = "lax"+str(nx)+".txt"
			x,t,u = np.loadtxt(name1,skiprows=1).T
	
			us1 = [u[i:i+nx] for i  in xrange(0,len(u),nx)]
			time = [t[i] for i in xrange(0,len(t),nx)]
			xs = [x[i:i+nx] for i in xrange(0,len(x),nx)]
			us1out = us1
			tout = time
			t_lax = time
			xout.append( xs[0])


			# create analytical solution corresponding to output
			real_sols = []
			vel = 1.	
			#print 'time: '+str(time)

			"""
			for  t in time:
				wrapped = 1
				#print "t = "+str(t)
				#print "window: [ " +  str((.25+t*vel)%1. )+", "+ str((.75+t*vel)%1. )+ "]"
				real_sol = []
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
				#print real_sol
				real_sols.append(real_sol)
			rsout = real_sols
			#print us1[-1]
			nx_sols.append(real_sols[-1])
			"""
			nx_us1.append(us1[-1])

		#for i,us in enumerate(nx_us1):
		#	print 'length of row '+str(i)+' of nx_us1 is '+ str(len(us))
		#	print 'length of row '+str(i)+' of nx_sols is '+ str(len(nx_sols[i]))
		#print 'x vector has length ' + str(len(xout))
		#print 'first analytical solution vector has length ' + str(len(rsout[0]))
		# ---------------------------------------------------------------------------#

		# ---------------Generate Data For Lax---------------------------------------#
		# only need to generate the numerical solutions, time, x and real sol should be good
		nx_us2 = []
		real_sols = []
		xreal = []
		cour = .5

		os.system("./prep.x")
		#os.system("ls")

		for i, nx in enumerate(nxs):
			cmd_1 = "./lax_wen %(nx)s %(cour)s > owen%(nx)s" % locals()
			os.system(cmd_1)
			cmd_2 = "mv advection_data.txt lax_wen%(nx)s.txt" %locals()
			os.system(cmd_2)
			cmd_4 = "mv real_sol_data.txt lax_real_sol%(nx)s.txt" %locals()
			os.system(cmd_4)
			name2 = "lax_real_sol"+str(nx)+".txt"
			x_real,t_real,u_real = np.loadtxt(name2,skiprows=1).T



			real_sols = [u_real[i:i+nx] for i  in xrange(0,len(u_real),nx)]
			time_real = [t_real[i] for i in xrange(0,len(t_real),nx)]
			xs_real = [x_real[i:i+nx] for i in xrange(0,len(x_real),nx)]
			xreal.append( xs_real[0])

			nx_sols.append(real_sols[-1])
		

			print 'Differences in time and space between cpp real x and t and numerical x and t' 
			print L2(xs_real[0],xs[0],nx)
			print




		for i, nx in enumerate(nxs):
			name1 = "lax_wen"+str(nx)+".txt"
			x,t,u = np.loadtxt(name1,skiprows=1).T

			us2 = [u[i:i+nx] for i  in xrange(0,len(u),nx)]
			t_wen = [t[i] for i in xrange(0,len(t),nx)]
			xs = [x[i:i+nx] for i in xrange(0,len(x),nx)]
			x_wen = xs[0]

			us2out = us2

			# create analytical solution corresponding to output
			nx_us2.append(us2[-1])
		# ---------------------------------------------------------------------------#
		print len(us2out)
		print len(us1out)
		print len(nx_us1)
		print len(nx_us2)
		#print "Time for lax: " + str(t_lax)
		#print "Time for wen: " + str(t_wen)
		#print "x for lax: " + str(xout)
		#print "x for wen: " + str(x_wen)

		return  tout, xout, us1out, us2out, real_sols, nx_us1, nx_us2, nx_sols


				
		
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


def benchmark_lax(nx_us_lax,nx_us_wen,nx_sols,nxs):
		
		"""
		nx_us and nx_sols should be the final solution vector for different nxs
		"""	
		"""
		# plot arrays which will be compared..
		print len(nx_us1[0])
		for i,nx_1 in enumerate(nx_us_lax):
			plt.plot(nx_1)
			plt.plot(nx_sols[i], '--')
			plt.show()
		for i,nx_1 in enumerate(nx_us_wen):
			plt.plot(nx_1)
			plt.plot(nx_sols[i], '--')
			plt.show()
		"""


		# ---------------Generate Errors For Lax---------------------------------------#
		errors_lax = []
		l2_error_lax = []	
		l1_error_lax = []	
		linf_error_lax = []	

		for i, nx in enumerate(nxs):
			l2 = L2(nx_us_lax[i],nx_sols[i],nx)
			l2_error_lax.append(l2)
			l1 = L1(nx_us_lax[i],nx_sols[i],nx)
			l1_error_lax.append(l1)
			linf = Linf(nx_us_lax[i],nx_sols[i],nx)
			linf_error_lax.append(linf)
		errors_lax.append(l2_error_lax)
		errors_lax.append(l1_error_lax)
		errors_lax.append(linf_error_lax)

		for i, error in enumerate(errors_lax):
			if(i==0):
				plt.plot(nxs, error,linewidth=2.5, label='$L_2$ Lax-Frd',marker = '1')
			if(i==1):
				plt.plot(nxs, error,linewidth=2.5, label='$L_1$ Lax-Frd',marker = '2')
			if(i==2):
				plt.plot(nxs, error,linewidth=2.5, label='$L_{\infty}$ Lax_Frd',marker = '*')

					
		roc_l2_lax = []
		roc_l1_lax = []
		roc_linf_lax = []

		x1 = [1.e2*a**-1 for a in nxs] 
		plt.loglog(nxs, x1,linewidth=2,label='$1{st}$ Order',linestyle='-.') 

		for j,error in enumerate(errors_lax):
			if(j==0):
				for l ,l2_er in enumerate(error):
					if(l>0):
						p_l2 = np.log(error[l-1]/l2_er)/np.log(2)
						roc_l2_lax.append(p_l2)	
			if(j==1):
				for l ,l1_er in enumerate(error):
					if(l>0):
						p_l1 = np.log(error[l-1]/l1_er)/np.log(2)
						roc_l1_lax.append(p_l1)
			if(j==2):
				for l ,linf_er in enumerate(error):
					if(l>0):
						p_linf = np.log(error[l-1]/linf_er)/np.log(2)
						roc_linf_lax.append(p_linf)

		print
		print "L2 ERROR_lax:"
		print errors_lax[0]
		print "L1 ERROR_lax:"
		print errors_lax[1]
		print "L1 ERROR_lax:"
		print errors_lax[2]
		print 

		print "ROC L2"
		print roc_l2_lax
		print "ROC L1"
		print roc_l1_lax
		print "ROC Linf"
		print roc_linf_lax
		print 
		print
		#------------------------------------------------------------------------------#

		# ---------------Generate Errors For Lax-Wendroff----------------------------------#
		errors_wen = []
		l2_error_wen = []	
		l1_error_wen = []	
		linf_error_wen = []	

		for i, nx in enumerate(nxs):
			l2 = L2(nx_us_wen[i],nx_sols[i],nx)
			l2_error_wen.append(l2)
			l1 = L1(nx_us_wen[i],nx_sols[i],nx)
			l1_error_wen.append(l1)
			linf = Linf(nx_us_wen[i],nx_sols[i],nx)
			linf_error_wen.append(linf)
		errors_wen.append(l2_error_wen)
		errors_wen.append(l1_error_wen)
		errors_wen.append(linf_error_wen)

		x2 = [1.e2*a**-2 for a in nxs] 
		plt.loglog(nxs, x2,linewidth=2,label='$2{nd}$ Order',linestyle='-.') 
		for i, error in enumerate(errors_wen):
			if(i==0):
				plt.plot(nxs, error,linewidth=2.5, linestyle='--', label='$L_2$ Lax-Wen',marker = '1')
			if(i==1):
				plt.plot(nxs, error,linewidth=2.5, linestyle='--', label='$L_1$ Lax-Wen',marker = '2')
			if(i==2):
				plt.plot(nxs, error,linewidth=2.5, linestyle='--', label='$L_{\infy}$ Lax-Wen',marker = '*')

					
		roc_l2_wen = []
		roc_l1_wen = []
		roc_linf_wen = []
		for j,error in enumerate(errors_wen):
			if(j==0):
				for l ,l2_er in enumerate(error):
					if(l>0):
						p_l2 = np.log(error[l-1]/l2_er)/np.log(2)
						roc_l2_wen.append(p_l2)	
			if(j==1):
				for l ,l1_er in enumerate(error):
					if(l>0):
						p_l1 = np.log(error[l-1]/l1_er)/np.log(2)
						roc_l1_wen.append(p_l1)
			if(j==2):
				for l ,linf_er in enumerate(error):
					if(l>0):
						p_linf = np.log(error[l-1]/linf_er)/np.log(2)
						roc_linf_wen.append(p_linf)


		print "L2 ERROR_wen:"
		print errors_wen[0]
		print "L1 ERROR_wen:"
		print errors_wen[1]
		print "L1 ERROR_wen:"
		print errors_wen[2]
		print
		print "ROC L2"
		print roc_l2_wen
		print "ROC L1"
		print roc_l1_wen
		print "ROC Linf"
		print roc_linf_wen
		print 'Done Plotting Errors'
		#----------------------------------------------------------------------------------#
		plt.title('Order of Convergence',fontsize=60 )

		plt.xlabel('$nx$',fontsize=50)
		plt.ylabel('Error',fontsize=50)
		plt.legend(prop={'size': 30})
		#plt.ylabel('Error')
		#plt.xlabel('$nx$')
		#plt.legend()

		plt.savefig('benchmark.png')
		plt.show()

#---------------------------------------------------------------------------------------------------------------#

matplotlib.rc('xtick',labelsize=40)
matplotlib.rc('ytick',labelsize=40)

pi = math.pi

cour = .5
nxs = [8,16,32,64,128,256,1024,2048]
nxs = [8,16,32,64,128,256]

time, xs, us1, us2, real_sols, nx_us1, nx_us2, nx_sols = gen_data(nxs,1,cour)
#time, x, us1, us2, real_sols, nx_us1, nx_us2, nx_sols = gen_data_sin(nxs,1,cour)
#time, x, us1, us2, real_sols, nx_us1, nx_us2, nx_sols = gen_data_square(nxs,1)
print us1[0]
print us1[0]
print real_sols[0]

plotAnim(time, xs[-1], us1, us2, real_sols)

benchmark_lax(nx_us1,nx_us2,nx_sols,nxs)


