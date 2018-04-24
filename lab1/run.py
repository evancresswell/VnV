import sys,os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



#---------------------------------------------------------------------------------------------------------------#
def plotAnim(i):

		filename_sol = "scalar_advec_sol.out"
		filename_t = "scalar_advec_t.out"
		filename_mass = "scalar_advec_mass.out"
		times = []
		masses = []
		solutions = []


		infile = open('scalar_advec_sol'+str(i)+'.out','r')
		temp = infile.readlines()
		for line in temp:
			solutions.append([float(x) for x in line.split()])
		#print solutions	

		infile = open('scalar_advec_mass'+str(i)+'.out','r')
		temp = infile.readlines()
		for line in temp:
			masses.append(float(line))
		#print masses

		infile = open('scalar_advec_t'+str(i)+'.out','r')
		temp = infile.readlines()
		for line in temp:
			times.append(float(line))
		#print times


		#plot solution
		fig, ax = plt.subplots()
		fig.set_tight_layout(True)

		x = np.linspace(0,2*3.14159,len(solutions[0]))
		#ax.set_ylim((-1.1,1.1))
		#line, = ax.plot(x,solutions[0],'r-')
		line, = ax.plot(solutions[0],'r-')

		def update(i):
			label = 't = {0}'.format(times[i])
			#print(label)
			# Update the line and the axes (with a new xlabel). Return a tuple of
			# "artists" that have to be redrawn for this frame.
			line.set_ydata(solutions[i])
			#line.set_ydata(x)
			#line2.set_ydata(massesp[i])
			ax.set_xlabel(label)
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
#We want to read in and plot 
fileName_error1 = "l2_error"
fileName_error2 = "l1_error"

c = .5
nx=20
orders = [2]
i=0
print i
cmd_compile = "g++ scalar_advec.cpp -o scalar_advec" 
os.system(cmd_compile)
cmd_1 = "./scalar_advec %(nx)s %(c)s > o" % locals()
os.system(cmd_1)
cmd_2 = "mv l2_error.out l2_error%(i)s.out" % locals()
os.system(cmd_2)
cmd_3 = "mv scalar_advec_mass.out scalar_advec_mass%(i)s.out" % locals()
os.system(cmd_3)
cmd_4 = "mv scalar_advec_sol.out scalar_advec_sol%(i)s.out" % locals()
os.system(cmd_4)
cmd_5 = "mv scalar_advec_t.out scalar_advec_t%(i)s.out" % locals()
os.system(cmd_5)
cmd_6 = "mv l1_error.out l1_error%(i)s.out" % locals()
os.system(cmd_6)

# remove previous simulation files
plotAnim(0) #old plotting for animation and benchmarking


cmd_2 = "tail o"
os.system(cmd_2)

