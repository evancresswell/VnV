import sys,os
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation
from itertools import groupby
import random
from random import randrange
from random import shuffle

#--------------------------------------------------------------------------------------------------------------#

def rho_post(rho_pre, gamma, M):
	rp = rho_pre*( ( ( gamma + 1. ) * (M**2.) ) / ( 2. + ( gamma - 1. ) * (M**2.) ) ) 
	return rp

def u_post(u_pre, gamma, M):
	up = u_pre*( ( 2. + ( gamma - 1.  ) * (M**2.) ) / ( ( gamma + 1. ) * (M**2.) ) )
	return up

def p_post(p_pre, gamma, M):
	pp = p_pre*( ( 2. * gamma * (M**2.) - ( gamma - 1. ) ) / ( gamma + 1. ) )
	return pp

def temp(rho, p):
	A_bar = 8.
	R = 8.3e7
	T = ( p * A_bar ) / ( R * rho )
	return T

def cs(rho, p, gamma):
	cs = (gamma*p/rho)**.5
	return cs

def get_u(rho, p, gamma, M):
	c = cs(rho, p, gamma)
	u = c * M
	return u

def shock(rho, p, gamma, M ):
	u = get_u(rho, p, gamma, M)
	rp = rho_post(rho, gamma, M)
	up = u_post(u, gamma, M)
	pp = p_post(p, gamma, M)
	return rp,up,pp

def get_ranges( rho, p , gamma, M, k):
	# min -> max parameter values
	ps = list(np.linspace(p-.03*p,p+.03*p,k))
	ps_available = list(np.ones(len(ps)))
	rhos = list(np.linspace(rho-.03*rho,rho+.03*rho,k))
	rhos_available = list(np.ones(len(rhos)))
	gammas = list(np.linspace(gamma-.05*gamma,gamma+.03*gamma,k))
	gammas_available = list(np.ones(len(gammas)))
	Ms = list(np.linspace(M-.01*M,M+.01*M,k))
	Ms_available = list(np.ones(len(Ms)))

	return ps, rhos, gammas, Ms 


def get_trajectories(k):

	# shuffle values
	p_perm = list(np.random.permutation(k)) 
	rho_perm = list(np.random.permutation(k)) 
	gamma_perm = list(np.random.permutation(k)) 
	M_perm = list(np.random.permutation(k)) 

	traj_matrix = []
	for i in range(k):
		line = [p_perm[i], rho_perm[i], gamma_perm[i], M_perm[i]]		
		traj_matrix.append(line)
	print '\n'
	print 'traj_matrix:'
	for line in traj_matrix:
		print line
	print '\n'
	return traj_matrix

def morris(traj_matrix, ps, rhos, gammas, Ms):
	EEs = []
	pre_par = []
	post_par = []
	T = []

	# iterate through reference trajectories
	# traj0 = [ p, rho, gamma, M  ]
	for traj0 in traj_matrix:
		EE = []

		# set reference values
		# calculate reference trajectory temp	
		p = ps[traj0[0]]
		rho = rhos[traj0[1]]
		gamma = gammas[traj0[2]]
		M = Ms[traj0[3]]

		# SAVE SHIT on FIRST STEP
		# save pre-shock parameter values
		pre_par.append([p, rho, gamma, M])	
		# shock the system
		rp,up,pp = shock(rho, p, gamma, M)
		# save post-shock parameter values
		cs = (gamma*pp/rp)**.5
		M = up/cs
		post_par.append([pp, rp, gamma, M])	
		Tpre = temp(rp,pp)
		T.append(Tpre)

		print 'Reference trajectory indices: ' + str(traj0)
		#print T0
		for i in range(len(traj0)):

			index = traj0[i]
			new_index = index
			while(new_index == index):
				new_index = randrange(len(traj_matrix))
			traj0[i] = new_index
			print 'new trajectory: ' + str(traj0)

			# save pre-shock parameter values
		
			# set up new values and get post temp	
			p = ps[traj0[0]]
			rho = rhos[traj0[1]]
			gamma = gammas[traj0[2]]
			M = Ms[traj0[3]]


			# SAVE SHIT 
			# save pre-shock parameter values
			pre_par.append([p, rho, gamma, M])	
			# shock the system
			rp,up,pp = shock(rho, p, gamma, M)
			# save post-shock parameter values
			cs = (gamma*pp/rp)**.5
			M = up/cs
			post_par.append([pp, rp, gamma, M])	
			Tpost = temp(rp,pp)
			T.append(Tpost)

			# caluculate shift for EE value 
			if(i==0):
				delta = abs(ps[new_index] - ps[index])
			if(i==1):
				delta = abs(rhos[new_index] - rhos[index])
			if(i==2):
				delta = abs(gammas[new_index] - gammas[index])
			if(i==3):
				delta = abs(Ms[new_index] - Ms[index])
			
			EE.append(abs(Tpre-Tpost)/delta)
			Tpost = Tpre
		EEs.append(EE)
		print
	
	print 'EEs:'
	print '[     p                   rho                   gamma                   M     ]'
	for EE in EEs:
		print EE
	print
	return pre_par, post_par, T, EEs
	#print pre_par
	#print post_par
	#print T

def plot_par_vs_temp(pre_par,post_par,T):

	for i in range(4):
		x = []
		for line in pre_par:
			x.append(line[i]) 	
		if(i==0):
			plt.plot(x,T,'.')
			plt.xlabel('$p$')
			plt.title('Pressure')
			plt.show()
		if(i==1):
			plt.plot(x,T,'.')
			plt.xlabel(r'$\rho$')
			plt.title('Density')
			plt.show()
		if(i==2):
			plt.plot(x,T,'.')
			plt.xlabel('$\gamma$')
			plt.title('Gas Constant')
			plt.show()
		if(i==3):
			plt.plot(x,T,'.')
			plt.xlabel('$M$')
			plt.title('Mach Number')
			plt.show()


def plot_std_vs_mean(EEs):
	tlist = list(zip(*EEs))
	std = []
	mu = []
	for line in tlist:
		std.append(np.std(line))
		mu.append(np.mean(line))
	plt.plot(mu,std, '.')
	plt.title('$\mu$ vs $\sigma$')
	plt.xlabel('$\mu$')	
	plt.ylabel('$\sigma$')	
	plt.show()

def plot_other(pre_par,post_par):
	print 'still to doo'

#------------------------------------------Run Simulation------------------------------------------------------#
"""
u = get_u(rho, p, gamma, M)
print '\n'
print rho_post(rho, gamma, M)
print u_post(u, gamma, M)
print p_post(p, gamma, M)
print temp(rho, p)
print '\n'
"""

gamma = 1.4
rho = 1.5e-9
p = 1
M = 20

k = 10

# get range of input parameters
ps, rhos, gammas, Ms = get_ranges(rho, p, gamma, M, k)

# get reference trajectories
traj_matrix = get_trajectories(k)

pre_par,post_par,T,EEs = morris(traj_matrix, ps, rhos, gammas, Ms)

#plot_par_vs_temp(pre_par,post_par,T)

plot_std_vs_mean(EEs)
plot_other(pre_par,post_par)




