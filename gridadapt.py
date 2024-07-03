#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 24 Jun 24

@author: vb
"""

import numpy as np
import random
import os
from os.path import join
import ratinabox
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
from ratinabox.Neurons import PlaceCells, GridCells
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm import tqdm

def main():

	SimulationName="test"
	random.seed(2024)
	N_I=200
	N_mEC=100
	dt=0.01
	tau_1=0.1
	tau_2=1
	psi_sat=10
	a0=0.1*psi_sat # 0.1 is fraction of active neurons
	s0=0.3 
	epsilon=0.001
	lr=0.001

	ratinabox.autosave_plots = True
	ratinabox.figure_directory = "figs"
	figs_directory = "figs"
	out_directory = "outputs"
	place_directory = join(figs_directory, 'place_fields')
	grid_directory = join(figs_directory, 'grid_fields')
	traj_directory = join(figs_directory, 'trajectories')

	for d in [out_directory, place_directory, grid_directory, traj_directory]:
		os.makedirs(d, exist_ok=True)

	# 1 Initialise environment.
	Env = Environment(params={"aspect": 1, "scale": 1})

	# 3 Add Agent.
	Ag = Agent(Env, params={"dt": dt})
	Agent.speed_mean = 1 #m/s
	Ag.pos = np.array([0.5, 0.5])
	n_steps = int(1e7)
	tol_a=0.1
	tol_s=0.1

	# 4 Add place cells.
	PCs = PlaceCells(
		Ag,
		params={
			"n": N_I,
			"description": "gaussian_threshold",
			"widths": 0.40,
			"wall_geometry": "line_of_sight",
			"max_fr": 1,
			"min_fr": 0.1,
			"color": "C1",
		},
	)

	r=np.zeros(N_I)
	r_tempmean=np.zeros(N_I)
	
	h=np.zeros(N_mEC)
	r_act=np.zeros(N_mEC)
	r_inact=np.zeros(N_mEC)
	psi=np.zeros(N_mEC)
	psi_tempmean=np.zeros(N_mEC)

	# J=np.zeros((N_mEC, N_I))
	# J=np.random.random((N_mEC, N_I))

	J = 0.9 + 0.1*np.random.uniform(size=(N_mEC, N_I)) # random weights (from hc to ec) initialization
	sum_weight = np.sum(J**2, axis=1)
	J /= np.sqrt(sum_weight)[:,None]

	# create space domain to visualize firing fields
	x_min, y_min = 0., 0. # ...
	x_max, y_max = 1., 1. # ...
	n_points_x = 30
	n_points_y = 30
	_dx, _dy = (x_max - x_min)/n_points_x, (y_max - y_min)/n_points_y
	xs, ys = np.meshgrid(np.linspace(x_min, x_max, n_points_x),
						 np.linspace(y_min, y_max, n_points_y)
			)
	_firing_fields = np.zeros((N_mEC, n_points_x, n_points_y))
	_place_fields = np.zeros((N_I, n_points_x, n_points_y, ))


	_trajectory = np.zeros((2,n_steps))
	neuron_ids = np.random.choice(N_mEC, 4*6) #np.arange(6)
	place_ids = np.random.choice(N_I, 4*6) #np.arange(6)
	snapshots = np.arange(0, n_steps, 1000)

	theta=0.0
	g=1

	for step in range(n_steps):
		# print(step)

		Ag.update()
		PCs.update()

		# fig, ax = Ag.plot_trajectory()
		# fig, ax = PCs.plot_rate_timeseries()

		r = np.ravel(PCs.history['firingrate'][step])
		# print('r', r)
		# print('\n')
		h = np.dot(J, r)  
		# print('h', h)
		# print('\n')
		r_act += dt*(h-r_inact-r_act)/tau_1
		# print('r_act', r_act)
		# print('\n')
		r_inact += dt*(h-r_inact)/tau_2
		# print('r_inact', r_inact)
		# print('\n')

		psi = transfer(r_act, theta, g, psi_sat)

		psi = Sparsify(psi, s0)
		a = np.mean(psi)

		g = g/(a+0.00001)
		psi = transfer(r_act, theta, g, psi_sat)

		a = np.mean(psi)
		s = np.sum(psi)**2/(N_mEC*np.sum(psi**2)+0.00001)

		# print('after')
		print('a', a)
		print('s', s)

		psi_tempmean = (psi + step*psi_tempmean)/(step+1)
		r_tempmean = (r + step*r_tempmean)/(step+1) 

		# update weights
		J += lr*( psi[:,None]*r[None,:] - (psi_tempmean[:,None] * r_tempmean[None,:]) ) 

		# set negative values to zero
		J[np.where(J<0)] = 0.
		# exit()

		## for each unit in mEC, normalize all ingoing weights onto it to the sum of all of them 
		for k in range(N_mEC):
			if np.sum(J[k, :]) > 1.0e-20:
				J[k, :] /= np.sqrt(np.sum(J[k, :]))
			else: 
				J[k, :] /= np.sqrt(N_I)

		# build the firing field for visualization
		xt, yt = Ag.pos
		_trajectory[:, step] = Ag.pos
		_kernel_map = _kernel(xs - xt, ys - yt)
		_place_fields += r[:, None, None] * _kernel_map[None,:,:] / n_steps
		_firing_fields += psi[:, None, None] * _kernel_map[None, :, :] / n_steps

		if step in snapshots:
			print(step)
			fig1, axs1 = plt.subplots(4, 6, figsize=(12, 6))
			fig2, axs2 = plt.subplots(4, 6, figsize=(12, 6))
			plt.tight_layout()

			for i, (id1, id2, ax1, ax2) in enumerate(zip(place_ids, neuron_ids, axs1.ravel(), axs2.ravel())):
				# plot reconstructed place fields
				ax1.set_title(f"Neuron {id1}")
				ax1.imshow(_place_fields[id1], origin='lower', extent=[x_min, x_max, y_min, y_max])
				ax1.scatter([PCs.place_cell_centres[id1][0]],[PCs.place_cell_centres[id1][1]], c='r', s=1)

				# plot psi fields
				ax2.set_title(f"Neuron {id2}")
				ax2.imshow(_firing_fields[id2], origin='lower', extent=[x_min, x_max, y_min, y_max])
				# ax2.scatter([PCs.place_cell_centres[id2][0]],[PCs.place_cell_centres[id2][1]], c='r', s=1)

			fig1.savefig(join(place_directory, 'heatmap_%s.svg'%step))
			plt.close(fig1)

			fig2.savefig(join(grid_directory, 'heatmap_%s.svg'%step))
			plt.close(fig2)

			fig, ax = plt.subplots()
			ax.plot(*_trajectory, c='r')
			fig.savefig(join(traj_directory, 'trajectory.svg'))
			plt.close(fig)

	np.save(join(out_directory, "place_fields.npy"), _place_fields)
	np.save(join(out_directory, "firing_fields.npy"), _firing_fields)
	return

# FUNCTIONS

def _saturating (x):
	_z = 2./np.pi*np.arctan(x)
	_z[np.where(x > 50)] = 1.
	return _z

def transfer(h, theta, g, psi_sat):
	_psi = psi_sat*_saturating(g*(h-theta))*Heaviside(h-theta)
	return _psi

def Heaviside(h):
	# _h = np.array([_x if _x>=0 else 0 for _x in x])
	_h = h.copy()
	_h[np.where(h<0)] = 0.
	return _h

def _kernel (x, y, sigma=.025):
	_z = 0.5 * (x**2 + y**2) / (2 * sigma**2)
	_norm = 1.
	return np.exp(-_z)/_norm

def Sparsify(h, s0):
        vout=h
        th=np.percentile(h, (1.0-s0)*100)
        vout = [0 if x < th else x-th for x in vout]
        return vout


# def Sparsify(V, theta):
# 	vout=V
# 	th=np.percentile(V,(1.0-f)*100)
# 	for i in range(len(V)):
# 		if vout[i]<th:
# 			vout[i]=0
# 		else:
# 			vout[i]=vout[i]-th
# 	return vout

if __name__ == "__main__":
	main()



		# delta_a = 1000
		# delta_s = 1000

		# if (delta_a > tol_a) or (delta_s > tol_s):

		# 	# 1. find threshold to match sparsity

		# 	def sparsity (theta):
		# 		psi = transfer(r_act, theta, g, psi_sat)
		# 		s = np.sum(psi)**2/(N_mEC*np.sum(psi**2)+0.00001)
		# 		return s

		# 	def distance(x):
		# 		_theta = x
		# 		s = sparsity(_theta)
		# 		return (s-s0)**2

		# 	opt = minimize(distance, np.array([theta]))
		# 	theta = opt['x'][0]

		# 	# 2. find gain to match activity
		# 	def activity (theta, g):
		# 		psi = transfer(r_act, theta, g, psi_sat)
		# 		a = np.mean(psi)
		# 		return a

		# 	def distance(x):
		# 		_theta, _g = x
		# 		a = activity(_theta,_g)
		# 		return (a-a0)**2

		# 	opt = minimize(distance, np.array([theta, g]))
		# 	theta, g = opt['x']

		# 	# 3. compute relative errors
		# 	delta_a = np.abs(a - a0)/a0
		# 	delta_s = np.abs(s - s0)/s0

		# 	# print('theta', theta)
		# 	# print('g', g)

		# 	psi = transfer(r_act, theta, g, psi_sat)

		# 	psi_tempmean += psi 
		# 	r_tempmean += r 
################################################
