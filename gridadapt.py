#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 24 Jun 24

@author: vb
"""

import numpy as np
import random
import os
import ratinabox
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
from ratinabox.Neurons import PlaceCells, GridCells
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm import tqdm

def main():

	SimulationName="test"
	N_I=200
	N_mEC=100
	dt=0.001
	tau=1
	tau_1=10
	tau_2=30
	psi_sat=30
	a0=0.1*psi_sat # 0.1 is fraction of active neurons
	s0=0.3 
	epsilon=0.001
	lr=0.001

	ratinabox.autosave_plots = True
	ratinabox.figure_directory = "figs/"

	# 1 Initialise environment.
	Env = Environment(params={"aspect": 1, "scale": 1})

	# 3 Add Agent.
	Ag = Agent(Env)#, params={"dt": dt})
	Agent.speed_mean = 0.5 #m/s
	Ag.pos = np.array([0.5, 0.5])
	n_steps = int(200/Ag.dt)
	tol_a=0.01
	tol_s=0.01

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

	h=np.zeros(N_mEC)
	r=np.zeros(N_I)
	r_act=np.zeros(N_mEC)
	r_inact=np.zeros(N_mEC)
	psi=np.zeros(N_mEC)
	psi_tempmean=np.zeros(N_mEC)
	r_tempmean=np.zeros(N_I)
	J=np.random.random((N_mEC, N_I))


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
	snapshots = np.arange(0, n_steps, 200)
	print(snapshots)

	for step in tqdm(range(n_steps)):

		Ag.update()
		PCs.update()

		# fig, ax = Ag.plot_trajectory()
		# fig, ax = PCs.plot_rate_timeseries()

		r = np.ravel(PCs.history['firingrate'][step])
		# print('r', r)
		# print('\n')
		h += dt*(np.dot(J, r))/tau  
		# print('h', h)
		# print('\n')
		r_act += dt*(h-r_inact-r_act)/tau_1
		# print('r_act', r_act)
		# print('\n')
		r_inact += dt*(h-r_inact)/tau_2
		# print('r_inact', r_inact)
		# print('\n')

		theta=0.1
		g=0.1

		psi = transfer(r_act, theta, g, psi_sat)
		a = np.mean(psi)
		s = np.sum(psi)**2/(N_mEC*np.sum(psi**2)+0.00001)

		delta_a = 1000
		delta_s = 1000

		if (delta_a > tol_a) or (delta_s > tol_s):

			# 1. find threshold to match sparsity

			def sparsity (theta):
				psi = transfer(r_act, theta, g, psi_sat)
				s = np.sum(psi)**2/(N_mEC*np.sum(psi**2)+0.00001)
				return s

			def distance(x):
				_theta = x
				s = sparsity(_theta)
				return (s-s0)**2

			opt = minimize(distance, np.array([theta]))
			theta = opt['x'][0]

			# 2. find gain to match activity
			def activity (theta, g):
				psi = transfer(r_act, theta, g, psi_sat)
				a = np.mean(psi)
				return a

			def distance(x):
				_theta, _g = x
				a = activity(_theta,_g)
				return (a-a0)**2

			opt = minimize(distance, np.array([theta, g]))
			theta, g = opt['x']

			# 3. compute errors
			delta_a = np.abs(a - a0)/a0
			delta_s = np.abs(s - s0)/s0

			print('a', a)
			print('s', s)

		# print('theta', theta)
		# print('g', g)

		psi = transfer(r_act, theta, g, psi_sat)
		psi_tempmean += psi 
		r_tempmean += r 

		# J += lr*(np.outer(psi,r) - np.outer(psi_tempmean, r_tempmean))
		J += lr*( psi[:,None]*r[None,:] - (psi_tempmean[:,None] * r_tempmean[None,:])/(step+1)**2 ) 

		# build the firing field for visualization
		xt, yt = Ag.pos
		_trajectory[:, step] = Ag.pos
		_kernel_map = _kernel(xs - xt, ys - yt)
		# print(_kernel_map)
		_firing_fields += psi[:, None, None] * _kernel_map[None, :, :] / n_steps
		_place_fields += r[:, None, None] * _kernel_map[None,:,:] / n_steps

		if step in snapshots:
			print(step)
			fig, axs = plt.subplots(4, 6, figsize=(12, 6))
			plt.tight_layout()
			for i, (n_id, ax) in enumerate(zip(neuron_ids, axs.ravel())):
				ax.imshow(_firing_fields[n_id], origin='lower', extent=[x_min, x_max, y_min, y_max])
				# ax.imshow(_place_fields[n_id], origin='lower', extent=[x_min, x_max, y_min, y_max])
				ax.scatter([PCs.place_cell_centres[n_id][0]],[PCs.place_cell_centres[n_id][1]], c='r', s=1)
				ax.set_title(f"Neuron {n_id}")
			fig.savefig('heatmap_%s.svg'%step)
			
			fig1 = plt.figure()
			plt.plot(*_trajectory, c='r')
			fig1.savefig('trajectory_%s.svg'%step)
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

def _kernel (x, y, sigma=.05):
	_z = 0.5 * (x**2 + y**2) / (2 * sigma**2)
	_norm = 1.
	return np.exp(-_z)/_norm

if __name__ == "__main__":
	main()
