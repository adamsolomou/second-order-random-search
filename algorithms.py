import math 
import time
import itertools
import autograd.numpy as np

from utils import uniform_angles_pss

def STP(f, x, a_init, step_upd='half', distribution='Uniform', T=10000):
	# Initialization 
	y = x
	a = a_init

	# Function values
	f_values = []
	f_values.append((0 ,f.eval(y)))

	# Gradient norm 
	g_norm = []
	g_norm.append((0, np.linalg.norm(f.gradient(y))))

	# Execution time per iteration 
	timer = []

	for t in range(1, T): 
		# Start timer 
		s_time = time.time()

		if distribution == 'Uniform': 
			s = np.random.multivariate_normal(np.zeros(f.d), np.identity(f.d))
			s = s/np.linalg.norm(s)
		elif distribution == 'Normal': 
			s = np.random.multivariate_normal(np.zeros(f.d), np.identity(f.d))
		else: 
			raise ValueError('The option %s is not a supported sampling distribution.' %(distribution))

		# List possible next iterates
		V = [y+a*s, y-a*s, y]

		f_v = []
		for v in V: 
			f_v.append(f.eval(v))

		# Select optimal point
		i_star = np.argmin(np.array(f_v))

		# Update step 
		y = V[i_star]

		# Step size update 
		if step_upd == 'half': 
			if t%10 == 0: 
				a = a/2
		elif step_upd == 'inv': 
			a = a_init/(t+1)
		elif step_upd == 'inv_sqrt': 
			a = a_init/np.sqrt(t+1)
		else: 
			raise ValueError('The option %s is not a supported step size update rule.' %(step_upd))

		# Stop timer
		e_time = time.time()
		timer.append(e_time - s_time)

		f_values.append((t, f.eval(y)))
		g_norm.append((t, np.linalg.norm(f.gradient(y))))

	# Summary 
	summary = {}
	summary['x_T'] = y
	summary['fval'] = np.array(f_values)
	summary['gnorm'] = np.array(g_norm)
	summary['time'] = np.mean(timer)

	return summary

def BDS(f, x, a_init, a_max, theta, gamma, rho, T=10000): 
	# Initialization 
	y = x # iterate @ t 
	a = a_init # step size 

	# Function values
	f_values = []
	f_values.append((0, f.eval(y)))

	# Gradient norm 
	g_norm = []
	g_norm.append((0, np.linalg.norm(f.gradient(y))))

	# Execution time per iteration 
	timer = []

	for t in range(1,T):
		# Start timer 
		s_time = time.time()

		# Reset variables 
		successful = False 
		d_opt = np.zeros(f.d) 
		f_y = f.eval(y) # function value at current iterate 

		# Generate a polling set 
		D, D_symmetric, D_size = uniform_angles_pss(f.d)

		# Search the polling set 
		for i in np.random.permutation(D_size):
			d = D[:,i]
			if f.eval(y + a*d) < f_y - rho(a): 
				# Iteration succesful 
				d_opt = d
				successful = True 
				# Stop searching PSS
				break 
	
		# Update step
		if successful: 
			y = y + a*d_opt
			a = np.minimum(gamma*a, a_max)
		else: 
			a = theta*a

		# Stop timer
		e_time = time.time()
		timer.append(e_time - s_time)

		f_values.append((t, f.eval(y)))
		g_norm.append((t, np.linalg.norm(f.gradient(y))))
	
	# Summary 
	summary = {}
	summary['x_T'] = y
	summary['fval'] = np.array(f_values)
	summary['gnorm'] = np.array(g_norm)
	summary['time'] = np.mean(timer)

	return summary

def AHDS(f, x, a_init, a_max, theta, gamma, rho, T=10000):
	# Initialization 
	y = x # iterate @ t 
	a = a_init # step size 

	# Function values
	f_values = []
	f_values.append((0, f.eval(y)))

	# Gradient norm 
	g_norm = []
	g_norm.append((0, np.linalg.norm(f.gradient(y))))

	# Execution time per iteration 
	timer = []

	# Flag to indicate if the algorithms get's stucked
	stacked = False
  
	for t in range(1,T): 
		# Start timer 
		s_time = time.time()

		# Reset variables 
		successful = False 
		H = np.zeros((f.d, f.d)) # Hessian 
		f_y = f.eval(y) # function value at current iterate 
		d_opt = np.zeros(f.d) # descent direction  
		B_opt = np.zeros((f.d, f.d)) # independent set of vectors
		D_table = np.zeros(f.d) # store function values for Hessian computation 
		B_table = np.zeros((f.d, f.d)) # store function values for Hessian computation

		""" ========= Step 1 ========= """
		# Generate a PSS D
		D, D_symmetric, D_size = uniform_angles_pss(f.d)

		# Search the PSS
		for i in np.random.permutation(D_size):
			d = D[:,i]
			if f.eval(y + a*d) < f_y - rho(a): 
				# Iteration succesful 
				d_opt = d
				successful = True 
				# Stop searching PSS
				break 

		""" ========= Step 2 ========= """
		# Search opposite directions in PSS 
		if not D_symmetric and not successful: 
			for i in np.random.permutation(D_size):
				d = -D[:,i]
				if f.eval(y + a*d) < f_y - rho(a): 
					# Iteration succesful 
					d_opt = d
					successful = True 
					# Stop searching PSS
					break

		""" ========= Step 3 ========= """
		if not successful: 
			# Choose B as a subset of D with f.d linearly independent vectors
			subsets = itertools.combinations(range(D_size), f.d)
			for subset in np.random.permutation(list(subsets)): 
				B = D[:,subset]
				if np.linalg.matrix_rank(B) == f.d: 
					B_opt = B
					# Stop search 
					break

			break_outer = False 
			for i in range(f.d-1): 
				for j in range(i+1,f.d): 
					d = B_opt[:,i] + B_opt[:,j]
					B_table[i,j] = f.eval(y + a*d)
					if B_table[i,j] < f_y - rho(a): 
						# Iteration successful 
						d_opt = d
						successful = True 
						# Stop searching 
						break_outer = True 
						break 
				if break_outer: 
					# Stop searching 
					break
	
		""" ========= Step 4 ========= """
		# if not successful and not stacked: 
		if not successful:
			# Hessian approximation: Diagonal elements 
			for i in range(f.d): 
				di = B_opt[:,i]
				D_table[i] = f.eval(y+a*di)
				H[i,i] = D_table[i] - 2*f_y + f.eval(y-a*di)

			# Hessian approximation: Off-diagonal elements 
			for i in range(f.d-1): 
				for j in range(i+1,f.d): 
					H[i,j] = B_table[i,j] - D_table[i] - D_table[j] + f_y
					H[j,i] = H[i,j]

			# Complete computation     
			H = H/(a**2)

			# When iterates get very close to a minimizer the Hessian approximation 
			# may result to NaN values. The try statement avoids such errors. 
			try:
				# Eigendecomposition
				L, V = np.linalg.eig(H)

				# Eigenvector corresponding to minimum eigenvalue 
				idx = np.argmin(L)
				d = V[:,idx]

				# Check d 
				if f.eval(y + a*d) < f.eval(y) - rho(a): 
					# Iteration successful 
					d_opt = d
					successful = True

				# Check -d
				if f.eval(y - a*d) < f.eval(y) - rho(a) and f.eval(y - a*d) < f.eval(y + a*d): 
					# Iteration successful 
					d_opt = -d
					successful = True
			except: 
				pass

		""" ========= Step 5 ========= """
		# Update step
		if successful: 
			y = y + a*d_opt
			a = np.minimum(gamma*a, a_max)
			stacked = False
		else: 
			a = theta*a
			stacked = True

		# Stop timer
		e_time = time.time()
		timer.append(e_time - s_time)

		f_values.append((t, f.eval(y)))
		g_norm.append((t, np.linalg.norm(f.gradient(y))))

	# Summary 
	summary = {}
	summary['x_T'] = y
	summary['fval'] = np.array(f_values)
	summary['gnorm'] = np.array(g_norm)
	summary['time'] = np.mean(timer)

	return summary

def RS(f, x, a_init, sigma_1, sigma_2, distribution='Normal', step_upd='half', theta=0.6, T_half=10, T=10000):
	# Initialization 
	y = x # iterate @ t 
	a = a_init # step size 

	# Function values
	f_values = []
	f_values.append((0, f.eval(y)))

	# Gradient norm 
	g_norm = []
	g_norm.append((0, np.linalg.norm(f.gradient(y))))

	# Execution time per iteration 
	timer = []

	for t in range(1,T): 
		# Start timer 
		s_time = time.time()

		""" ========= Random Step 1 ========= """
		if distribution == 'Uniform': 
			d1 = np.random.multivariate_normal(np.zeros(f.d), np.identity(f.d))
			d1 = sigma_1*(d1/np.linalg.norm(d1))
		elif distribution == 'Normal': 
			d1 = np.random.multivariate_normal(np.zeros(f.d), np.power(sigma_1,2.0)*np.identity(f.d))
		else: 
			raise ValueError('The option %s is not a supported sampling distribution.' %(distribution))

		V = [y, y+a*d1, y-a*d1]

		f_v = []
		for v in V: 
			f_v.append(f.eval(v))

		# Select optimal point
		i_star = np.argmin(np.array(f_v))

		# Update iterate 
		y = V[i_star]

		""" ========= Random Step 2 ========= """
		if i_star == 0: 
			if distribution == 'Uniform': 
				d2 = np.random.multivariate_normal(np.zeros(f.d), np.identity(f.d))
				d2 = sigma_2*(d2/np.linalg.norm(d2))
			elif distribution == 'Normal': 
				d2 = np.random.multivariate_normal(np.zeros(f.d), np.power(sigma_2,2.0)*np.identity(f.d))
			else: 
				raise ValueError('The option %s is not a supported sampling distribution.' %(distribution))

			V = [y, y+a*d2, y-a*d2]

			f_v = []
			for v in V: 
				f_v.append(f.eval(v))

			# Select optimal point
			i_star = np.argmin(np.array(f_v))

			# Update iterate 
			y = V[i_star]

		# Update step-size 
		if step_upd == 'half': 
			if t%T_half == 0: 
				a = theta*a
		elif step_upd == 'inv': 
			a = a_init/(t+1)
		elif step_upd == 'inv_sqrt': 
			a = a_init/np.sqrt(t+1)
		else: 
			raise ValueError('The option %s is not a supported step size update rule.' %(step_upd))

		# Stop timer
		e_time = time.time()
		timer.append(e_time - s_time)

		f_values.append((t, f.eval(y)))
		g_norm.append((t, np.linalg.norm(f.gradient(y))))

	# Summary 
	summary = {}
	summary['x_T'] = y
	summary['fval'] = np.array(f_values)
	summary['gnorm'] = np.array(g_norm)
	summary['time'] = np.mean(timer)

	return summary

def DFPI_SPSA(f, y, c_init, beta, T_power):
	# Power iteration - Compute eigenvector for max eigenvalue 
	r = 0.001
	T_power_approx = 5
	d2 = np.random.rand(f.d)

	c = c_init 
	for i in range(T_power_approx): 
		Delta = np.random.binomial(n=1, p=0.5, size=f.d)
		Delta[Delta == 0] = -1
		# Approximate gradient vectors 
		d_rplus = f.eval(y + r*d2 + c*Delta) - f.eval(y + r*d2 - c*Delta)
		G_rplus = np.divide(d_rplus, 2*c*Delta)

		d_rminus = f.eval(y - r*d2 + c*Delta) - f.eval(y - r*d2 - c*Delta)
		G_rminus = np.divide(d_rminus, 2*c*Delta)

		# Approximate Hessian-vector product
		Hd = (G_rplus - G_rminus)/(2*r)
		
		# Power iteration - update
		d2 = Hd/np.linalg.norm(Hd)

	# Approximate gradient vectors 
	d_rplus = f.eval(y + r*d2 + c*Delta) - f.eval(y + r*d2 - c*Delta)
	G_rplus = np.divide(d_rplus, 2*c*Delta)

	d_rminus = f.eval(y - r*d2 + c*Delta) - f.eval(y - r*d2 - c*Delta)
	G_rminus = np.divide(d_rminus, 2*c*Delta)

	# Approximate Hessian-vector product
	Hd = (G_rplus - G_rminus)/(2*r)

	# Largest eigenvalue 
	lmax = np.linalg.norm(Hd)/np.linalg.norm(d2)

	# Power iteration - Compute eigenvector for min eigenvalue 
	b_power = 1/lmax
	d2 = np.random.rand(f.d)
	for i in range(T_power): 
		Delta = np.random.binomial(n=1, p=0.5, size=f.d)
		Delta[Delta == 0] = -1

		# Approximate gradient vectors 
		d_rplus = f.eval(y + r*d2 + c*Delta) - f.eval(y + r*d2 - c*Delta)
		G_rplus = np.divide(d_rplus, 2*c*Delta)

		d_rminus = f.eval(y - r*d2 + c*Delta) - f.eval(y - r*d2 - c*Delta)
		G_rminus = np.divide(d_rminus, 2*c*Delta)

		# Approximate Hessian-vector product
		Hd = (G_rplus - G_rminus)/(2*r)
		
		# Power iteration - update
		d2_ = d2 - b_power*Hd
		d2  = d2_/np.linalg.norm(d2_)

	# Negative curvature 
	return d2

def DFPI_FD(f, y, c, T_power):
	r = 0.01

	# Power iteration - Compute eigenvector for max eigenvalue 
	T_power_approx = 15
	d2 = np.random.rand(f.d)

	# Basis vectors
	I = np.identity(f.d)

	for i in range(T_power_approx): 
		# Initialize 
		g_p = np.empty(f.d)
		g_m = np.empty(f.d)

		# Approximate gradient vectors
		for j in range(f.d): 
			g_p[j] = (f.eval(y + r*d2 + c*I[:,j]) - f.eval(y + r*d2 - c*I[:,j]))/(2*c)
			g_m[j] = (f.eval(y - r*d2 + c*I[:,j]) - f.eval(y - r*d2 - c*I[:,j]))/(2*c)

		# Approximate Hessian-vector product
		Hd = (g_p - g_m)/(2*r)
	
		# Power iteration - update
		d2 = Hd/np.linalg.norm(Hd)

	# Approximate gradient vectors 
	g_p = np.empty(f.d)
	g_m = np.empty(f.d)

	for j in range(f.d): 
		g_p[j] = (f.eval(y + r*d2 + c*I[:,j]) - f.eval(y + r*d2 - c*I[:,j]))/(2*c)
		g_m[j] = (f.eval(y - r*d2 + c*I[:,j]) - f.eval(y - r*d2 - c*I[:,j]))/(2*c)

	# Approximate Hessian-vector product
	Hd = (g_p - g_m)/(2*r)

	# Largest eigenvalue 
	lmax = np.linalg.norm(Hd)/np.linalg.norm(d2)

	# Power iteration - Compute eigenvector for min eigenvalue 
	b_power = 1/lmax
	d2 = np.random.rand(f.d)

	for i in range(T_power): 
		# Initialize 
		g_p = np.empty(f.d)
		g_m = np.empty(f.d)

		# Approximate gradient vectors
		for j in range(f.d): 
			g_p[j] = (f.eval(y + r*d2 + c*I[:,j]) - f.eval(y + r*d2 - c*I[:,j]))/(2*c)
			g_m[j] = (f.eval(y - r*d2 + c*I[:,j]) - f.eval(y - r*d2 - c*I[:,j]))/(2*c)

		# Approximate Hessian-vector product
		Hd = (g_p - g_m)/(2*r)
	
		# Power iteration - update
		d2_ = d2 - b_power*Hd
		d2  = d2_/np.linalg.norm(d2_)

	return d2

def RSPI_SPSA(f, x, a_init, c_init, beta, sigma_1, sigma_2, distribution='Normal', step_upd='half', theta=0.6, T_half=10, T_power=100, T=10000):
	# Initialization 
	y = x # iterate @ t 
	a = a_init # step size 
	c = c_init # SPSA step 

	# Function values
	f_values = []
	f_values.append((0, f.eval(y)))

	# Gradient norm 
	g_norm = []
	g_norm.append((0, np.linalg.norm(f.gradient(y))))

	# Execution time per iteration 
	timer = []

	for t in range(1,T): 
		# Start timer 
		s_time = time.time()

		""" ========= Random Step ========= """
		if distribution == 'Uniform': 
			d1 = np.random.multivariate_normal(np.zeros(f.d), np.identity(f.d))
			d1 = sigma_1*(d1/np.linalg.norm(d1))
		elif distribution == 'Normal': 
			d1 = np.random.multivariate_normal(np.zeros(f.d), np.power(sigma_1,2.0)*np.identity(f.d))
		else: 
			raise ValueError('The option %s is not a supported sampling distribution.' %(distribution))

		V = [y, y+a*d1, y-a*d1]

		f_v = []
		for v in V: 
			f_v.append(f.eval(v))

		# Select optimal point
		i_star = np.argmin(np.array(f_v))

		# Update iterate 
		y = V[i_star]
	
		""" ========= Negative Curvature ========= """
		if i_star == 0: 
			d2 = DFPI_SPSA(f, y, c, beta, T_power)
		
			while d2 is None: 
				d2 = DFPI_SPSA(f, y, c, beta, T_power)
			
			""" ========= Update Step ========= """
			V = [y, y+sigma_2*d2, y-sigma_2*d2]

			f_v = []
			for v in V: 
				f_v.append(f.eval(v))

			# Select optimal point
			i_star = np.argmin(np.array(f_v))

			# Update iterate 
			y = V[i_star]

		# Decrease SPSA parameter 
		c = c_init/pow(t,beta)

		# Update step-size 
		if step_upd == 'half': 
			if t%T_half == 0: 
				a = theta*a
		elif step_upd == 'inv': 
			a = a_init/(t+1)
		elif step_upd == 'inv_sqrt': 
			a = a_init/np.sqrt(t+1)
		else: 
			raise ValueError('The option %s is not a supported step size update rule.' %(step_upd))

		# Stop timer
		e_time = time.time()
		timer.append(e_time - s_time)

		f_values.append((t, f.eval(y)))
		g_norm.append((t, np.linalg.norm(f.gradient(y))))

	# Summary 
	summary = {}
	summary['x_T'] = y
	summary['fval'] = np.array(f_values)
	summary['gnorm'] = np.array(g_norm)
	summary['time'] = np.mean(timer)

	return summary

def RSPI_FD(f, x, a_init, c_init, beta, sigma_1, sigma_2, distribution='Normal', step_upd='half', theta=0.6, T_half=10, T_power=100, T=10000):
	# Initialization 
	y = x # iterate @ t 
	a = a_init # step size 
	c = c_init # SPSA step 

	# Function values
	f_values = []
	f_values.append((0, f.eval(y)))

	# Gradient norm 
	g_norm = []
	g_norm.append((0, np.linalg.norm(f.gradient(y))))

	# Execution time per iteration 
	timer = []

	for t in range(1,T): 
		# Start timer 
		s_time = time.time()
	
		""" ========= Random Step ========= """
		if distribution == 'Uniform': 
			d1 = np.random.multivariate_normal(np.zeros(f.d), np.identity(f.d))
			d1 = sigma_1*(d1/np.linalg.norm(d1))
		elif distribution == 'Normal': 
			d1 = np.random.multivariate_normal(np.zeros(f.d), np.power(sigma_1,2.0)*np.identity(f.d))
		else: 
			raise ValueError('The option %s is not a supported sampling distribution.' %(distribution))

		V = [y, y+a*d1, y-a*d1]

		f_v = []
		for v in V: 
			f_v.append(f.eval(v))

		# Select optimal point
		i_star = np.argmin(np.array(f_v))

		# Update iterate 
		y = V[i_star]
		
		""" ========= Negative Curvature ========= """
		if i_star == 0: 
			d2 = DFPI_FD(f, y, c, T_power)
		
			while d2 is None: 
				d2 = DFPI_FD(f, y, c, T_power)
			
			""" ========= Update Step ========= """
			V = [y, y+sigma_2*d2, y-sigma_2*d2]

			f_v = []
			for v in V: 
				f_v.append(f.eval(v))

			# Select optimal point
			i_star = np.argmin(np.array(f_v))

			# Update iterate 
			y = V[i_star]

		# Decrease SPSA parameter 
		c = c_init/pow(t,beta)

		# Update step-size 
		if step_upd == 'half': 
			if t%T_half == 0: 
				a = theta*a
		elif step_upd == 'inv': 
			a = a_init/(t+1)
		elif step_upd == 'inv_sqrt': 
			a = a_init/np.sqrt(t+1)
		else: 
			raise ValueError('The option %s is not a supported step size update rule.' %(step_upd))

		# Stop timer
		e_time = time.time()
		timer.append(e_time - s_time)

		f_values.append((t, f.eval(y)))
		g_norm.append((t, np.linalg.norm(f.gradient(y))))

	# Summary 
	summary = {}
	summary['x_T'] = y
	summary['fval'] = np.array(f_values)
	summary['gnorm'] = np.array(g_norm)
	summary['time'] = np.mean(timer)

	return summary