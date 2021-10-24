import autograd.numpy as np
import matplotlib.pyplot as plt

from autograd import grad 
from autograd import hessian 

class Benchmark(object):
	def __init__(self, dimensions):

		self._dimensions = dimensions
		self.fglob = np.nan
		self.global_optimum = None

	def __str__(self):
		return '{0} ({1} dimensions)'.format(self.__class__.__name__, self.d)

	def __repr__(self):
		return self.__class__.__name__

	def initial_vector(self):
		"""
		Random initialisation for the benchmark problem.
		Returns
		-------
		x : sequence
			a vector of length ``N`` that contains random floating point
			numbers that lie between the lower and upper bounds for a given
			parameter.
		"""

		return np.asarray([np.random.uniform(l, u) for l, u in self.bounds])

	def eval(self, x):
		"""
		Evaluation of the benchmark function.
		Parameters
		----------
		x : sequence
			The candidate vector for evaluating the benchmark problem. Must
			have ``len(x) == self.N``.
		Returns
		-------
		val : float
			  the evaluated benchmark function
		"""

		raise NotImplementedError

	def gradient(self, x): 
		raise NotImplementedError

	def H(self, x): 
		raise NotImplementedError

	def _eval_2D(self, x, y):
		raise NotImplementedError

	def _grad_2D(self, x, y):
		raise NotImplementedError
	
	def _grad_norm_2D(self, x, y):
		raise NotImplementedError

	@property
	def bounds(self):
		"""
		The lower/upper bounds to be used for minimizing the problem.
		This a list of (lower, upper) tuples that contain the lower and upper
		bounds for the problem.  The problem should not be asked for evaluation
		outside these bounds. ``len(bounds) == N``.
		"""
		if self.change_dimensionality:
			return [self._bounds[0]] * self.d
		else:
			return self._bounds

	@property
	def d(self):
		"""        
		The dimensionality of the problem.
		
		Returns
		-------
		N : int
			The dimensionality of the problem
		"""
		return self._dimensions

	@property
	def xmin(self):
		"""
		The lower bounds for the problem
		Returns
		-------
		xmin : sequence
			The lower bounds for the problem
		"""
		return np.asarray([b[0] for b in self.bounds])

	@property
	def xmax(self):
		"""
		The upper bounds for the problem
		Returns
		-------
		xmax : sequence
			The upper bounds for the problem
		"""
		return np.asarray([b[1] for b in self.bounds])
	
	def visualize(self, xmin, xmax, ymin, ymax, delta_x=0.1, delta_y=0.1, paths=None, save_name=None): 

		fig, axs = plt.subplots(1, 3, figsize=(25,5))
	  
		""" Contour Lines """
		X, Y = np.meshgrid(np.arange(xmin, xmax, delta_x), np.arange(ymin, ymax, delta_y))
		Z = self._eval_2D(X, Y)
		axs[0].contour(X, Y, Z)
		# Set limits 
		axs[0].set_xlim(xmin, xmax)
		axs[0].set_ylim(ymin, ymax)
		# Labels 
		axs[0].set_xlabel(r'$x_1$')
		axs[0].set_ylabel(r'$x_2$')
		# Title 
		axs[0].set_title('Level Lines')
		# Plot local/global minima
		for x_star in self.global_optimum: 
			axs[0].plot(*x_star, 'r*', markersize=10)
		# Plot saddle points 
		for x_saddle in self.saddle: 
			axs[0].plot(*x_saddle, 'b+', markersize=10)
		# Plot optimization trajectory 
		if paths is not None: 
			color=iter(plt.cm.rainbow(np.linspace(0,1.0,len(paths))))
			for path in paths: 
				c=next(color)
				trajectory, method = path 
				axs[0].quiver(
					trajectory[0,:-1],
					trajectory[1,:-1],
					trajectory[0,1:]-trajectory[0,:-1],
					trajectory[1,1:]-trajectory[1,:-1], 
					scale_units='xy', 
					angles='xy', 
					scale=1, 
					color=c, 
					label=method
				)

		# Add legend  
		axs[0].legend()

		""" Gradient Field """
		X, Y = np.meshgrid(np.arange(xmin, xmax, delta_x), np.arange(ymin, ymax, delta_y))
		dX, dY = self._grad_2D(X,Y)
		axs[1].quiver(X, Y, -dX, -dY)
		# Set limits 
		axs[1].set_xlim(xmin, xmax)
		axs[1].set_ylim(ymin, ymax)
		# Title 
		axs[1].set_title('Anti-Gradient Field')
		# Labels 
		axs[1].set_xlabel(r'$x_1$')
		axs[1].set_ylabel(r'$x_2$')
		# Plot local/global minima
		for x_star in self.global_optimum: 
			axs[1].plot(*x_star, 'r*', markersize=10)
		# Plot saddle points 
		for x_saddle in self.saddle: 
			axs[1].plot(*x_saddle, 'b+', markersize=10)
		if paths is not None: 
			color=iter(plt.cm.rainbow(np.linspace(0,1.0,len(paths))))
			for path in paths: 
				c=next(color)
				trajectory, method = path 
				axs[1].quiver(
					trajectory[0,:-1],
					trajectory[1,:-1],
					trajectory[0,1:]-trajectory[0,:-1],
					trajectory[1,1:]-trajectory[1,:-1], 
					scale_units='xy', 
					angles='xy', 
					scale=1, 
					color=c, 
					label=method
				)

		""" Gradient Norm """
		X, Y = np.meshgrid(np.arange(xmin, xmax, delta_x), np.arange(ymin, ymax, delta_y))
		grad_norm = self._grad_norm_2D(X,Y)
		axs[2].contour(X, Y, grad_norm)
		# Set limits 
		axs[2].set_xlim(xmin, xmax)
		axs[2].set_ylim(ymin, ymax)
		# Title 
		axs[2].set_title('Gradient Norm')
		# Labels 
		axs[2].set_xlabel(r'$x_1$')
		axs[2].set_ylabel(r'$x_2$')
		# Plot local/global minima
		for x_star in self.global_optimum: 
			axs[2].plot(*x_star, 'r*', markersize=10)
		# Plot saddle points 
		for x_saddle in self.saddle: 
			axs[2].plot(*x_saddle, 'b+', markersize=10)
		if paths is not None: 
			color=iter(plt.cm.rainbow(np.linspace(0,1.0,len(paths))))
			for path in paths: 
				c=next(color)
				trajectory, method = path 
				axs[2].quiver(
					trajectory[0,:-1],
					trajectory[1,:-1],
					trajectory[0,1:]-trajectory[0,:-1],
					trajectory[1,1:]-trajectory[1,:-1], 
					scale_units='xy', 
					angles='xy', 
					scale=1, 
					color=c, 
					label=method
				)
		  
		if save_name is not None: 
			plt.savefig(save_name+'.pdf')

class SaddleBench(Benchmark): 
	def __init__(self, d=2):
		Benchmark.__init__(self, d)

		self._bounds = list(zip([-3.0] * self.d, [3.0] * self.d))

		# Stationary points 
		self.fglob = - (self.d-1)/4
		self.saddle = [np.zeros(self.d)]
		self.global_optimum = [np.ones(self.d), -np.ones(self.d)]

	def eval(self, x): 
		return np.sum(0.25*np.power(x[:-1],4)) - np.sum(x[:-1])*x[-1] + 0.5*(self.d-1)*np.power(x[-1],2)

	def gradient(self,x): 
		g = np.array([x_i**3 - x[-1] for x_i in x])
		g[-1] = (self.d-1)*x[-1] -np.sum(x[:-1])

		return g

	def H(self, x): 
		# Evaluate the hessian 
		h = np.diag(np.append(3*np.power(x[:-1],2), self.d-1))
		h[-1,:-1] = -1 
		h[:-1,-1] = -1 
		return h

	def _eval_2D(self, x, y): 
		return 0.25*x**4.0 - x*y + 0.5*y**2.0

	def _grad_2D(self, x, y): 
		dx = x**3.0 - y
		dy = y - x
		return (dx, dy)

	def _grad_norm_2D(self, x, y): 
		dx = x**3.0 - y
		dy = y - x
		return np.sqrt(dx**2 + dy**2)

class Rastrigin(Benchmark):

	def __init__(self, d=2):
		Benchmark.__init__(self, d)
		self._bounds = list(zip([-5.0] * self.d, [5.0] * self.d))

		# Stationary points 
		self.fglob = 0.0
		self.global_optimum = [np.array([0 for _ in range(self.d)])]

		# Initialize gradient function 
		self.grad_func = grad(self.eval)

		# Initialize hessian function 
		self.hess_func = hessian(self.eval)

	def eval(self, x, *args):
		return 10.0 * self.d + np.sum(x ** 2.0 - 10.0 * np.cos(2.0 * np.pi * x))

	def gradient(self,x): 
		return self.grad_func(x)

	def H(self,x): 
		return self.hess_func(x)


class LeadingEigenvector(Benchmark): 
	def __init__(self, d=2, M=None):
		Benchmark.__init__(self, d)

		self._bounds = list(zip([-5.0] * self.d, [5.0] * self.d))

		if M is None: 
			self.M = np.random.random(size=(self.d,self.d))
			self.M = np.dot(self.M, np.transpose(self.M))
		else: 
			self.M = M

		# Eigenvalue decomposition 
		L, V = np.linalg.eig(self.M)

		i_star = np.argmax(L)

		L_star = L[i_star]
		V_star = V[:,i_star]

		# Global Minima
		self.global_optimum = [np.sqrt(L_star)*V_star, -np.sqrt(L_star)*V_star]
		self.fglob = self.eval(self.global_optimum[0])

		# Saddle Points 
		i_saddle = np.array(list(set(np.arange(self.d)) - set(np.array([i_star]))))

		self.saddle = []
		for i in i_saddle: 
			self.saddle.append(+np.sqrt(L[i])*V[:,i])
			self.saddle.append(-np.sqrt(L[i])*V[:,i])

	def eval(self, x): 
		xx_T = np.dot(np.reshape(x,(self.d, 1)), np.reshape(x, (1, self.d)))
		return 0.5*(np.linalg.norm(xx_T - self.M, ord='fro'))**2.0

	def gradient(self,x): 
		xx_T = np.dot(np.reshape(x,(self.d, 1)), np.reshape(x, (1, self.d)))
		return np.dot(xx_T - self.M, x)

	def H(self, x): 
		# Evaluate the hessian 
		xx_T = np.dot(np.reshape(x,(self.d, 1)), np.reshape(x, (1, self.d)))

		return (np.linalg.norm(x)**2.0)*np.identity(self.d) + 2*xx_T - self.M