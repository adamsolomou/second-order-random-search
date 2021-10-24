import autograd.numpy as np

def uniform_angles_pss(dim): 
	# Define A
	A = np.empty((dim,dim))
	for i in range(dim): 
		for j in range(dim): 
			if i==j: 
				A[i,j] = 1
			else:
				A[i,j] = - 1/dim

	# Find C such that A = CC^T
	C = np.linalg.cholesky(A)

	# V = C^T
	V = np.transpose(C)

	v = -np.sum(V, axis=1)
	v = np.reshape(v, (dim,1))

	# Minimal PSS
	D = np.hstack((V,v))

	# Size of PSS
	size = D.shape[1]

	# Flag indicating if PSS is symmetric or not 
	symmetric = False

	return D, symmetric, size 

