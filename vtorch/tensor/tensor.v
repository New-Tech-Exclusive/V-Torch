module tensor

// Tensor represents a multi-dimensional array with automatic differentiation support.
// It is the core building block of the v-torch library.
@[heap]
pub struct Tensor {
pub mut:
	// data holds the flattened elements of the tensor.
	data []f32
	// shape describes the dimensions of the tensor (e.g., [2, 3] for a 2x3 matrix).
	shape []int
	// strides allows for efficient indexing into the flat data array.
	strides []int

	// requires_grad indicates if gradients should be computed for this tensor.
	requires_grad bool
	// grad holds the gradient of this tensor (if requires_grad is true).
	grad &Tensor = unsafe { nil }

	// Graph construction for autograd
	// parents are the tensors that were used to create this tensor.
	parents []&Tensor
	// backward_fn is the closure that computes the gradients for the parents.
	backward_fn fn (t &Tensor) = unsafe { nil }
}

// new_tensor creates a new Tensor from the given data and shape.
// It automatically computes the strides.
pub fn new_tensor(data []f32, shape []int, requires_grad bool) &Tensor {
	return &Tensor{
		data: data
		shape: shape
		strides: compute_strides(shape)
		requires_grad: requires_grad
	}
}

// zeros creates a new Tensor of the given shape filled with zeros.
pub fn zeros(shape []int, requires_grad bool) &Tensor {
	size := size_from_shape(shape)
	return new_tensor([]f32{len: size, init: 0.0}, shape, requires_grad)
}

// ones creates a new Tensor of the given shape filled with ones.
pub fn ones(shape []int, requires_grad bool) &Tensor {
	size := size_from_shape(shape)
	return new_tensor([]f32{len: size, init: 1.0}, shape, requires_grad)
}

// compute_strides calculates the strides for a given shape.
// Strides are used to map multi-dimensional indices to a flat 1D index.
fn compute_strides(shape []int) []int {
	mut s := []int{len: shape.len}
	mut acc := 1
	for i := shape.len - 1; i >= 0; i-- {
		s[i] = acc
		acc *= shape[i]
	}
	return s
}

// size_from_shape calculates the total number of elements from the shape.
fn size_from_shape(shape []int) int {
	mut size := 1
	for dim in shape {
		size *= dim
	}
	return size
}

// backward triggers the backpropagation process starting from this tensor.
// It computes the gradients for all tensors in the computation graph.
pub fn (mut t Tensor) backward() {
	// 1. Initialize gradient for the starting tensor if not exists (usually 1.0)
	if unsafe { t.grad == nil } {
		t.grad = ones(t.shape, false)
	}

	// 2. Topological sort (simplified: just recursive for now or list based)
	// For a robust implementation, we need a proper topological sort.
	// Here we use a simple recursive approach for demonstration.
	t.execute_backward()
}

// execute_backward recursively calls the backward function.
// Note: This is a naive implementation. A proper engine would use a work queue.
fn (mut t Tensor) execute_backward() {
	if unsafe { t.backward_fn != nil } {
		t.backward_fn(t)
	}
	for mut parent in t.parents {
		parent.execute_backward()
	}
}

// add performs element-wise addition of two tensors.
pub fn (a &Tensor) add(b &Tensor) &Tensor {
	// TODO: Add broadcasting support
	if a.data.len != b.data.len {
		panic('Shapes must match for now')
	}

	mut new_data := []f32{len: a.data.len}
	for i in 0 .. a.data.len {
		new_data[i] = a.data[i] + b.data[i]
	}

	mut res := new_tensor(new_data, a.shape, a.requires_grad || b.requires_grad)
	
	if res.requires_grad {
		res.parents = [a, b]
		res.backward_fn = fn [a, b] (t &Tensor) {
			// Gradient for addition is passed directly to parents
			if a.requires_grad {
				// a.grad += t.grad
				accumulate_grad(a, t.grad)
			}
			if b.requires_grad {
				// b.grad += t.grad
				accumulate_grad(b, t.grad)
			}
		}
	}
	return res
}

// mul performs element-wise multiplication of two tensors.
pub fn (a &Tensor) mul(b &Tensor) &Tensor {
	if a.data.len != b.data.len {
		panic('Shapes must match for now')
	}

	mut new_data := []f32{len: a.data.len}
	for i in 0 .. a.data.len {
		new_data[i] = a.data[i] * b.data[i]
	}

	mut res := new_tensor(new_data, a.shape, a.requires_grad || b.requires_grad)

	if res.requires_grad {
		res.parents = [a, b]
		res.backward_fn = fn [a, b] (t &Tensor) {
			// Product rule: d(ab)/da = b, d(ab)/db = a
			if a.requires_grad {
				// grad_a = grad_output * b
				grad_a := t.grad.mul_no_grad(b) 
				accumulate_grad(a, grad_a)
			}
			if b.requires_grad {
				// grad_b = grad_output * a
				grad_b := t.grad.mul_no_grad(a)
				accumulate_grad(b, grad_b)
			}
		}
	}
	return res
}

// mul_no_grad is a helper for multiplication without tracking gradients (to avoid infinite loops in backward).
fn (a &Tensor) mul_no_grad(b &Tensor) &Tensor {
	mut new_data := []f32{len: a.data.len}
	for i in 0 .. a.data.len {
		new_data[i] = a.data[i] * b.data[i]
	}
	return new_tensor(new_data, a.shape, false)
}

// accumulate_grad adds the given gradient to the tensor's existing gradient.
fn accumulate_grad(t &Tensor, g &Tensor) {
	if unsafe { t.grad == nil } {
		mut t_mut := unsafe { &Tensor(t) } // Cast to mutable to set initial grad
		t_mut.grad = g
	} else {
		// t.grad += g
		// We need to implement in-place add or similar
		// For now, let's just replace it (inefficient but works for logic)
		mut t_mut := unsafe { &Tensor(t) }
		t_mut.grad = t.grad.add_no_grad(g)
	}
}

// add_no_grad helper
fn (a &Tensor) add_no_grad(b &Tensor) &Tensor {
	mut new_data := []f32{len: a.data.len}
	for i in 0 .. a.data.len {
		new_data[i] = a.data[i] + b.data[i]
	}
	return new_tensor(new_data, a.shape, false)
}

// str provides a string representation of the tensor.
pub fn (t &Tensor) str() string {
	return 'Tensor(shape=${t.shape}, data=${t.data})'
}