module tensor

fn test_autograd() {
	println('Testing Tensor Autograd...')

	// Create two tensors
	// a = 2.0, requires_grad = true
	a := new_tensor([f32(2.0)], [1], true)
	// b = 3.0, requires_grad = true
	b := new_tensor([f32(3.0)], [1], true)

	// c = a * b = 6.0
	c := a.mul(b)
	assert c.data[0] == 6.0

	// d = c + a = 6.0 + 2.0 = 8.0
	mut d := c.add(a)
	assert d.data[0] == 8.0

	// Backward
	// d = a*b + a
	// dd/da = b + 1 = 3 + 1 = 4
	// dd/db = a = 2
	d.backward()

	// Check gradients
	// Note: we need to handle the unsafe nil check or assume it's set
	if unsafe { a.grad != nil } {
		assert a.grad.data[0] == 4.0
	} else {
		assert false, 'a.grad should not be nil'
	}

	if unsafe { b.grad != nil } {
		assert b.grad.data[0] == 2.0
	} else {
		assert false, 'b.grad should not be nil'
	}
}
